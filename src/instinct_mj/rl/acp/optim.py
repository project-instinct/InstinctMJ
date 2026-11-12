"""Optimizers and the captured world-model update of the official HybridRobotics/supermimic (``learners/learner_utils.py``).

:class:`MuonAdam` (official ``build_optimizers``): Muon (``adjust_lr_fn="match_rms_adamw"``) on the Linear weights between the first and the last Linear of every
``nn.Sequential`` of the modules, fused capturable Adam (``adam_cls``: Adam or AdamW) on the rest, both at the same learning rate. The pair looks like one optimizer to the
runner (``param_groups``, ``state``, ``zero_grad``, ``step``, ``state_dict``). The ACP runner uses it for the world model, the actor and every critic.

:class:`CapturedUpdate`: runs a gradient step eagerly for a few calls, then records it once as a CUDA graph and replays it (official ``CapturedUpdate``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn


def hidden_linear_weights(modules: Sequence[nn.Module]) -> list[torch.Tensor]:
    """The Linear weights between the first and the last Linear of every ``nn.Sequential`` of ``modules`` (official ``build_optimizers``)."""
    out, seen = [], set()
    for module in modules:
        for seq in module.modules():
            if not isinstance(seq, nn.Sequential):
                continue
            for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]:
                if id(linear.weight) not in seen and linear.weight.requires_grad:
                    seen.add(id(linear.weight))
                    out.append(linear.weight)
    return out


def set_lr(opt, lr: float) -> None:
    """Set the learning rate of every group of ``opt``: in place when it is a device tensor (a CUDA graph that read it replays with the new value; no host sync), else assigned
    (a Python float is baked into a recorded CUDA graph: see :class:`MuonAdam`)."""
    for g in opt.param_groups:
        if torch.is_tensor(g["lr"]):
            g["lr"].fill_(lr)
        else:
            g["lr"] = lr


class MuonAdam:
    """Muon on ``hidden`` + fused capturable Adam(W) on the other parameters, presented as one optimizer. ``fused=True`` and ``capturable=True`` (the Adam step counters live
    on the device: no host sync per step, and the step can be recorded in a CUDA graph) as the official ``build_optimizers``; on CPU parameters (unit tests) only ``fused``.

    Learning rate (official ``build_optimizers``): CONSTANT. The Adam(W) group's ``lr`` is a 0-dim float32 device tensor (what the official passes to its fused capturable Adam), the Muon
    group's ``lr`` the Python float: ``torch.optim.Muon`` (torch 2.10) converts it with ``_to_scalar`` (a host sync) and a CUDA graph bakes it in at the recording. As the learning rate
    never changes, a recorded graph applies the same learning rate as the eager update. ``load_state_dict`` keeps the constructor's learning rate (not the one stored in the checkpoint),
    and the Adam tensor object (its address). The format is that of the previous MuonAdam."""

    def __init__(self, modules: Sequence[nn.Module], lr: float, betas=(0.9, 0.999), adam_cls=torch.optim.Adam, **adam_kwargs):
        self.lr = lr
        hidden = hidden_linear_weights(modules)
        hidden_ids = {id(p) for p in hidden}
        rest, seen = [], set()
        for m in modules:
            for p in m.parameters():
                if id(p) not in hidden_ids and id(p) not in seen:
                    seen.add(id(p))
                    rest.append(p)
        on_cuda = all(p.is_cuda for p in rest)
        dev = rest[0].device if rest else None
        self.adam = adam_cls(rest, lr=torch.tensor(lr, dtype=torch.float32, device=dev) if on_cuda else lr, betas=betas, fused=True, **({"capturable": True} if on_cuda else {}), **adam_kwargs)
        self.muon = torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw") if hidden else None  # (Python-float lr: see the class docstring)
        self.optimizers = [self.adam] + ([self.muon] if self.muon is not None else [])

    @property
    def param_groups(self) -> list[dict]:
        return [g for o in self.optimizers for g in o.param_groups]

    @property
    def state(self) -> dict:
        out = {}
        for o in self.optimizers:
            out.update(o.state)
        return out

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for o in self.optimizers:
            o.step()

    def state_dict(self) -> dict:
        def plain(o):
            sd = o.state_dict()
            for g in sd["param_groups"]:
                if torch.is_tensor(g.get("lr")):
                    g["lr"] = float(g["lr"])
            return sd

        return {"muon_adam": True, "adam": plain(self.adam), "muon": plain(self.muon) if self.muon is not None else None}

    def load_state_dict(self, state: dict) -> None:
        if not state.get("muon_adam", False):
            raise ValueError("optimizer state of a plain Adam run: start the Muon optimizer fresh (do not resume across the optimizer switch)")
        for o, sd in ((self.adam, state["adam"]), (self.muon, state["muon"])):
            if o is None:
                continue
            tensors = [g["lr"] for g in o.param_groups if torch.is_tensor(g["lr"])]  # (torch replaces the groups by the saved ones, their lr included: keep the tensor objects, a CUDA graph reads them)
            o.load_state_dict(sd)
            for g, t in zip(o.param_groups, tensors):
                g["lr"] = t
            set_lr(o, self.lr)  # the constructor's learning rate is the constant one


def graph_state(modules: Sequence[nn.Module] = (), optimizers: Sequence = (), skip_buffers: Sequence[str] = ("count",), grad_modules: Sequence[nn.Module] = ()) -> dict[str, torch.Tensor]:
    """The tensors a captured update reads through Python objects and that must keep their address: the parameters and buffers of ``modules``, the tensor state of
    ``optimizers`` (``state`` / ``param_groups``: moments, step counters, Muon momentum) and tensor learning rates. Buffers named in ``skip_buffers`` are left out: a
    running normalizer rebinds its ``count`` on every update but its forward does not read it, so checking it would fire on a healthy run.

    ``grad_modules``: the modules whose parameters the captured update OPTIMIZES (zero_grad(set_to_none=False), backward, clip and step are recorded on their ``.grad`` storage):
    the ``.grad`` tensors that exist at the call are registered too (a parameter without a gradient is not read by the graph and is left out; one that loses its gradient later is
    reported as vanished). Modules that the graph only reads (a frozen planner whose trainer sets its own gradients to None every step) do not belong here."""
    out: dict[str, torch.Tensor] = {}
    for i, m in enumerate(modules):
        for name, t in m.named_parameters():
            out[f"module{i}.{name}"] = t
        for name, t in m.named_buffers():
            if name.rsplit(".", 1)[-1] not in skip_buffers:
                out[f"module{i}.{name}"] = t
    for i, m in enumerate(grad_modules):
        for name, t in m.named_parameters():
            if t.grad is not None:
                out[f"grad{i}.{name}"] = t.grad
    for i, opt in enumerate(optimizers):
        state = opt.state  # (MuonAdam builds this dict on every access)
        for j, p in enumerate(p for g in opt.param_groups for p in g["params"]):
            for key, t in state.get(p, {}).items():
                if torch.is_tensor(t):
                    out[f"opt{i}.param{j}.{key}"] = t
        for j, g in enumerate(opt.param_groups):
            if torch.is_tensor(g.get("lr")):
                out[f"opt{i}.group{j}.lr"] = g["lr"]
    return out


class CapturedUpdate:
    """``fn(inputs: dict[str, Tensor]) -> dict[str, Tensor]`` eagerly for the first ``warmup`` calls (on a side stream), then captured once as a CUDA graph and
    replayed. ``fn`` must be capture-safe: no host syncs, fixed shapes, every tensor it reads besides its inputs updated in place. Python values it reads (e.g. a
    Muon learning rate) are fixed at capture: call ``invalidate`` when they change. Each replay overwrites the returned tensors.

    ``state`` (optional callable -> dict name -> tensor, or a sequence of tensors; see :func:`graph_state`): the tensors the graph reads that Python could rebind. Their
    addresses are recorded at the capture and checked at every replay (official ``CapturedUpdate``): a moved tensor raises instead of being silently read at its old address. The
    registered tensors are kept referenced until the graph is invalidated.

    The update is bound to the CUDA device of its inputs: the side stream lives there and warm-up, capture and replay run under ``torch.cuda.device``."""

    def __init__(self, fn: Callable[[dict], dict], warmup: int = 3, state: Callable[[], dict | Sequence[torch.Tensor]] | None = None):
        self.fn, self.warmup_left, self.warmup = fn, warmup, warmup
        self.state = state
        self.inputs: dict[str, torch.Tensor] | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.outputs: dict[str, torch.Tensor] = {}
        self.device: torch.device | None = None
        self.stream: torch.cuda.Stream | None = None
        self.state_ptrs: dict[str, int] = {}
        self.state_refs: dict[str, torch.Tensor] = {}
        self.num_captures = 0

    def invalidate(self) -> None:
        """Re-record at the next call (after one eager warm-up call)."""
        self.graph, self.outputs, self.warmup_left = None, {}, max(self.warmup_left, 1)
        self.state_refs = {}

    def _bind(self, inputs: dict[str, torch.Tensor]) -> None:
        dev = next(iter(inputs.values())).device
        if dev.type != "cuda":
            raise ValueError(f"CapturedUpdate needs CUDA inputs, got {dev}")
        if any(v.device != dev for v in inputs.values()):
            raise ValueError(f"CapturedUpdate inputs are on different devices: {sorted({str(v.device) for v in inputs.values()})}")
        if self.device != dev:
            self.device, self.stream = dev, torch.cuda.Stream(device=dev)
            self.graph, self.outputs, self.inputs = None, {}, None

    def _state_tensors(self) -> dict[str, torch.Tensor]:
        if self.state is None:
            return {}
        st = self.state()
        return dict(st) if isinstance(st, dict) else {f"state[{i}]": t for i, t in enumerate(st)}

    def _record_state(self) -> None:
        tensors = self._state_tensors()
        wrong = [n for n, t in tensors.items() if t.device != self.device]
        if wrong:
            raise ValueError(f"CapturedUpdate state tensors are not on the device of the inputs ({self.device}): {wrong[:5]}")
        self.state_ptrs = {n: t.data_ptr() for n, t in tensors.items()}
        self.state_refs = tensors  # (keeps the registered tensors, the gradients included, alive: the graph's memory must not be freed and reused while it can be replayed)

    def _check_state(self) -> None:
        if self.state is None:
            return
        tensors = self._state_tensors()
        moved = [n for n, t in tensors.items() if self.state_ptrs.get(n) != t.data_ptr()]
        moved += [n for n in self.state_ptrs if n not in tensors]
        if moved:
            raise RuntimeError(f"CapturedUpdate: {len(moved)} tensor(s) the CUDA graph reads changed address (or disappeared) since the capture, e.g. {moved[:5]} "
                               "(a load_state_dict / .to() / rebinding after the capture, or a .grad that was set to None / replaced by an eager update or zero_grad(set_to_none=True)): "
                               "the replay would read the old memory. Call invalidate() after such a change.")

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._bind(inputs)
        with torch.cuda.device(self.device):
            return self._call(inputs)

    def _call(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.inputs is None or any(self.inputs[k].shape != v.shape for k, v in inputs.items()):
            self.inputs = {k: v.clone() for k, v in inputs.items()}
            self.graph, self.warmup_left = None, max(self.warmup_left, self.warmup if self.num_captures == 0 else 1)
        else:
            for k, v in inputs.items():
                self.inputs[k].copy_(v)
        if self.warmup_left > 0:
            self.warmup_left -= 1
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                out = self.fn(self.inputs)
            torch.cuda.current_stream().wait_stream(self.stream)
            return out
        if self.graph is None:
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            # cuBLAS keeps a persistent workspace per (handle, stream), allocated outside any graph pool. torch.compile's cudagraph trees
            # ("reduce-overhead") free ALL cuBLAS workspaces before and after each warm-up / recording (cudagraph_trees.clear_cublass_cache),
            # so a graph that captured matmuls on a pre-existing workspace replays into freed (and reused) memory: illegal memory access.
            # Clearing here makes the capture allocate its workspace inside this graph's private pool, owned by the graph (as cudagraph trees do).
            # (Never capture this inside an outer capture with a reduce-overhead function: the compiled functions must not be "reduce-overhead" here.)
            torch._C._cuda_clearCublasWorkspaces()
            try:
                with torch.cuda.graph(graph, stream=self.stream):
                    self.outputs = self.fn(self.inputs)
            finally:
                torch._C._cuda_clearCublasWorkspaces()
            self.graph = graph
            self.num_captures += 1
            self._record_state()
        else:
            self._check_state()
        self.graph.replay()
        return self.outputs


@torch.no_grad()
def step_if_finite(opt, ok: torch.Tensor) -> None:
    """``opt.step()`` whose effect is undone on the device where the 0-dim bool tensor ``ok`` is False: the parameters and every state tensor (moments, step counters, Muon
    momentum) are restored. Capture-safe replacement of ``if torch.isfinite(grad_norm): opt.step()`` (no host sync). The optimizer state must already exist (Adam's lazy state
    initialization inside a capture would be re-run on every replay): call it eagerly a few times before capturing."""
    params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    state = opt.state
    missing = [p for p in params if p not in state]
    if missing:
        raise RuntimeError("step_if_finite: the optimizer state does not exist yet (run an eager step before capturing)")
    tensors = params + [t for p in params for t in state[p].values() if torch.is_tensor(t)]
    saved = [t.clone() for t in tensors]
    opt.step()
    for t, old in zip(tensors, saved):
        t.copy_(torch.where(ok, t, old))
