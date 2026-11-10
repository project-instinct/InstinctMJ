"""WM-only terrain/contact representations. No changes to observations or ACP rewards.

The first linear layer retains the old contiguous weight and evaluates new zero columns
as a separate GEMM, preserving the old forward function at warm start (including bf16).
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F
from instinct_mj.rl.supermimic.networks import WorldModel
from instinct_mj.rl.supermimic.rotations import yaw_mat,mm,mv
from .rays import ScanFloorRunningNorm,_ray_cast_normal

DIMS={'patch':392,'point':54,'ar':9,'targets':36}

def parse(spec):
    parts=tuple(x for x in spec.split(',') if x) if isinstance(spec,str) else tuple(spec)
    if len(parts)!=len(set(parts)) or set(parts)-DIMS.keys():raise ValueError(f'invalid WM contact inputs: {spec}')
    return parts

class ExtraLinear(nn.Linear):
    def __init__(self,old_dim,extra,out):
        super().__init__(old_dim,out)
        self.extra_weight=nn.Parameter(torch.zeros(out,extra))
    def forward(self,x):
        return F.linear(x[...,:self.in_features],self.weight,self.bias)+F.linear(x[...,self.in_features:],self.extra_weight)

class ContactRunningNorm(ScanFloorRunningNorm):
    """Keep probability feedback on its physical [-1,1] scale, including rare contacts."""
    def __init__(self,dim,scan_slice,ar_slice=None):
        super().__init__(dim,scan_slice);self.ar_slice=ar_slice
    @torch.no_grad()
    def update(self,x):
        super().update(x)
        if self.ar_slice is not None:
            self.mean[self.ar_slice]=0
            self.var[self.ar_slice]=1

class ContactInputWorldModel(WorldModel):
    def __init__(self,context_dim,num_contact,xi_dim,*,spec,num_joints=23,no_scan=False,bridge_updates=100):
        self.contact_spec=parse(spec);self.extra_dim=sum(DIMS[p] for p in self.contact_spec)
        super().__init__(context_dim+self.extra_dim,num_contact,xi_dim,0,num_joints=num_joints)
        self.old_in_dim=self.in_dim-self.extra_dim
        self.net[0]=ExtraLinear(self.old_in_dim,self.extra_dim,self.net[0].out_features)
        scan_start=self.old_in_dim-context_dim+1  # the context starts with the pelvis height
        self.scan_slice=slice(scan_start,scan_start+187)
        ar_slice=None
        if 'ar' in self.contact_spec:
            offset=self.old_in_dim+sum(DIMS[x] for x in self.contact_spec[:self.contact_spec.index('ar')])
            ar_slice=slice(offset,offset+9)
        self.in_norm=ContactRunningNorm(self.in_dim,self.scan_slice,ar_slice)
        self.no_scan=no_scan;self.bridge_updates=bridge_updates
        self.register_buffer('contact_updates',torch.zeros((),dtype=torch.long))
        self.register_buffer('scan_blend',torch.zeros(()))
        self.use_scan=True;self._updates=0

    def load_compatible(self,state):
        if 'net.0.extra_weight' in state:
            self.load_state_dict(state)
        else:
            merged=self.state_dict()
            assert state['in_norm.mean'].numel()==self.old_in_dim,'only append input columns to the same base WM'
            for key,val in state.items():
                if val.shape==merged[key].shape:merged[key]=val
                elif key.startswith('in_norm.') and val.ndim==1:
                    merged[key]=merged[key].clone();merged[key][:self.old_in_dim]=val
                else:raise ValueError(f'cannot migrate {key}: {val.shape} -> {merged[key].shape}')
            self.load_state_dict(merged)
        self._updates=int(self.contact_updates)
        self.use_scan=not self.no_scan or self._updates<self.bridge_updates

    def load_optimizer(self,opt,state):
        named=list(self.named_parameters());ids=opt.state_dict()['param_groups'][0]['params']
        if len(state['param_groups'][0]['params'])==len(named):opt.load_state_dict(state);return
        old_ids=state['param_groups'][0]['params'];old_named=[(i,p) for i,(n,p) in zip(ids,named) if n!='net.0.extra_weight']
        assert len(old_ids)==len(old_named)
        mapped={newid:copy.deepcopy(state['state'][oldid]) for oldid,(newid,p) in zip(old_ids,old_named) if oldid in state['state']}
        ref=next(iter(mapped.values()),None)
        if ref is not None:
            for newid,(name,p) in zip(ids,named):
                if name=='net.0.extra_weight':
                    mapped[newid]=dict(step=torch.zeros_like(ref['step']),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        groups=copy.deepcopy(state['param_groups']);groups[0]['params']=ids
        opt.load_state_dict(dict(state=mapped,param_groups=groups))
        for v in opt.state.values():v['step']=v['step'].cpu()

    def advance(self):
        self._updates+=1;self.contact_updates.fill_(self._updates)
        if self.no_scan:
            self.scan_blend.fill_(min(1.,self._updates/max(1,self.bridge_updates)))
            self.use_scan=self._updates<self.bridge_updates

    def coarse_scan(self,mdp,S):
        mean=self.in_norm.mean[self.scan_slice]
        if not self.use_scan:return mean.expand(S[0].shape[0],-1)
        scan=mdp.scan(S[0],S[1])
        return scan if not self.no_scan else scan*(1-self.scan_blend)+mean*self.scan_blend

class ContactInputs:
    """Features are recomputed on predicted states, and use fixed physical scales."""
    def __init__(self,mdp,kin,spec,env=None):
        self.mdp,self.kin,self.spec=mdp,kin,parse(spec)
        self.point_query=None
        if 'point' in self.spec:
            raise NotImplementedError('Exact 9-point terrain queries are deferred.')
        grid=torch.linspace(-.06,.06,7,device=mdp.xi_mean.device)
        xx,yy=torch.meshgrid(grid,grid,indexing='xy')
        self.offset=torch.stack((xx.flatten(),yy.flatten()),-1)
        self.ends=[3,4,7,8]

    def features(self,S,a,a_prev,xi_n,contact,cmd):
        out=[];fk=None
        for part in self.spec:
            if part=='ar':
                if contact is None:raise ValueError('AR requires a contact initial state')
                out.append((contact-.5)*2)
            elif part=='targets':
                # the contact command the tracker's observation actually carries (27: reference with noise, the planner's, or the per-env mix; invalid entries are 0)
                if cmd is None:raise ValueError('targets require the contact command of the observation')
                valid=(cmd.view(-1,9,3)!=0).any(-1)
                out.append(torch.cat((cmd/.5,valid.float()),-1))
            else:
                if fk is None:fk=self.kin.forward(*S)
                bp,bR,bv,bw=fk
                if part=='patch':
                    p=bp[:,self.ends];R=bR[:,self.ends]
                    xy=p[...,None,:2]+mm(self.offset,yaw_mat(R)[...,:2,:2].transpose(-1,-2))
                    origins=torch.cat((xy,(p[...,None,2:3]+.25).expand(*xy.shape[:-1],1)),-1)
                    dist,norm=_ray_cast_normal(origins.reshape(S[0].shape[0],-1,3).detach(),self.mdp.scanner.scanner_id)
                    dist=dist.view(*p.shape[:2],49);normal=norm.view(*p.shape[:2],49,3)
                    valid=(dist>=0)&(dist<.85)
                    ground=(origins[...,2]-dist).detach()
                    nz=normal[...,2];slope=torch.where((nz.abs()>.2)[...,None],-normal[...,:2]/torch.where(nz.abs()>.2,nz,torch.ones_like(nz))[...,None],0.)
                    ground=ground+(slope*(xy-xy.detach())).sum(-1)
                    height=torch.where(valid,(p[...,2,None]-ground)/.2,torch.zeros_like(dist))
                    out.append(torch.cat((height.flatten(1),valid.float().flatten(1)),-1))
                elif part=='point':
                    distance,normal,hit=self.point_query.query(bp)
                    # Exact point distance; a contact-link origin is not its whole collision body.
                    nlocal=mv(yaw_mat(S[1]).transpose(-1,-2)[:,None],normal)
                    vn=(bv*normal).sum(-1)
                    out.append(torch.cat((distance[...,None]/.1,nlocal,vn[...,None]/.5,hit[...,None]),-1).flatten(1))
        return torch.cat(out,-1).clamp(-8,8)
