from __future__ import annotations

import os
from typing import TYPE_CHECKING

import joblib
import torch
import yaml

from instinct_mj.motion_reference import MotionSequence
from instinct_mj.motion_reference.motion_files.amass_motion import AmassMotion

if TYPE_CHECKING:
    from .aistpp_motion_cfg import AistppMotionCfg


class AistppMotion(AmassMotion):
    """Processing AIST++ motion files."""

    cfg: AistppMotionCfg

    """
    Internal helper functions
    """

    def _refresh_motion_file_list(self):
        """Refresh the motion file from list."""
        assert self.cfg.path.endswith("motions"), "The path should end with 'motions'."
        self._all_motion_files: list[str] = [os.path.join(self.cfg.path, f) for f in os.listdir(self.cfg.path)]
        if self.cfg.filtered_motion_selection_filepath:
            with open(self.cfg.filtered_motion_selection_filepath) as f:
                filtered_files = yaml.safe_load(f)
            self._all_motion_files = [os.path.join(self.cfg.path, f) for f in filtered_files]
        print(f"[AIST++ Motion] {len(self._all_motion_files)} motion files found.")
        assert len(self._all_motion_files) > 0, (
            "No motion files found. Please check the path."
            "The supported file are shown in the source code of this function."
        )

    def _read_motion_file(self, motion_file_idx) -> MotionSequence:
        """Read a motion file and return the motion sequence."""
        raw_data = joblib.load(self._all_motion_files[motion_file_idx])
        poses = torch.from_numpy(raw_data["smpl_poses"]).to(self.buffer_device)
        root_trans = torch.from_numpy(raw_data["smpl_trans"]).to(self.buffer_device)
        framerate = self.cfg.assumed_file_framerate
        return self._build_motion_sequence_from_smpl(poses, root_trans, framerate)
