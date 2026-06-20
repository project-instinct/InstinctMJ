"""The implementation for reading and returning motion reference data.

FAQ:
    - How to use this module in AMP/WASABI implementation?
        * Use the motion reference to build discriminator observation and feed to the algorithm
        * Use the discriminator observation and discriminator (in algo) to label the discrimintaor
        reward.
"""

from __future__ import annotations

from .motion_reference_cfg import MotionReferenceManagerCfg, NoCollisionPropertiesCfg
from .motion_reference_data import MotionReferenceData, MotionReferenceState, MotionSequence
from .motion_reference_hoi_data import HoiMotionReferenceData, HoiMotionReferenceState, HoiMotionSequence
from .motion_reference_manager import MotionReferenceManager

__all__ = [name for name in globals() if not name.startswith("_")]
