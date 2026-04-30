# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Reference 'keyholes' — synthetic calibration targets for framework validation.

These are NOT part of the generic library surface. They exist to exercise
the Omega-Lock pipeline end-to-end with known hidden structure, so the
calibrator's discovery can be verified against ground truth.

Public:
    from omega_lock.keyholes.phantom import PhantomKeyhole
"""
from omega_lock.keyholes.phantom import PhantomKeyhole
from omega_lock.keyholes.phantom_deep import PhantomKeyholeDeep

__all__ = ["PhantomKeyhole", "PhantomKeyholeDeep"]
