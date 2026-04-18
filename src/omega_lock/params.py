"""Parameter container + clip/perturbation helpers.

Generic successor to research/omega_lock_p1/params.py. The 21 HeartCore-specific
defaults are gone; parameters are now driven entirely by a target's param_space().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omega_lock.target import ParamSpec


def clip(spec: ParamSpec, value: Any) -> Any:
    """Clip a value to the spec's valid range, respecting dtype."""
    if spec.dtype == "bool":
        return bool(value)
    if spec.dtype == "int":
        return max(spec.low, min(spec.high, int(round(value))))
    return max(spec.low, min(spec.high, float(value)))


def default_epsilon(spec: ParamSpec) -> float:
    """Perturbation ε for stress measurement.

    - Continuous: 10% of range
    - Integer: 1 (smallest meaningful step)
    - Boolean: 0.0 (caller flips explicitly, ε is unused)
    """
    if spec.dtype == "bool":
        return 0.0
    if spec.dtype == "int":
        return 1.0
    return (float(spec.high) - float(spec.low)) * 0.1


def neutral_defaults(specs: list[ParamSpec]) -> dict[str, Any]:
    """Collect neutral values from specs into a params dict."""
    return {s.name: s.neutral for s in specs}


@dataclass
class LockedParams:
    """Lock/unlock state container.

    Philosophy ("열쇠구멍을 거푸집으로 쓴다"):
        All params default to locked. Unlock only the ones you want to
        search over. Perturbation respects current values + type + range.
    """
    specs: dict[str, ParamSpec]
    values: dict[str, Any] = field(default_factory=dict)
    locked: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_specs(cls, specs: list[ParamSpec]) -> "LockedParams":
        specs_by_name = {s.name: s for s in specs}
        return cls(
            specs=specs_by_name,
            values=neutral_defaults(specs),
            locked={s.name: True for s in specs},
        )

    def clone(self) -> "LockedParams":
        return LockedParams(
            specs=dict(self.specs),
            values=dict(self.values),
            locked=dict(self.locked),
        )

    def unlock(self, *names: str) -> None:
        for n in names:
            self._check(n)
            self.locked[n] = False

    def lock(self, *names: str) -> None:
        for n in names:
            self._check(n)
            self.locked[n] = True

    def unlocked_names(self) -> tuple[str, ...]:
        return tuple(n for n, spec in self.specs.items() if not self.locked[n])

    def set_value(self, name: str, value: Any) -> None:
        self._check(name)
        self.values[name] = clip(self.specs[name], value)

    def perturbed(self, name: str, delta: float) -> "LockedParams":
        """Return a clone with `name` perturbed by `delta` (flip for bool)."""
        self._check(name)
        c = self.clone()
        spec = self.specs[name]
        if spec.dtype == "bool":
            c.values[name] = not bool(c.values[name])
            return c
        c.values[name] = clip(spec, c.values[name] + delta)
        return c

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)

    def _check(self, name: str) -> None:
        if name not in self.specs:
            raise KeyError(f"unknown param: {name}")
