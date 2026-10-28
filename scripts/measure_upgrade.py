"""Deterministic resource/correctness comparison; run outside the source tree."""
import json
import math
import platform
import sys
from concurrent.futures import Executor, Future
from importlib.metadata import version

from omega_lock._parallel import _ordered_eval_map
from omega_lock.walk_forward import pearson_result


class TrackedFuture(Future):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.consumed = False

    def result(self, timeout=None):
        if not self.consumed:
            self.owner.retained -= 1
            self.consumed = True
        return super().result(timeout)


class TrackedExecutor(Executor):
    def __init__(self):
        self.retained = self.peak = self.calls = 0

    def submit(self, fn, /, *args, **kwargs):
        self.retained += 1
        self.calls += 1
        self.peak = max(self.peak, self.retained)
        future = TrackedFuture(self)
        future.set_result(fn(*args, **kwargs))
        return future


cases = {
    "huge": ([1e200, 2e200, 3e200], [3e200, 2e200, 1e200]),
    "subnormal": ([1e-310, 2e-310, 3e-310], [3e-310, 2e-310, 1e-310]),
    "opposite_extremes": ([-1e308, 0, 1e308], [1e308, 0, -1e308]),
    "large_offset": ([1e16, 1e16+2, 1e16+4], [3, 2, 1]),
    "nonfinite": ([1, float("nan"), 3], [1, 2, 3]),
}
outcomes = {}
for name, (xs, ys) in cases.items():
    try:
        result = pearson_result(xs, ys)
        value = result.value
        if value is not None and not math.isfinite(value):
            value = repr(value)
        outcomes[name] = {"value": value, "status": result.status}
    except Exception as exc:
        outcomes[name] = {"error": type(exc).__name__}
executor = TrackedExecutor()
results = _ordered_eval_map(executor, list(range(10_000)), lambda x: x * 2)
print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "package_version": version("omega-lock"),
    "pearson_cases": outcomes,
    "executor": {"items": len(results), "calls": executor.calls,
                 "peak_unconsumed_futures": executor.peak,
                 "input_order_correct": results == [x*2 for x in range(10_000)]},
}, indent=2, allow_nan=False))
