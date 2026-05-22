"""Paced replay of phantom_demo output for screencast recording.

Runs phantom_demo's REAL output (captured in `_demo_output.txt`) at a
deliberate tempo so a viewer can read each phase. No fabricated numbers —
the file is a verbatim capture of the actual run.

Total wall time tuned for ~55 seconds. Adjust SECTION_PAUSES to taste.

Usage::

    python examples/demo_replay.py

For higher-quality recording (no scrollback flicker), open a fresh terminal,
size it to 100x40 minimum, set font size 16-18pt, then run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Pacing rules — line content -> pause AFTER printing that line (seconds).
# First match wins. Order matters.
SECTION_PAUSES: list[tuple[str, float]] = [
    # Stress probe lines (12 axes) — 1.0s each = 12s total
    ("] alpha", 1.0),
    ("] window", 1.0),
    ("] long_mode", 1.5),  # extra pause before decoys start
    ("] decoy_scale", 0.6),
    ("] decoy_offset", 0.6),
    ("] decoy_bias", 0.6),
    ("] decoy_mag", 0.6),
    ("] decoy_ofi", 0.6),
    ("] decoy_mult", 0.6),
    ("] decoy_exp", 0.6),
    ("] decoy_mode", 0.6),
    ("] decoy_flag", 1.5),  # end of stress phase
    # Grid search lines
    ("grid: 50 combos", 1.0),
    ("[  25/50]", 1.5),
    ("[  50/50]", 2.0),
    # Summary header
    ("-- PhantomKeyhole P1 summary --", 1.5),
    ("status:", 1.0),
    ("baseline:", 0.8),
    ("top_k:", 1.5),
    ("grid_best:", 1.5),
    ("walk_forward:", 2.0),
    ("hybrid_top[0]:", 1.0),
    # KC reports
    ("KC reports:", 1.5),
    ("[PASS] KC-2:", 1.2),
    ("[PASS] KC-4:", 2.0),  # KC-4 is the headline
    ("[PASS] KC-1:", 1.0),
    ("[PASS] KC-3:", 1.5),
    # Stress ranking
    ("stress top-3 vs bottom-3:", 1.0),
    ("alpha           raw=", 0.6),
    ("long_mode       raw=", 0.6),
    ("window          raw=", 1.5),
    ("...", 0.8),
    ("decoy_exp", 0.4),
    ("decoy_mult", 0.4),
    ("decoy_mag", 1.5),
    # Output path
    ("output:", 1.5),
    # Fractal-vise section header — slower, this is the "iteration shines" beat
    ("PhantomKeyhole fractal-vise", 1.5),
    ("final_status:", 1.0),
    ("stop_reason:", 0.8),
    ("rounds run:", 0.8),
    ("locked order:", 1.5),
    ("round_best:", 1.5),
    ("refined effective:", 2.0),
    ("(plain coarse grid", 1.5),
    # Final
    ("PhantomKeyhole demo PASSED.", 2.0),
]

# Default tempo for any line not matching a rule above
DEFAULT_PAUSE = 0.05


def _capture_path() -> Path:
    return Path(__file__).parent / "_demo_output.txt"


def _pause_for(line: str) -> float:
    for pattern, pause in SECTION_PAUSES:
        if pattern in line:
            return pause
    return DEFAULT_PAUSE


def build_check_summary(text: str) -> dict[str, object]:
    lines = text.splitlines()
    markers = {
        "status_pass": "  status:   PASS" in text,
        "top_k_effective": "top_k:    ['alpha', 'long_mode', 'window']" in text,
        "grid_best_effective": (
            "grid_best: {'alpha': 0.5, 'long_mode': True, 'window': 3}" in text
        ),
        "walk_forward_gate_pass": "[PASS] KC-4: PASS:" in text,
        "kc3_pass": "[PASS] KC-3: PASS:" in text,
        "fractal_final_pass": "final_status:  PASS" in text,
        "demo_completed": "PhantomKeyhole demo PASSED." in text,
    }
    return {
        "demo": "demo_replay",
        "source": "examples/_demo_output.txt",
        "line_count": len(lines),
        "capture_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "markers": markers,
        "status": "PASS" if all(markers.values()) else "FAIL",
    }


def check_capture(capture: Path | None = None) -> dict[str, object]:
    capture = capture or _capture_path()
    if not capture.exists():
        return {
            "demo": "demo_replay",
            "source": capture.as_posix(),
            "status": "FAIL",
            "error": "capture missing",
        }
    text = capture.read_text(encoding="utf-8", errors="replace")
    return build_check_summary(text)


def _run_check() -> int:
    summary = check_capture()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


def _run_replay() -> int:
    capture = _capture_path()
    if not capture.exists():
        print(f"ERROR: capture missing at {capture}", file=sys.stderr)
        print(
            "Regenerate with: python examples/phantom_demo.py > "
            "examples/_demo_output.txt 2>&1",
            file=sys.stderr,
        )
        return 1

    text = capture.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    started = time.perf_counter()
    for line in lines:
        print(line, flush=True)
        time.sleep(_pause_for(line))

    elapsed = time.perf_counter() - started
    # Tiny tail pause so the recorder can stop cleanly
    time.sleep(0.5)
    print(f"\n[demo_replay] elapsed: {elapsed:.1f}s", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run deterministic self-check")
    args = parser.parse_args(argv)
    return _run_check() if args.check else _run_replay()


if __name__ == "__main__":
    raise SystemExit(main())
