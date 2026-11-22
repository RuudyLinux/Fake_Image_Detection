"""
Quick "how far along is this" check for a running train.py / evaluate_by_category.py
background job — parses their own log output rather than guessing from wall-clock
time, since epoch/batch duration varies (phase 1 vs phase 2, GPU vs CPU, contention
from other concurrent jobs).

Caveat this script cannot fix: Python's stdout is block-buffered when redirected to
a file (not a tty), so a log can lag real progress by however much hasn't hit a
flush boundary yet — this script reports the log's most recent known position, not
a guarantee of the CURRENT instant. The checkpoint epoch field caps that lag for
training (torch.save happens immediately, unbuffered) but only updates on epochs
that improved on the previous best, not every epoch.

Usage:
    python training/progress_status.py --log checkpoints/logo_runs/train_logo_nosd14.log --kind train
    python training/progress_status.py --log checkpoints/logo_runs/eval_logo_sd14.log --kind eval
"""

import argparse
import re
import sys
from pathlib import Path

_EPOCH_RE = re.compile(r"^Epoch (\d+)/(\d+)")
_STEP_RE = re.compile(r"^\s*step (\d+)/(\d+)\s+loss=")
# download_research_data.py's streaming progress: "  scanned 4000 — real=667, ..."
_SCAN_RE = re.compile(r"^\s*scanned (\d+) [—-] (.+)$")
_DONE_RE = re.compile(r"(organized\.|Training complete|Saved ->|LOGO accuracy)")
# Two eval scripts print progress in slightly different shapes — match both
# rather than making one conform, since each line reads naturally in its own
# log: evaluate_by_category.py counts batches, evaluate_selective.py counts
# inference steps.
_EVAL_RE = re.compile(
    r"^\s*(?:eval progress|inference):?\s*(\d+)/(\d+)(?:\s+batches)?\s*\(([\d.]+)%\)"
)


def parse_train_log(text: str) -> dict:
    epoch, total_epochs, step, total_steps = None, None, None, None
    for line in text.splitlines():
        m = _EPOCH_RE.match(line)
        if m:
            epoch, total_epochs = int(m.group(1)), int(m.group(2))
            step, total_steps = None, None  # reset within-epoch progress on a new epoch header
        m = _STEP_RE.match(line)
        if m:
            step, total_steps = int(m.group(1)), int(m.group(2))

    if epoch is None:
        return {"status": "no epoch header seen yet in log (still loading data, or buffered)"}

    epoch_frac = (epoch - 1) / total_epochs
    within_epoch_frac = (step / total_steps / total_epochs) if step else 0.0
    overall_pct = 100 * (epoch_frac + within_epoch_frac)

    result = {
        "epoch": f"{epoch}/{total_epochs}",
        "overall_pct": round(overall_pct, 1),
    }
    if step:
        result["step_within_epoch"] = f"{step}/{total_steps}"
    return result


def parse_eval_log(text: str) -> dict:
    last = None
    for line in text.splitlines():
        m = _EVAL_RE.match(line)
        if m:
            last = m
    if last is None:
        return {"status": "no progress line seen yet (still loading model/data, or an older run "
                           "without the progress-print patch — see evaluate_by_category.py)"}
    return {
        "batches": f"{last.group(1)}/{last.group(2)}",
        "overall_pct": float(last.group(3)),
    }


def parse_download_log(text: str) -> dict:
    last = None
    for line in text.splitlines():
        m = _SCAN_RE.match(line)
        if m:
            last = m
    if last is None:
        return {"status": "no scan line yet (connecting / buffered)"}
    return {"scanned": int(last.group(1)), "buckets": last.group(2).strip()}


def detect_kind(text: str) -> str:
    """Guess a log's type from its content, so --scan-dir works without the
    caller knowing which script wrote which file."""
    if _EPOCH_RE.search(text) or "Epoch " in text:
        return "train"
    if _EVAL_RE.search(text) or "inference " in text:
        return "eval"
    if _SCAN_RE.search(text) or "Streaming" in text:
        return "download"
    return "unknown"


def _bar(pct: float, width: int = 24) -> str:
    filled = int(round(width * min(max(pct, 0.0), 100.0) / 100.0))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def scan_dir(dir_path: Path):
    logs = sorted(dir_path.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        print(f"No .log files in {dir_path}")
        return

    print(f"{'job':<38} {'kind':<9} progress")
    print("-" * 78)
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        kind = detect_kind(text)
        name = log.stem[:37]

        if kind == "train":
            r = parse_train_log(text)
        elif kind == "eval":
            r = parse_eval_log(text)
        elif kind == "download":
            r = parse_download_log(text)
        else:
            r = {"status": "unrecognized log format"}

        finished = bool(_DONE_RE.search(text))
        pct = r.get("overall_pct")
        if pct is not None:
            detail = f"{_bar(pct)} {pct:5.1f}%"
            if "epoch" in r:
                detail += f"  epoch {r['epoch']}"
            if "batches" in r:
                detail += f"  batch {r['batches']}"
        elif "scanned" in r:
            detail = f"scanned {r['scanned']:,}  {r['buckets'][:34]}"
        else:
            detail = r.get("status", "?")

        if finished:
            detail += "  <DONE>"
        print(f"{name:<38} {kind:<9} {detail}")

    print("-" * 78)
    print("Note: logs are block-buffered when redirected to a file, so a job can be")
    print("further along than shown. <DONE> means a completion line was found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", default=None,
                         help="Report every .log in this directory, auto-detecting each "
                              "one's type. This is what progress.bat uses.")
    parser.add_argument("--log", default=None)
    parser.add_argument("--kind", choices=["train", "eval", "download"], default=None)
    parser.add_argument("--checkpoint", default=None,
                         help="Optional: also report this checkpoint's saved epoch — unbuffered "
                              "ground truth, but only updates on epochs that beat the prior best.")
    args = parser.parse_args()

    if args.scan_dir:
        d = Path(args.scan_dir)
        if not d.is_dir():
            print(f"No such directory: {d}")
            sys.exit(1)
        scan_dir(d)
        return

    if not args.log:
        parser.error("Pass --scan-dir, or --log with --kind")

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"No such log: {log_path}")
        sys.exit(1)
    text = log_path.read_text(encoding="utf-8", errors="replace")

    kind = args.kind or detect_kind(text)
    if kind == "train":
        result = parse_train_log(text)
    elif kind == "eval":
        result = parse_eval_log(text)
    elif kind == "download":
        result = parse_download_log(text)
    else:
        result = {"status": "unrecognized log format"}
    print(f"[{log_path.name}] ({kind}) {result}")

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            import torch
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print(f"[{ckpt_path.name}] last-saved epoch={state.get('epoch')}  "
                  f"val_macro_f1={state.get('val_macro_f1')}  "
                  f"(unbuffered, but only updates on improvement — may be behind the log's "
                  f"current epoch if recent epochs haven't beaten it)")


if __name__ == "__main__":
    main()
