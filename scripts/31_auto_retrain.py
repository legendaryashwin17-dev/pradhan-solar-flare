"""
PRADHAN Auto-Retrain: Run the full pipeline after data update.

Steps:
  1. Build balanced samples from new GOES data
  2. Extract HEL1OS features
  3. Extract SOLEXS features
  4. Run 4-expert stacking pipeline
  5. Save updated models and metrics

Usage:
  python scripts/31_auto_retrain.py
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent
SCRIPTS_DIR = WORKSPACE / "scripts"
RESULTS_DIR = WORKSPACE / "data" / "experiments" / "exp2_stacked"
RETRAIN_LOG = WORKSPACE / "data" / "retrain_log.json"


def log_retrain(status, details):
    """Append to retrain log."""
    log = []
    if RETRAIN_LOG.exists():
        with open(RETRAIN_LOG) as f:
            log = json.load(f)
    log.append({
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "details": details,
    })
    with open(RETRAIN_LOG, "w") as f:
        json.dump(log, f, indent=2)


def run_step(name, script, cwd=None):
    """Run a pipeline step and return success/failure."""
    print(f"\n{'='*70}")
    print(f"STEP: {name}")
    print(f"Script: {script}")
    print(f"{'='*70}")

    if cwd is None:
        cwd = str(WORKSPACE)

    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode == 0:
            print(f"  ✓ {name} completed successfully")
            if result.stdout:
                # Print last 20 lines of output
                lines = result.stdout.strip().split("\n")
                for line in lines[-20:]:
                    print(f"    {line}")
            return True, result.stdout
        else:
            print(f"  ✗ {name} failed (exit code {result.returncode})")
            if result.stderr:
                print(f"    stderr: {result.stderr[:500]}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        print(f"  ✗ {name} timed out (600s)")
        return False, "timeout"
    except Exception as e:
        print(f"  ✗ {name} error: {e}")
        return False, str(e)


def main():
    print("=" * 70)
    print("PRADHAN AUTO-RETRAIN")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    steps = [
        ("Build Balanced Samples", "22_build_balanced_samples.py"),
        ("Train Single Experts", "23_train_balanced.py"),
        ("Extract SOLEXS Features", "27_extract_solexs_features.py"),
        ("Run 4-Expert Stacking", "25_stacking_pipeline.py"),
    ]

    results = {}
    all_ok = True

    for name, script in steps:
        ok, output = run_step(name, script)
        results[name] = {"ok": ok, "output_tail": output[-500:] if output else ""}
        if not ok:
            all_ok = False
            print(f"\n  ⚠ Stopping pipeline at '{name}' (previous step failed)")
            break

    # Load latest results if available
    metrics_summary = {}
    results_file = RESULTS_DIR / "stacked_results_4exp.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        metrics_summary = {
            "samples": data.get("samples"),
            "tss": data.get("metrics_by_expert", {}).get("Stacked_4exp", {}).get("mean"),
            "auc": data.get("metrics_by_expert", {}).get("Stacked_4exp", {}).get("mean"),
        }

    status = "ok" if all_ok else "partial"
    log_retrain(status, {"steps": results, "metrics": metrics_summary})

    print("\n" + "=" * 70)
    print("RETRAIN SUMMARY")
    print("=" * 70)
    for name, res in results.items():
        icon = "✓" if res["ok"] else "✗"
        print(f"  {icon} {name}")
    if metrics_summary:
        print(f"\n  Latest metrics: {metrics_summary}")
    print("=" * 70)

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
