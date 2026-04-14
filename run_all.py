"""
run_all.py

Master script that runs all analysis scripts in order and logs console output.

Usage:
    python run_all.py

Output:
    output_logs/run_all_<timestamp>.log    (full console output)

Scripts 3, 6, and 7 require vectorbtpro. If it is not installed,
those scripts will print a skip message and continue.
"""

import os
import sys
import subprocess
import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE, "output_logs")
os.makedirs(LOG_DIR, exist_ok=True)

SCRIPTS = [
    ("1_data-preprocessing.py", "Data Preprocessing"),
    ("2_cluster_analysis.py",   "Cluster Analysis"),
    ("3_methodology.py",        "Methodology Visuals"),
    ("4_results.py",            "Results and Performance"),
    ("5_sensitivity.py",        "Sensitivity Analysis"),
    ("6_cpcv.py",               "CPCV Robustness"),
    ("7_nco.py",                "NCO Comparison"),
]

DIVIDER = "=" * 70


def run_script(script_name, label, log_file):
    """Run a single script, stream output to console and log file."""
    header = f"\n{DIVIDER}\n  [{label}] Running {script_name}\n{DIVIDER}\n"
    print(header, end="")
    log_file.write(header)

    script_path = os.path.join(BASE, script_name)
    if not os.path.exists(script_path):
        msg = f"  [ERROR] {script_name} not found. Skipping.\n"
        print(msg, end="")
        log_file.write(msg)
        return False

    proc = subprocess.Popen(
        [sys.executable, script_path],
        cwd=BASE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for line in proc.stdout:
        print(line, end="")
        log_file.write(line)

    proc.wait()
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    footer = f"\n  [{label}] {status}\n"
    print(footer, end="")
    log_file.write(footer)
    return proc.returncode == 0


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"run_all_{ts}.log")

    print(DIVIDER)
    print("  SCI999 — Full Pipeline Run")
    print(f"  Log: {log_path}")
    print(DIVIDER)

    with open(log_path, "w") as log_file:
        log_file.write(f"SCI999 Full Pipeline Run — {ts}\n")
        log_file.write(f"Python: {sys.executable}\n\n")

        results = []
        for script_name, label in SCRIPTS:
            ok = run_script(script_name, label, log_file)
            results.append((label, script_name, ok))

        # Summary
        summary = f"\n{DIVIDER}\n  SUMMARY\n{DIVIDER}\n"
        for label, script_name, ok in results:
            status = "OK" if ok else "FAILED/SKIPPED"
            summary += f"  {label:<30} {script_name:<30} {status}\n"
        summary += DIVIDER + "\n"

        print(summary)
        log_file.write(summary)

    print(f"\n  Full log saved to: {log_path}\n")


if __name__ == "__main__":
    main()
