#!/usr/bin/env python3
"""Regression check of the SAR ADC model outputs.

Reads task1/task1_summary.csv and task2/task2_summary.csv from the given results folder and compares the
key metrics with the reference values reported for this project. Used by CI after both models run.

Usage: python tests/check_results.py [results_dir]      (default: results)
"""
import csv
import os
import sys

# (section, parameter) -> (reference value, absolute tolerance)
REFERENCE = {
    "task1": {
        ("Dynamic performance", "SNDR (SINAD)"): (59.69, 0.05),
        ("Dynamic performance", "SFDR"): (79.42, 0.05),
        ("Dynamic performance", "ENOB (normalised to full scale)"): (9.77, 0.01),
        ("Power", "TOTAL POWER"): (2.772, 0.005),
        ("Figures of merit", "Walden FoM = P/(2^ENOB*fs)"): (31.78, 0.05),
        ("Static performance (ramp code-density test)", "Missing codes"): (0, 0),
    },
    "task2": {
        ("Dynamic performance", "SNDR (SINAD)"): (59.55, 0.05),
        ("Dynamic performance", "SFDR"): (72.75, 0.05),
        ("Dynamic performance", "ENOB (normalised to full scale)"): (9.74, 0.01),
        ("Power", "TOTAL POWER"): (0.859, 0.005),
        ("Figures of merit", "Walden FoM = P/(2^ENOB*fs)"): (10.04, 0.05),
        ("Static performance (ramp code-density test)", "Missing codes"): (0, 0),
        ("CDAC switching-energy study (ideal array, all codes equally likely)",
         "Vcm-based differential equivalent (x2)"): (170.2, 0.05),
        ("CDAC switching-energy study (ideal array, all codes equally likely)",
         "Conventional differential equivalent (x2)"): (1363.3, 0.05),
        ("CDAC switching-energy study (ideal array, all codes equally likely)", "Average saving"): (87.5, 0.05),
    },
}


def read_summary(path):
    with open(path, newline="") as fh:
        return {(r["section"], r["parameter"]): r["value"] for r in csv.DictReader(fh)}


def main(argv):
    root = argv[1] if len(argv) > 1 else "results"
    failures = 0
    for task, checks in REFERENCE.items():
        summary = read_summary(os.path.join(root, task, f"{task}_summary.csv"))
        for key, (ref, tol) in checks.items():
            if key not in summary:
                print(f"FAIL {task}: '{key[1]}' missing from the summary")
                failures += 1
                continue
            value = float(summary[key])
            ok = abs(value - ref) <= tol
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'} {task}: {key[1]:<45} {value:>10.4g}  (ref {ref} +/- {tol})")
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
