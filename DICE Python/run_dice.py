"""
DICE2023 Python – main runner.

Usage
─────
    python run_dice.py                  # run all 11 scenarios
    python run_dice.py opt T2 paris     # run specific scenarios

Output
──────
    DICE2023-Python.csv  (same directory as this script)

Dependencies
────────────
    numpy, scipy
    (install with:  pip install numpy scipy)
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows (avoids codec errors for special characters)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from dice2023 import scenarios, output


def main():
    print("=" * 60)
    print("  DICE2023 - Dynamic Integrated Climate-Economy Model")
    print("  Python / NumPy / SciPy implementation")
    print("  Translated from DICE2023-b-4-3-10 (Nordhaus, Oct 2023)")
    print("=" * 60)

    # Select scenarios from command-line args, or default to all
    requested = sys.argv[1:] if len(sys.argv) > 1 else None
    if requested:
        unknown = [s for s in requested if s not in scenarios.SCENARIOS]
        if unknown:
            print(f"\nUnknown scenario(s): {unknown}")
            print(f"Available: {list(scenarios.SCENARIOS)}")
            sys.exit(1)
        keys = requested
    else:
        keys = scenarios.RUN_ORDER

    print(f"\nScenarios to run: {keys}\n")
    t0 = time.time()

    results = scenarios.run_all(scenario_keys=keys, verbose=True)

    elapsed = time.time() - t0
    print(f"\nAll scenarios completed in {elapsed:.1f} s")

    output.print_summary(results)

    out_file = Path(__file__).parent / 'DICE2023-Python.csv'
    output.write_csv(results, filename=str(out_file))


if __name__ == '__main__':
    main()
