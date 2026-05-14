"""
CSV output for DICE2023 Python results.

Produces a file with the same column structure as the GAMS put-*.gms output:
  Variable, Scenario, 2020, 2025, 2030, …, 2420
plus one scalar row per scenario for UTILITY.

The variable list mirrors put_list_module-b-4-3-10.gms.
"""

import csv
from pathlib import Path

import numpy as np

from . import params as P

# Calendar years for each period
YEARS = [P.yr0 + P.tstep * t for t in range(P.T)]

# Ordered list of time-series variables to write.
# (result_key, CSV_label)  – result_key must exist in the simulate() output dict.
TIME_SERIES_VARS = [
    ('MIU',        'MIU'),
    ('S',          'S'),
    ('K',          'K'),
    ('YGROSS',     'YGROSS'),
    ('YNET',       'YNET'),
    ('Y',          'Y'),
    ('I',          'I'),
    ('C',          'C'),
    ('CPC',        'CPC'),
    ('DAMFRAC',    'DAMFRAC'),
    ('DAMAGES',    'DAMAGES'),
    ('ABATECOST',  'ABATECOST'),
    ('abaterat',   'ABATERAT'),
    ('MCABATE',    'MCABATE'),
    ('CPRICE',     'CPRICE'),
    ('scc',        'SCC'),
    ('ECO2',       'ECO2'),
    ('EIND',       'EIND'),
    ('ECO2E',      'ECO2E'),
    ('CCATOT',     'CCATOT'),
    ('CACC',       'CACC'),
    ('MAT',        'MAT'),
    ('ppm',        'PPM'),
    ('atfrac2020', 'ATFRAC2020'),
    ('atfrac1765', 'ATFRAC1765'),
    ('RES0',       'RES0'),
    ('RES1',       'RES1'),
    ('RES2',       'RES2'),
    ('RES3',       'RES3'),
    ('alpha',      'ALPHA'),
    ('IRFt',       'IRFT'),
    ('FORC',       'FORC'),
    ('FORC_CO2',   'FORC_CO2'),
    ('F_GHGabate', 'F_GHGABATE'),
    ('TATM',       'TATM'),
    ('TBOX1',      'TBOX1'),
    ('TBOX2',      'TBOX2'),
    ('PERIODU',    'PERIODU'),
    ('TOTPERIODU', 'TOTPERIODU'),
    ('RFACTLONG',  'RFACTLONG'),
    ('RLONG',      'RLONG'),
    ('RSHORT',     'RSHORT'),
    # Exogenous arrays (same across scenarios, written once each)
    ('L',          'L'),
    ('aL',         'AL'),
]


def write_csv(
    all_results: dict,
    filename: str = 'DICE2023-Python.csv',
) -> Path:
    """Write all scenario results to a CSV file.

    Parameters
    ----------
    all_results : {scenario_name: result_dict}  from scenarios.run_all()
    filename    : output file path

    Returns
    -------
    Path to the written file.
    """
    out_path = Path(filename)
    year_strs = [str(y) for y in YEARS]

    with out_path.open('w', newline='') as fh:
        writer = csv.writer(fh)

        # Header row
        writer.writerow(['Variable', 'Scenario'] + year_strs)

        for name, result in all_results.items():
            if result is None:
                continue
            label = result.get('label', name)

            # Time-series rows
            for key, csv_label in TIME_SERIES_VARS:
                arr = result.get(key)
                if arr is None:
                    continue
                arr = np.asarray(arr, dtype=float)
                row = [csv_label, label] + [f'{v:.6g}' for v in arr]
                writer.writerow(row)

            # Scalar row: UTILITY
            writer.writerow(['UTILITY', label, f"{result['UTILITY']:.6g}"])

    print(f"Results written to: {out_path.resolve()}")
    return out_path


def print_summary(all_results: dict) -> None:
    """Print a compact summary table to stdout."""
    header = f"{'Scenario':<18} {'UTILITY':>12} {'TATM_max':>10} {'SCC_2020':>10} {'MIU_2100':>10}"
    print('\n' + header)
    print('-' * len(header))
    for name, res in all_results.items():
        if res is None:
            print(f"{name:<18} {'ERROR':>12}")
            continue
        label    = res.get('label', name)
        utility  = res['UTILITY']
        tatm_max = res['TATM'].max()
        scc_2020 = res['scc'][0]
        miu_2100 = res['MIU'][16]    # t=16 → GAMS t.val=17 → 2020+16*5=2100
        print(f"{label:<18} {utility:>12.4f} {tatm_max:>10.3f} {scc_2020:>10.2f} {miu_2100:>10.3f}")
    print()
