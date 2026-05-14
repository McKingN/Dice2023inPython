"""
DICE2023 scenario definitions and runner.

Each scenario mirrors the corresponding Include/def-*.gms file in the GAMS source.

Scenario key → Include file mapping
─────────────────────────────────────
opt     ← def-opt-b-4-3-10.gms      (welfare-maximising optimal run)
base    ← def-base-b-4-3-1.gms      (business-as-usual, carbon-price-constrained)
T2      ← def-T2-b-4-3-1.gms        (2 °C temperature ceiling)
T15     ← def-T15-b-4-3-1.gms       (1.5 °C temperature ceiling)
altdam  ← def-altdam-b-4-3-1.gms    (higher damage: a2base = 0.01)
paris   ← def-paris-b-4-3-1.gms     (Paris-consistent MIU ramp)
disc1   ← def_DISC1%-b-4-3-1.gms    (prstp = 1 %, elasmu ≈ 0, k0 = 420)
disc2   ← def_DISC2%-b-4-3-1.gms    (prstp = 2 %, k0 = 409)
disc3   ← (same pattern)             (prstp = 3 %, k0 = 370)
disc4                                (prstp = 4 %, k0 = 326)
disc5                                (prstp = 5 %, k0 = 290)
"""

from . import precompute
from .optimize import run as _opt_run


# ── Scenario catalogue ────────────────────────────────────────────────────────
# Each entry specifies:
#   label           : human-readable name written to CSV
#   build_kwargs    : keyword arguments forwarded to precompute.build()
#   tatm_max        : TATM ceiling (None = no constraint)
#   miu_fixed_after : GAMS t.val after which MIU is fixed at 1 (base scenario)

SCENARIOS = {
    'opt': {
        'label': 'Optimal',
        'build_kwargs': {},
        'tatm_max': None,
        'miu_fixed_after': None,
    },
    'base': {
        'label': 'Baseline',
        'build_kwargs': {'base_scenario': True},
        'tatm_max': 15.0,
        'miu_fixed_after': 57,     # GAMS: miu.fx(t)$(t.val > 57) = 1
    },
    'T2': {
        'label': '2 Deg C',
        'build_kwargs': {},
        'tatm_max': 2.0,
        'miu_fixed_after': None,
    },
    'T15': {
        'label': '1.5 Deg C',
        'build_kwargs': {},
        'tatm_max': 1.5,
        'miu_fixed_after': None,
    },
    'altdam': {
        'label': 'Alt Damages',
        'build_kwargs': {'a2base': 0.01},
        'tatm_max': None,
        'miu_fixed_after': None,
    },
    'paris': {
        'label': 'Paris',
        'build_kwargs': {'miuup_paris': True},
        'tatm_max': None,
        'miu_fixed_after': None,
    },
    # ── Discount-rate sensitivity scenarios ──────────────────────────────────
    # GAMS sets elasmu = 0.001 and removes the precautionary premium (RR = RR1)
    'disc1': {
        'label': '1pct Discount',
        'build_kwargs': {'prstp': 0.01, 'elasmu': 0.001, 'k0': 420,
                         'no_precaution': True},
        'tatm_max': None,
        'miu_fixed_after': None,
    },
    'disc2': {
        'label': '2pct Discount',
        'build_kwargs': {'prstp': 0.02, 'elasmu': 0.001, 'k0': 409,
                         'no_precaution': True},
        'tatm_max': None,
        'miu_fixed_after': None,
    },
    'disc3': {
        'label': '3pct Discount',
        'build_kwargs': {'prstp': 0.03, 'elasmu': 0.001, 'k0': 370,
                         'no_precaution': True},
        'tatm_max': None,
        'miu_fixed_after': None,
    },
    'disc4': {
        'label': '4pct Discount',
        'build_kwargs': {'prstp': 0.04, 'elasmu': 0.001, 'k0': 326,
                         'no_precaution': True},
        'tatm_max': None,
        'miu_fixed_after': None,
    },
    'disc5': {
        'label': '5pct Discount',
        'build_kwargs': {'prstp': 0.05, 'elasmu': 0.001, 'k0': 290,
                         'no_precaution': True},
        'tatm_max': None,
        'miu_fixed_after': None,
    },
}

# Canonical run order (same as Putlong-4-3-10.gms)
RUN_ORDER = ['opt', 'T2', 'T15', 'altdam', 'paris', 'base',
             'disc1', 'disc2', 'disc3', 'disc4', 'disc5']


# ── Single-scenario runner ────────────────────────────────────────────────────

def run_scenario(name: str, verbose: bool = True) -> dict:
    """Run a single named scenario and return a result dict.

    The result dict is the output of forward.simulate() plus extra keys:
      'label'  : scenario display name
      'name'   : scenario key
      'scc'    : social cost of carbon (≡ CPRICE at welfare optimum; see note)
      'par'    : the precompute parameter dict (for diagnostics)

    Note on SCC
    ───────────
    The GAMS model computes SCC from dual variables of the NLP solver
    (shadow price of the CO2 equation over the shadow price of consumption).
    At the welfare optimum these equal the marginal abatement cost (CPRICE).
    For non-optimal scenarios (base, T2, T15, Paris) CPRICE ≈ MAC ≠ true welfare
    SCC, but we report CPRICE as the best readily available proxy.
    """
    defn = SCENARIOS[name]
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Scenario : {defn['label']}  ({name})")
        print(f"{'='*60}")

    par = precompute.build(**defn['build_kwargs'])

    miu, srate, result = _opt_run(
        par,
        tatm_max        = defn['tatm_max'],
        miu_fixed_after = defn['miu_fixed_after'],
        verbose         = verbose,
    )

    result['scc']   = result['CPRICE'].copy()
    result['label'] = defn['label']
    result['name']  = name
    result['par']   = par

    return result


# ── Multi-scenario runner ─────────────────────────────────────────────────────

def run_all(
    scenario_keys=None,
    verbose: bool = True,
) -> dict:
    """Run multiple scenarios and return {name: result_dict}.

    Parameters
    ----------
    scenario_keys : list of scenario names to run; defaults to RUN_ORDER
    verbose       : print progress
    """
    keys = scenario_keys if scenario_keys is not None else RUN_ORDER
    results = {}
    for name in keys:
        try:
            results[name] = run_scenario(name, verbose=verbose)
        except Exception as exc:
            print(f"\n  *** ERROR in scenario '{name}': {exc}")
            results[name] = None
    return results
