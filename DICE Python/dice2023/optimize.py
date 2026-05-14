"""
DICE2023 NLP optimizer.

Translates the GAMS `solve CO2 maximizing UTILITY using nlp` into a scipy
minimization problem.  Decision variables are:

  x = [MIU[1], …, MIU[80],   S[0], …, S[36]]
       (80 values)             (37 values)   = 117 values

Fixed values:
  MIU[0] = miu1 = 0.05   (historical 2020 value, not optimised)
  S[37:] = 0.28           (applied inside forward.simulate)

For temperature-constrained scenarios (T2, T15) a quadratic penalty is added to
the objective for any period where TATM exceeds tatm_max.

GAMS runs each scenario three times for robustness; we do the same by re-running
from the best-found solution.
"""

import warnings
import numpy as np
from scipy.optimize import minimize

from . import params as P
from .forward import simulate


# ── Pack / unpack helpers ─────────────────────────────────────────────────────

def _pack(miu: np.ndarray, srate: np.ndarray) -> np.ndarray:
    """Flatten free decision variables into a 1-D optimizer vector."""
    return np.concatenate([miu[1:], srate[:37]])


def _unpack(x: np.ndarray, miuup: np.ndarray) -> tuple:
    """Reconstruct MIU and S arrays from the optimizer vector.

    Clips to bounds so the forward simulation always receives valid controls.
    """
    T      = P.T
    n_miu  = T - 1          # MIU[1..80]

    miu = np.empty(T)
    miu[0]  = P.miu1                                      # fixed historical value
    miu[1:] = np.clip(x[:n_miu], 0.0, miuup[1:])

    srate = np.empty(T)
    srate[:37] = np.clip(x[n_miu:], 0.01, 0.99)
    srate[37:] = 0.28                                     # fixed by GAMS s.fx

    return miu, srate


# ── Objective factory ─────────────────────────────────────────────────────────

def _make_objective(par, miuup, tatm_max, penalty_coeff):
    """Return a callable f(x) → scalar for scipy.optimize.minimize."""
    def objective(x):
        miu, srate = _unpack(x, miuup)
        try:
            res = simulate(miu, srate, par)
        except Exception:
            return 1e10     # penalise infeasible points

        utility = res['UTILITY']

        # Temperature penalty (hard constraint approximated as smooth penalty)
        if tatm_max is not None:
            excess = np.maximum(0.0, res['TATM'] - tatm_max)
            utility -= penalty_coeff * np.sum(excess ** 2)

        return -utility     # scipy minimises

    return objective


# ── Main solver ───────────────────────────────────────────────────────────────

def run(
    par: dict,
    tatm_max: float = None,
    miu_fixed_after: int = None,
    verbose: bool = True,
) -> tuple:
    """Solve the DICE2023 welfare-maximisation NLP.

    Parameters
    ----------
    par             : dict from precompute.build()
    tatm_max        : temperature ceiling constraint (°C); None = unconstrained
    miu_fixed_after : fix MIU = 1 for all GAMS t.val > this value (base scenario)
    verbose         : print progress

    Returns
    -------
    (miu_opt, srate_opt, result_dict)
      miu_opt, srate_opt : optimal numpy arrays
      result_dict        : output of forward.simulate() at the optimum
    """
    T      = P.T
    miuup  = par['miuup'].copy()
    optlrsav = par['optlrsav']

    # ── Build bounds ──────────────────────────────────────────────────────────
    # MIU[1..80]:  lower bound = 0, upper bound = miuup[1..]
    # For base scenario, periods where MIU is fixed at 1.0:
    miu_lb = np.zeros(T - 1)
    miu_ub = miuup[1:].copy()

    if miu_fixed_after is not None:
        for i in range(1, T):                   # i is Python 0-index
            if (i + 1) > miu_fixed_after:       # (i+1) is GAMS t.val
                miu_lb[i - 1] = 1.0
                miu_ub[i - 1] = 1.0

    # S[0..36]: realistic bounds for the savings rate
    s_lb = np.full(37, 0.01)
    s_ub = np.full(37, 0.99)

    bounds = list(zip(miu_lb, miu_ub)) + list(zip(s_lb, s_ub))

    # ── Initial guess ─────────────────────────────────────────────────────────
    miu_init = np.empty(T)
    miu_init[0] = P.miu1
    for i in range(1, T):
        miu_init[i] = np.clip(miuup[i] * 0.5, miu_lb[i - 1], miu_ub[i - 1])

    srate_init = np.full(T, optlrsav)
    x0 = _pack(miu_init, srate_init)

    # ── Penalty coefficient for temperature constraint ────────────────────────
    # Chosen large enough to enforce the constraint to < 0.01 °C tolerance
    penalty_coeff = 5e4 if tatm_max is not None else 0.0

    obj = _make_objective(par, miuup, tatm_max, penalty_coeff)

    # ── Run optimisation (3 passes, mirroring GAMS triple-solve) ─────────────
    opt_options = {'ftol': 1e-9, 'maxiter': 3000, 'disp': False}
    best_x   = x0
    best_val = obj(x0)

    for pass_num in range(3):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = minimize(
                obj,
                best_x,
                method='SLSQP',
                bounds=bounds,
                options=opt_options,
            )
        if result.fun < best_val:
            best_val = result.fun
            best_x   = result.x
        if verbose:
            status = 'OK' if result.success else result.message[:40]
            print(f"    pass {pass_num + 1}/3 : -UTILITY = {result.fun:.6f}  [{status}]")

    # ── Final evaluation ──────────────────────────────────────────────────────
    miu_opt, srate_opt = _unpack(best_x, miuup)
    res = simulate(miu_opt, srate_opt, par)

    if verbose:
        print(f"    UTILITY = {res['UTILITY']:.4f} | "
              f"TATM_max = {res['TATM'].max():.3f} C | "
              f"MIU_2100 = {miu_opt[16]:.3f}")

    return miu_opt, srate_opt, res
