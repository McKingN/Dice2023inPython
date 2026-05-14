"""
DICE2023 NLP optimizer.

Translates the GAMS `solve CO2 maximizing UTILITY using nlp` into a scipy
minimization problem.  Decision variables are:

  x = [MIU[1], ..., MIU[80],   S[0], ..., S[36]]
       (80 values)              (37 values)   = 117 values

Fixed values:
  MIU[0] = miu1 = 0.05   (historical 2020 value, not optimised)
  S[37:] = 0.28           (applied inside forward.simulate)

Temperature-constrained scenarios (T2, T15) are handled via SLSQP's native
inequality constraint interface: max(TATM) <= tatm_max.  This mirrors the
GAMS tatm.up(t) hard constraint and avoids the inaccuracy of a penalty method.

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


# ── Simulation cache (avoids double-evaluating per SLSQP iteration) ───────────

class _SimCache:
    """Small LRU-style cache keyed on x.tobytes()."""
    def __init__(self, capacity: int = 8):
        self._d = {}
        self._cap = capacity

    def get(self, x: np.ndarray, par: dict, miuup: np.ndarray):
        key = x.tobytes()
        if key not in self._d:
            if len(self._d) >= self._cap:
                self._d.pop(next(iter(self._d)))
            miu, srate = _unpack(x, miuup)
            try:
                self._d[key] = simulate(miu, srate, par)
            except Exception:
                self._d[key] = None
        return self._d[key]


# ── Objective and constraint factories ────────────────────────────────────────

def _make_funcs(par, miuup, tatm_max):
    """Return (objective_fn, constraints_list) for scipy.optimize.minimize."""
    cache = _SimCache()

    def objective(x):
        res = cache.get(x, par, miuup)
        return 1e10 if res is None else -res['UTILITY']

    constraints = []
    if tatm_max is not None:
        def temp_slack(x):
            res = cache.get(x, par, miuup)
            if res is None:
                return -1e10
            # SLSQP requires ineq constraints to be >= 0
            return tatm_max - res['TATM'].max()

        constraints = [{'type': 'ineq', 'fun': temp_slack}]

    return objective, constraints


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
    tatm_max        : temperature ceiling constraint (deg C); None = unconstrained
    miu_fixed_after : fix MIU = 1 for all GAMS t.val > this value (base scenario)
    verbose         : print progress

    Returns
    -------
    (miu_opt, srate_opt, result_dict)
      miu_opt, srate_opt : optimal numpy arrays
      result_dict        : output of forward.simulate() at the optimum
    """
    T        = P.T
    miuup    = par['miuup'].copy()
    optlrsav = par['optlrsav']

    # ── Build bounds ──────────────────────────────────────────────────────────
    # MIU[1..80]:  lower bound = 0, upper bound = miuup[1..]
    miu_lb = np.zeros(T - 1)
    miu_ub = miuup[1:].copy()

    if miu_fixed_after is not None:
        for i in range(1, T):
            if (i + 1) > miu_fixed_after:       # (i+1) is GAMS t.val
                miu_lb[i - 1] = 1.0
                miu_ub[i - 1] = 1.0

    # S[0..36]: realistic bounds for the savings rate
    s_lb = np.full(37, 0.01)
    s_ub = np.full(37, 0.99)

    bounds = list(zip(miu_lb, miu_ub)) + list(zip(s_lb, s_ub))

    # ── Initial guess ─────────────────────────────────────────────────────────
    # For temperature-constrained scenarios start at maximum abatement so the
    # first iterate is inside (or close to) the feasible region.
    miu_init = np.empty(T)
    miu_init[0] = P.miu1
    for i in range(1, T):
        if tatm_max is not None:
            miu_init[i] = miu_ub[i - 1]   # start at ceiling for constrained runs
        else:
            miu_init[i] = np.clip(miuup[i] * 0.5, miu_lb[i - 1], miu_ub[i - 1])

    srate_init = np.full(T, optlrsav)
    x0 = _pack(miu_init, srate_init)

    # ── Build objective and (optional) temperature constraint ─────────────────
    obj, constraints = _make_funcs(par, miuup, tatm_max)

    # ── Run optimisation (3 passes, mirroring GAMS triple-solve) ─────────────
    # Constrained scenarios start near the feasible point so fewer iterations
    # are needed; 400 is sufficient and avoids the very long 3000-iter runs.
    maxiter  = 400 if tatm_max is not None else 3000
    opt_options = {'ftol': 1e-9, 'maxiter': maxiter, 'disp': False}
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
                constraints=constraints,
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
