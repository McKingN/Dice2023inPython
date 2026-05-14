"""
DICE2023 forward simulation.

Given emission-control rates MIU(t) and savings rates S(t) as numpy arrays,
propagates all state variables one period at a time and returns the full
trajectory.  Implements the same equations as the GAMS NLP but in reduced form
(only MIU and S are free; everything else is derived).

Key design choices
──────────────────
* The FAIR model involves an implicit equation for the carbon-decay scaling
  factor α(t): LHS(α) = irf0 + irC·CACC(t) + irT·TATM(t).  Because CACC and
  TATM both depend on α (through the reservoir and temperature equations), this
  is a coupled scalar root-finding problem solved with scipy.brentq at each step.

* ECO2(t+1) can be precomputed before solving the climate state at t+1 because
  YGROSS(t+1) depends only on K(t+1) and exogenous TFP/population – not on
  temperature.  This makes the per-step system a pure function of α.

* Capital accumulation uses equality (K is always at its lower-bound constraint
  at the optimum in a Ramsey model).

Sources: DICE2023-b-4-3-10.gms (lines 162–188), FAIR-beta-4-3-1.gms (lines 81–93),
         Nonco2-b-4-3-1.gms (lines 28–45).
"""

import numpy as np
from scipy.optimize import brentq

from . import params as P


# ── FAIR helper functions ─────────────────────────────────────────────────────

def _irflhs(alpha: float) -> float:
    """Impulse-response function at 100 years as a function of α (LHS of IRF eq).

    GAMS irfeqlhs(t): IRFt = Σ_i α·emshare_i·τ_i·(1 − exp(−100/(α·τ_i)))
    """
    total = 0.0
    for es, tau in (
        (P.emshare0, P.tau0),
        (P.emshare1, P.tau1),
        (P.emshare2, P.tau2),
        (P.emshare3, P.tau3),
    ):
        total += alpha * es * tau * (1.0 - np.exp(-100.0 / (alpha * tau)))
    return total


def _climate_from_alpha(
    alpha: float,
    eco2_gtco2yr: float,
    res_prev: tuple,
    tbox1_prev: float,
    tbox2_prev: float,
    ccatot_next: float,
    f_ghg_next: float,
    f_misc_next: float,
) -> tuple:
    """Compute all climate-state variables at t+1 for a given α value.

    Parameters
    ----------
    alpha         : carbon-decay scaling factor at t+1
    eco2_gtco2yr  : ECO2(t+1) in GtCO2/year
    res_prev      : (RES0, RES1, RES2, RES3) at time t  (GtC)
    tbox1_prev    : TBOX1 at time t  (°C)
    tbox2_prev    : TBOX2 at time t  (°C)
    ccatot_next   : CCATOT(t+1) already propagated  (GtC)
    f_ghg_next    : F_GHGabate(t+1) already propagated  (W/m²)
    f_misc_next   : F_Misc(t+1) from precomputed array  (W/m²)

    Returns
    -------
    (mat, cacc, forc, tatm, tbox1, tbox2, r0, r1, r2, r3)
    """
    eco2_gtc = eco2_gtco2yr / 3.667   # GtCO2 → GtC  (GAMS uses 3.667)

    def _res(es, tau, r_prev):
        at = alpha * tau
        return es * at * eco2_gtc * (1.0 - np.exp(-P.tstep / at)) + r_prev * np.exp(-P.tstep / at)

    r0 = _res(P.emshare0, P.tau0, res_prev[0])
    r1 = _res(P.emshare1, P.tau1, res_prev[1])
    r2 = _res(P.emshare2, P.tau2, res_prev[2])
    r3 = _res(P.emshare3, P.tau3, res_prev[3])

    mat   = max(P.mateq + r0 + r1 + r2 + r3, 10.0)   # MAT.LO = 10
    cacc  = ccatot_next - (mat - P.mateq)              # CACC = CCATOT − (MAT − mateq)
    forc  = P.fco22x * np.log(mat / P.mateq) / np.log(2.0) + f_misc_next + f_ghg_next
    tbox1 = (tbox1_prev * np.exp(-P.tstep / P.d1)
             + P.teq1 * forc * (1.0 - np.exp(-P.tstep / P.d1)))
    tbox2 = (tbox2_prev * np.exp(-P.tstep / P.d2)
             + P.teq2 * forc * (1.0 - np.exp(-P.tstep / P.d2)))
    tatm  = max(min(tbox1 + tbox2, 20.0), 0.5)   # TATM bounds from FAIR file

    return mat, cacc, forc, tatm, tbox1, tbox2, r0, r1, r2, r3


def _solve_alpha(
    eco2_gtco2yr: float,
    res_prev: tuple,
    tbox1_prev: float,
    tbox2_prev: float,
    ccatot_next: float,
    f_ghg_next: float,
    f_misc_next: float,
) -> float:
    """Find α by solving LHS(α) = RHS(α) via Brent's method.

    Both sides depend on α because CACC and TATM (entering RHS) are themselves
    functions of the reservoir/temperature state which depends on α.
    """
    def residual(alpha):
        _, cacc, _, tatm, *_ = _climate_from_alpha(
            alpha, eco2_gtco2yr, res_prev,
            tbox1_prev, tbox2_prev,
            ccatot_next, f_ghg_next, f_misc_next,
        )
        return _irflhs(alpha) - (P.irf0 + P.irC * cacc + P.irT * tatm)

    r_lo = residual(0.1)
    r_hi = residual(100.0)
    if r_lo >= 0.0:
        return 0.1
    if r_hi <= 0.0:
        return 100.0
    return brentq(residual, 0.1, 100.0, xtol=1e-8, rtol=1e-8)


# ── Forward simulation ────────────────────────────────────────────────────────

def simulate(miu: np.ndarray, srate: np.ndarray, par: dict) -> dict:
    """Forward-simulate DICE2023 given control arrays.

    Parameters
    ----------
    miu   : shape (T,) – emission-control rate MIU(t)
    srate : shape (T,) – savings rate S(t); periods > 37 are overridden to 0.28
    par   : parameter dict produced by precompute.build()

    Returns
    -------
    dict of all model variables as numpy arrays of shape (T,), plus UTILITY scalar.

    Notes
    -----
    For t < T−1 the function:
      1. Propagates capital, cumulative emissions, and non-CO2 forcing.
      2. Previews YGROSS(t+1) from the already-known K(t+1) to obtain ECO2(t+1).
      3. Solves the coupled (α, climate) system at t+1 via scalar root-finding.
    This exactly mirrors the GAMS NLP equilibrium conditions.
    """
    T    = P.T
    tstep = P.tstep

    # ── Apply fixed savings rates (GAMS: s.fx(t)$(t.val > 37) = .28) ─────────
    s = srate.copy()
    s[37:] = 0.28

    # ── Unpack precomputed arrays ─────────────────────────────────────────────
    L              = par['L']
    aL             = par['aL']
    sigma          = par['sigma']
    cost1tot       = par['cost1tot']
    eland          = par['eland']
    CO2E_GHGabateB = par['CO2E_GHGabateB']
    F_Misc         = par['F_Misc']
    RR             = par['RR']
    pbacktime      = par['pbacktime']
    _elasmu        = par['elasmu']
    _a2base        = par['a2base']

    # ── Allocate output arrays ────────────────────────────────────────────────
    K         = np.zeros(T)
    MAT       = np.zeros(T)
    RES0      = np.zeros(T)
    RES1      = np.zeros(T)
    RES2      = np.zeros(T)
    RES3      = np.zeros(T)
    TATM      = np.zeros(T)
    TBOX1     = np.zeros(T)
    TBOX2     = np.zeros(T)
    CCATOT    = np.zeros(T)
    F_GHGabate = np.zeros(T)
    alpha     = np.zeros(T)
    CACC      = np.zeros(T)
    IRFt      = np.zeros(T)
    FORC      = np.zeros(T)

    YGROSS    = np.zeros(T)
    YNET      = np.zeros(T)
    Y         = np.zeros(T)
    C         = np.zeros(T)
    CPC       = np.zeros(T)
    I_inv     = np.zeros(T)
    DAMFRAC   = np.zeros(T)
    DAMAGES   = np.zeros(T)
    ABATECOST = np.zeros(T)
    MCABATE   = np.zeros(T)
    CPRICE    = np.zeros(T)
    ECO2      = np.zeros(T)
    EIND      = np.zeros(T)
    ECO2E     = np.zeros(T)
    PERIODU   = np.zeros(T)
    TOTPERIODU = np.zeros(T)
    RFACTLONG = np.zeros(T)
    RLONG     = np.zeros(T)
    RSHORT    = np.zeros(T)

    # ── Initial conditions ────────────────────────────────────────────────────
    K[0]          = par['k0']
    MAT[0]        = P.mat0
    RES0[0]       = P.res00
    RES1[0]       = P.res10
    RES2[0]       = P.res20
    RES3[0]       = P.res30
    TATM[0]       = P.tatm0
    TBOX1[0]      = P.tbox10
    TBOX2[0]      = P.tbox20
    CCATOT[0]     = P.CumEmiss0
    F_GHGabate[0] = P.F_GHGabate2020
    RFACTLONG[0]  = P.SRF

    # Solve α(0) from initial conditions – no coupling since CACC/TATM are given
    CACC[0] = P.CumEmiss0 - (P.mat0 - P.mateq)

    def _res0(a):
        return _irflhs(a) - (P.irf0 + P.irC * CACC[0] + P.irT * P.tatm0)

    r0_lo, r0_hi = _res0(0.1), _res0(100.0)
    if r0_lo >= 0.0:
        alpha[0] = 0.1
    elif r0_hi <= 0.0:
        alpha[0] = 100.0
    else:
        alpha[0] = brentq(_res0, 0.1, 100.0, xtol=1e-8)
    IRFt[0] = _irflhs(alpha[0])

    # ── Main simulation loop ──────────────────────────────────────────────────
    for t in range(T):

        # ── Production and economic accounting ───────────────────────────────
        YGROSS[t]    = aL[t] * (L[t] / 1000.0) ** (1.0 - P.gama) * K[t] ** P.gama
        DAMFRAC[t]   = P.a1 * TATM[t] + _a2base * TATM[t] ** P.a3
        DAMAGES[t]   = YGROSS[t] * DAMFRAC[t]
        ABATECOST[t] = YGROSS[t] * cost1tot[t] * miu[t] ** P.expcost2
        MCABATE[t]   = pbacktime[t] * miu[t] ** (P.expcost2 - 1.0)
        CPRICE[t]    = MCABATE[t]
        YNET[t]      = YGROSS[t] * (1.0 - DAMFRAC[t])
        Y[t]         = YNET[t] - ABATECOST[t]
        I_inv[t]     = s[t] * Y[t]
        C[t]         = max(Y[t] - I_inv[t], 2.0)      # C.LO = 2
        CPC[t]       = max(1000.0 * C[t] / L[t], 0.01)  # CPC.LO = 0.01

        # ── Emissions ─────────────────────────────────────────────────────────
        ECO2[t]  = (sigma[t] * YGROSS[t] + eland[t]) * (1.0 - miu[t])
        EIND[t]  = sigma[t] * YGROSS[t] * (1.0 - miu[t])
        ECO2E[t] = (sigma[t] * YGROSS[t] + eland[t] + CO2E_GHGabateB[t]) * (1.0 - miu[t])

        # ── Radiative forcing (MAT[t] and F_GHGabate[t] are already set) ─────
        FORC[t] = (P.fco22x * np.log(MAT[t] / P.mateq) / np.log(2.0)
                   + F_Misc[t] + F_GHGabate[t])

        # ── Period welfare ────────────────────────────────────────────────────
        PERIODU[t]    = (CPC[t] ** (1.0 - _elasmu) - 1.0) / (1.0 - _elasmu) - 1.0
        TOTPERIODU[t] = PERIODU[t] * L[t] * RR[t]

        # ── Propagate state to t+1 ────────────────────────────────────────────
        if t < T - 1:
            # Capital: KK(t+1) =L= (1−dk)^tstep·K(t) + tstep·I(t)  [binding at optimum]
            K[t + 1] = max((1.0 - P.dk) ** tstep * K[t] + tstep * I_inv[t], 1.0)

            # Cumulative CO2 emissions (GtCO2 → GtC via /3.666)
            CCATOT[t + 1] = CCATOT[t] + ECO2[t] * (tstep / 3.666)

            # Non-CO2 abatable GHG forcing  (GAMS F_GHGabateEQ)
            F_GHGabate[t + 1] = (P.Fcoef2 * F_GHGabate[t]
                                  + P.Fcoef1 * CO2E_GHGabateB[t] * (1.0 - miu[t]))

            # Preview ECO2(t+1): YGROSS(t+1) depends only on K(t+1) and exogenous params
            YGROSS_next = aL[t + 1] * (L[t + 1] / 1000.0) ** (1.0 - P.gama) * K[t + 1] ** P.gama
            ECO2_next   = (sigma[t + 1] * YGROSS_next + eland[t + 1]) * (1.0 - miu[t + 1])

            # Solve coupled (α, climate) system at t+1
            a_next = _solve_alpha(
                ECO2_next,
                (RES0[t], RES1[t], RES2[t], RES3[t]),
                TBOX1[t], TBOX2[t],
                CCATOT[t + 1], F_GHGabate[t + 1], F_Misc[t + 1],
            )
            alpha[t + 1] = a_next
            IRFt[t + 1]  = _irflhs(a_next)

            (MAT[t + 1], CACC[t + 1], _forc_next, TATM[t + 1],
             TBOX1[t + 1], TBOX2[t + 1],
             RES0[t + 1], RES1[t + 1], RES2[t + 1], RES3[t + 1]) = _climate_from_alpha(
                a_next, ECO2_next,
                (RES0[t], RES1[t], RES2[t], RES3[t]),
                TBOX1[t], TBOX2[t],
                CCATOT[t + 1], F_GHGabate[t + 1], F_Misc[t + 1],
            )

    # ── Discount factors and interest rates (post-loop) ───────────────────────
    # GAMS RFACTLONGeq(t+1), RLONGeq(t+1), RSHORTeq(t+1)
    for t in range(1, T):
        RFACTLONG[t] = max(
            P.SRF * (CPC[t] / CPC[0]) ** (-_elasmu) * RR[t], 0.0001
        )
        RLONG[t]  = -np.log(RFACTLONG[t] / P.SRF) / (5.0 * t)   # t is 0-indexed here
        RSHORT[t] = -np.log(RFACTLONG[t] / RFACTLONG[t - 1]) / 5.0

    # ── Welfare objective ─────────────────────────────────────────────────────
    UTILITY = P.tstep * P.scale1 * np.sum(TOTPERIODU) + P.scale2

    # ── Post-solution derived quantities (mirrors def-opt-b-4-3-10.gms) ──────
    ppm        = MAT / 2.13
    # Atmospheric fraction since 2020
    denom_2020 = CCATOT - P.CumEmiss0 + 1e-5
    atfrac2020 = np.where(np.abs(denom_2020) > 1e-6,
                          (MAT - P.mat0) / denom_2020, 0.0)
    # Atmospheric fraction since 1765
    atfrac1765 = (MAT - P.mateq) / (CCATOT + 1e-5)
    abaterat   = np.where(Y > 1e-6, ABATECOST / Y, 0.0)
    FORC_CO2   = P.fco22x * np.log(MAT / P.mateq) / np.log(2.0)

    return {
        # Climate / carbon cycle
        'MAT':        MAT,
        'RES0':       RES0,
        'RES1':       RES1,
        'RES2':       RES2,
        'RES3':       RES3,
        'TATM':       TATM,
        'TBOX1':      TBOX1,
        'TBOX2':      TBOX2,
        'CCATOT':     CCATOT,
        'CACC':       CACC,
        'alpha':      alpha,
        'IRFt':       IRFt,
        'FORC':       FORC,
        'FORC_CO2':   FORC_CO2,
        'F_GHGabate': F_GHGabate,
        # Economic
        'K':          K,
        'YGROSS':     YGROSS,
        'YNET':       YNET,
        'Y':          Y,
        'C':          C,
        'CPC':        CPC,
        'I':          I_inv,
        'DAMFRAC':    DAMFRAC,
        'DAMAGES':    DAMAGES,
        'ABATECOST':  ABATECOST,
        'MCABATE':    MCABATE,
        'CPRICE':     CPRICE,
        # Emissions
        'ECO2':       ECO2,
        'EIND':       EIND,
        'ECO2E':      ECO2E,
        # Controls
        'MIU':        miu,
        'S':          s,
        # Welfare and rates
        'PERIODU':    PERIODU,
        'TOTPERIODU': TOTPERIODU,
        'UTILITY':    UTILITY,
        'RFACTLONG':  RFACTLONG,
        'RLONG':      RLONG,
        'RSHORT':     RSHORT,
        # Derived output
        'ppm':        ppm,
        'atfrac2020': atfrac2020,
        'atfrac1765': atfrac1765,
        'abaterat':   abaterat,
        # Exogenous (pass-through for CSV output)
        'L':          L,
        'aL':         aL,
    }
