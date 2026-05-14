"""
Precomputed time-series parameter arrays for DICE2023.

All formulas are taken directly from the GAMS source:
  - DICE2023-b-4-3-10.gms  (lines 88–105)
  - Nonco2-b-4-3-1.gms     (lines 28–33)

build() returns a dict of numpy arrays (shape (T,)) and scalar quantities.
Keyword arguments override the defaults for sensitivity/scenario runs.
"""

import numpy as np
from . import params as P


def build(
    prstp=None,
    k0=None,
    a2base=None,
    elasmu=None,
    no_precaution=False,   # True for DISC scenarios (RR = RR1, skip precautionary term)
    miuup_paris=False,     # True for Paris scenario miuup formula
    base_scenario=False,   # True for BASE scenario (cprice constraint on MIU)
):
    """Build all precomputed parameter arrays.

    Parameters override the defaults in params.py where supplied.
    Returns a dict consumed by forward.simulate() and optimize.run().
    """
    _prstp   = prstp   if prstp   is not None else P.prstp
    _k0      = k0      if k0      is not None else P.k0
    _a2base  = a2base  if a2base  is not None else P.a2base
    _elasmu  = elasmu  if elasmu  is not None else P.elasmu

    T  = P.T
    tv = np.arange(1, T + 1, dtype=float)   # GAMS t.val  = 1 … 81

    # ── Discount rates ────────────────────────────────────────────────────────
    # GAMS lines 88–92
    if no_precaution:
        # DISC scenarios: simple geometric discounting, no precautionary premium
        RR1      = 1.0 / (1.0 + _prstp) ** (P.tstep * (tv - 1))
        RR       = RR1.copy()
        varpcc   = np.zeros(T)
        rprecaut = np.zeros(T)
        rartp    = _prstp   # used only for optlrsav below
    else:
        rartp    = np.exp(_prstp + P.betaclim * P.pi) - 1
        varpcc   = np.minimum(P.siggc1**2 * 5 * (tv - 1), P.siggc1**2 * 5 * 47)
        rprecaut = -0.5 * varpcc * _elasmu**2
        RR1      = 1.0 / (1.0 + rartp) ** (P.tstep * (tv - 1))
        RR       = RR1 * (1.0 + rprecaut) ** (-P.tstep * (tv - 1))

    # ── Population (logistic growth, GAMS line 93) ────────────────────────────
    L = np.empty(T)
    L[0] = P.pop1
    for i in range(T - 1):
        L[i + 1] = L[i] * (P.popasym / L[i]) ** P.popadj

    # ── TFP (GAMS line 94) ────────────────────────────────────────────────────
    # gA is computed directly (not iteratively) for all t
    gA = P.gA1 * np.exp(-P.delA * 5 * (tv - 1))
    aL = np.empty(T)
    aL[0] = P.AL1
    for i in range(T - 1):
        aL[i + 1] = aL[i] / (1.0 - gA[i])

    # ── Optimal long-run savings rate (GAMS line 95) ──────────────────────────
    optlrsav = (P.dk + 0.004) / (P.dk + 0.004 * _elasmu + rartp) * P.gama

    # ── Carbon-price baseline (GAMS line 96) ──────────────────────────────────
    cpricebase = P.cprice1 * (1.0 + P.gcprice) ** (5 * (tv - 1))

    # ── Backstop price (GAMS lines 97–98) ────────────────────────────────────
    # Period 7 (GAMS t.val=7) corresponds to 2050; pback2050 is the reference.
    # For t ≤ 7: price rises going backwards at 1%/yr; for t > 7 it declines at 0.1%/yr.
    pbacktime = np.where(
        tv <= 7,
        P.pback2050 * np.exp(-5 * 0.01  * (tv - 7)),
        P.pback2050 * np.exp(-5 * 0.001 * (tv - 7)),
    )

    # ── CO₂/output ratio sigma (GAMS lines 99–101) ───────────────────────────
    sig1  = P.e1 / (P.q1 * (1.0 - P.miu1))
    # gsig is computed directly for all t (not iteratively)
    gsig  = np.minimum(P.gsigma1 * P.delgsig ** (tv - 1), P.asymgsig)
    sigma = np.empty(T)
    sigma[0] = sig1
    for i in range(T - 1):
        sigma[i + 1] = sigma[i] * np.exp(5.0 * gsig[i])

    # ── Emission-control upper bounds miuup (GAMS lines 103–105) ─────────────
    miuup = np.empty(T)
    for i in range(T):
        t_val = i + 1
        if   t_val == 1:        miuup[i] = 0.05
        elif t_val == 2:        miuup[i] = 0.10
        elif t_val <= 8:        miuup[i] = P.delmiumax * (t_val - 1)
        elif t_val <= 11:       miuup[i] = 0.85 + 0.05 * (t_val - 8)
        elif t_val <= 20:       miuup[i] = P.limmiu2070
        elif t_val <= 37:       miuup[i] = P.limmiu2120
        elif t_val <= 57:       miuup[i] = P.limmiu2200
        else:                   miuup[i] = P.limmiu2300

    # Paris scenario: override miuup with Paris-consistent ramp
    # GAMS: miuup(t) = MIN(.05 + .04*(t.val-1) - (.01*(t.val-5))$(t.val>5), 1.00)
    if miuup_paris:
        for i in range(T):
            t_val = i + 1
            base_val = 0.05 + 0.04 * (t_val - 1)
            if t_val > 5:
                base_val -= 0.01 * (t_val - 5)
            miuup[i] = min(base_val, 1.0)

    # Base scenario: miuup = 1 everywhere, then tighten by carbon-price constraint
    # GAMS: miuup(t) = 1.0;  cprice.up(t)$(t.val < 47) = cpricebase(t)
    if base_scenario:
        miuup[:] = 1.0
        # Translate cprice.up → MIU upper bound:  pback * miu^(exp-1) ≤ cprice_base
        for i in range(min(46, T)):   # t.val 1..46
            if pbacktime[i] > 0:
                miu_max_cprice = (cpricebase[i] / pbacktime[i]) ** (1.0 / (P.expcost2 - 1))
                miuup[i] = min(miuup[i], miu_max_cprice)

    # ── Non-CO₂ GHG arrays (Nonco2-b-4-3-1.gms lines 28–33) ─────────────────
    # Linear interpolation 2020→2100, flat thereafter
    CO2E_GHGabateB = np.where(
        tv <= 16,
        P.ECO2eGHGB2020 + (P.ECO2eGHGB2100 - P.ECO2eGHGB2020) / 16 * (tv - 1),
        P.ECO2eGHGB2020 + (P.ECO2eGHGB2100 - P.ECO2eGHGB2020),
    )
    F_Misc = np.where(
        tv <= 16,
        P.F_Misc2020 + (P.F_Misc2100 - P.F_Misc2020) / 16 * (tv - 1),
        P.F_Misc2020 + (P.F_Misc2100 - P.F_Misc2020),
    )
    emissrat = np.where(
        tv <= 16,
        P.emissrat2020 + (P.emissrat2100 - P.emissrat2020) / 16 * (tv - 1),
        P.emissrat2020 + (P.emissrat2100 - P.emissrat2020),
    )
    sigmatot = sigma * emissrat

    # Land-use CO₂ emissions (exponential decay)
    eland = P.eland0 * (1.0 - P.deland) ** (tv - 1)

    # ── Abatement cost factor (Nonco2, line 33) ───────────────────────────────
    # cost1tot(t) = pbacktime(t) * sigmatot(t) / expcost2 / 1000
    cost1tot = pbacktime * sigmatot / P.expcost2 / 1000.0

    return {
        # Scalars
        'T':        T,
        'elasmu':   _elasmu,
        'prstp':    _prstp,
        'k0':       _k0,
        'a2base':   _a2base,
        'optlrsav': optlrsav,
        'rartp':    rartp,
        # Time-series arrays (all shape (T,))
        'L':               L,
        'aL':              aL,
        'gA':              gA,
        'sigma':           sigma,
        'sigmatot':        sigmatot,
        'gsig':            gsig,
        'eland':           eland,
        'pbacktime':       pbacktime,
        'cpricebase':      cpricebase,
        'miuup':           miuup,
        'RR':              RR,
        'RR1':             RR1,
        'varpcc':          varpcc,
        'rprecaut':        rprecaut,
        'cost1tot':        cost1tot,
        'CO2E_GHGabateB':  CO2E_GHGabateB,
        'F_Misc':          F_Misc,
        'emissrat':        emissrat,
        # Flags
        'base_scenario':   base_scenario,
    }
