"""
DICE2023-beta-4-3-10 scalar parameters.
All values taken verbatim from DICE2023-b-4-3-10.gms, FAIR-beta-4-3-1.gms,
and Nonco2-b-4-3-1.gms.
"""

# ── Time horizon ──────────────────────────────────────────────────────────────
T = 81          # number of 5-year periods (2020–2425)
tstep = 5       # years per period
yr0 = 2020      # base calendar year

# ── Objective scaling ─────────────────────────────────────────────────────────
SRF    = 1_000_000       # scaling factor for discounting
scale1 = 0.00891061      # multiplicative scaling coefficient
scale2 = -6275.91        # additive scaling coefficient

# ── Population and technology ─────────────────────────────────────────────────
gama    = 0.300      # capital elasticity in Cobb-Douglas production
pop1    = 7752.9     # initial world population 2020 (millions)
popadj  = 0.145      # growth rate calibrated to 2050 UN projection
popasym = 10825.0    # asymptotic population (millions)
dk      = 0.100      # annual capital depreciation rate
q1      = 135.7      # initial gross world output 2020 (trill 2019 USD)
AL1     = 5.84       # initial total factor productivity (TFP)
gA1     = 0.066      # initial TFP growth rate per 5-year period
delA    = 0.0015     # decline rate of TFP growth per 5-year period
k0      = 295.0      # initial capital stock 2020 (trill 2019 USD)

# ── CO₂ emissions and decarbonisation ────────────────────────────────────────
gsigma1  = -0.015    # initial annual growth rate of CO2/output ratio
delgsig  = 0.96      # decline rate of gsigma per period
asymgsig = -0.005    # asymptotic gsigma
e1       = 37.56     # industrial CO2 emissions 2020 (GtCO2/year)
miu1     = 0.05      # historical emission control rate 2020
fosslim  = 6000.0    # max cumulative fossil-fuel extraction (GtC)
CumEmiss0 = 633.5    # cumulative CO2 emissions through 2020 (GtC)

# ── Climate damage function ───────────────────────────────────────────────────
a1     = 0.0         # damage intercept (linear term)
a2base = 0.003467    # damage quadratic coefficient (Nordhaus 2023 calibration)
a3     = 2.00        # damage exponent

# ── Abatement cost function ───────────────────────────────────────────────────
expcost2 = 2.6       # exponent of control cost function
pback2050 = 515.0    # backstop technology cost 2050 (2019 USD/tCO2)
gback    = -0.012    # initial annual decline rate of backstop cost
cprice1  = 6.0       # carbon price 2020 (2019 USD/tCO2)
gcprice  = 0.025     # annual growth rate of baseline carbon price

# ── Emission-control rate limits ─────────────────────────────────────────────
limmiu2070 = 1.0     # MIU upper bound from 2070 (GAMS t.val > 11)
limmiu2120 = 1.1     # MIU upper bound from 2120 (t.val > 20); >1 allows CDR
limmiu2200 = 1.05    # MIU upper bound from 2220 (t.val > 37)
limmiu2300 = 1.0     # MIU upper bound from 2300 (t.val > 57)
delmiumax  = 0.12    # maximum MIU increase per period

# ── Preferences and time discounting ─────────────────────────────────────────
betaclim = 0.5       # climate beta (risk premium loading)
elasmu   = 0.95      # elasticity of marginal utility of consumption
prstp    = 0.001     # pure rate of social time preference (per year)
pi       = 0.05      # capital risk premium
siggc1   = 0.01      # annual std dev of consumption growth (precautionary term)

# ── FAIR v1 carbon-cycle parameters ──────────────────────────────────────────
emshare0 = 0.2173    # fraction of emissions entering reservoir 0 (permanent)
emshare1 = 0.2240    # fraction entering reservoir 1
emshare2 = 0.2824    # fraction entering reservoir 2
emshare3 = 0.2763    # fraction entering reservoir 3
tau0 = 1_000_000.0   # decay time constant R0 (years) – permanent sink
tau1 = 394.4         # decay time constant R1 (years)
tau2 = 36.53         # decay time constant R2 (years)
tau3 = 4.304         # decay time constant R3 (years)
teq1 = 0.324         # thermal equilibration parameter box 1 (m²/KW)
teq2 = 0.44          # thermal equilibration parameter box 2 (m²/KW)
d1   = 236.0         # thermal response timescale deep ocean (years)
d2   = 4.07          # thermal response timescale upper ocean (years)
irf0 = 32.4          # pre-industrial IRF100 (years)
irC  = 0.019         # IRF100 increase per GtC of cumulative uptake (yr/GtC)
irT  = 4.165         # IRF100 increase per degree K of warming (yr/K)
fco22x = 3.93        # radiative forcing from CO2 doubling (W/m²)

# ── FAIR initial conditions (calibrated to 2020) ─────────────────────────────
mat0   = 886.5128014  # atmospheric carbon 2020 (GtC from 1765)
res00  = 150.093      # reservoir 0 carbon 2020 (GtC)
res10  = 102.698      # reservoir 1 carbon 2020 (GtC)
res20  = 39.534       # reservoir 2 carbon 2020 (GtC)
res30  = 6.1865       # reservoir 3 carbon 2020 (GtC)
mateq  = 588.0        # equilibrium atmospheric concentration (GtC)
tbox10 = 0.1477       # temperature box 1 in 2020 (°C above 1765)
tbox20 = 1.099454     # temperature box 2 in 2020 (°C above 1765)
tatm0  = 1.24715      # atmospheric temperature in 2020 (°C above 1765)

# ── Non-CO₂ GHG parameters ───────────────────────────────────────────────────
eland0          = 5.9       # land-use CO2 emissions 2020 (GtCO2/year)
deland          = 0.1       # annual decline rate of land emissions per period
F_Misc2020      = -0.054    # non-abatable miscellaneous forcing 2020 (W/m²)
F_Misc2100      = 0.265     # non-abatable miscellaneous forcing 2100 (W/m²)
F_GHGabate2020  = 0.518     # abatable non-CO2 GHG forcing 2020 (W/m²)
F_GHGabate2100  = 0.957     # abatable non-CO2 GHG forcing 2100 (W/m²)
ECO2eGHGB2020   = 9.96      # abatable non-CO2 GHG emissions 2020 (GtCO2e/yr)
ECO2eGHGB2100   = 15.5      # abatable non-CO2 GHG emissions 2100 (GtCO2e/yr)
emissrat2020    = 1.40      # ratio of total CO2e to industrial CO2 in 2020
emissrat2100    = 1.21      # ratio of total CO2e to industrial CO2 in 2100
Fcoef1          = 0.00955   # non-CO2 abatable forcing coefficient 1
Fcoef2          = 0.861     # non-CO2 abatable forcing coefficient 2
