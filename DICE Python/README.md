# DICE2023 — Python Implementation

A Python / NumPy / SciPy translation of the **DICE2023** model  
(Dynamic Integrated Climate-Economy, version beta-4-3-10)  
originally written by William D. Nordhaus in GAMS.

This README is written for readers who have no prior knowledge of DICE or its
original GAMS code. It explains the model theory in plain language, shows the
equations, and points to the exact Python file and function that implements each
equation.

---

## Table of Contents

1. [What is DICE?](#1-what-is-dice)
2. [Model overview — the big picture](#2-model-overview--the-big-picture)
3. [The equations, explained](#3-the-equations-explained)
   - 3.1 The economy
   - 3.2 CO₂ emissions
   - 3.3 The carbon cycle (FAIR)
   - 3.4 Radiative forcing and temperature
   - 3.5 Damages
   - 3.6 Abatement costs
   - 3.7 The welfare function
   - 3.8 Discount rates
4. [The 11 scenarios](#4-the-11-scenarios)
5. [Installation](#5-installation)
6. [How to run](#6-how-to-run)
7. [Expected output](#7-expected-output)
8. [How to visualise the results](#8-how-to-visualise-the-results)
9. [Code structure — file by file](#9-code-structure--file-by-file)
10. [How to add a new scenario](#10-how-to-add-a-new-scenario)
11. [Limitations](#11-limitations)
12. [References](#12-references)

---

## 1. What is DICE?

Climate change is an **economic problem** as much as a scientific one.
Burning fossil fuels raises global temperatures, which causes economic damages
(lower agricultural yields, coastal flooding, health costs).
Reducing emissions costs money too — new clean technology, retired factories.
Society must decide how aggressively to cut emissions now versus accepting
future damages.

**DICE** (Dynamic Integrated Climate-Economy) is the most widely used model for
this trade-off. It was built by William Nordhaus at Yale University, work for
which he received the **2018 Nobel Prize in Economics**.

DICE connects a macroeconomic model of the world economy to a simplified
climate model in a single optimisation framework. It finds the path of
emission cuts and investment that maximises global human welfare over a
400-year horizon.

> **The model's most cited output: the Social Cost of Carbon (SCC)** — the
> dollar value of the economic harm caused by emitting one additional tonne of
> CO₂ today. It is the key number used to set carbon taxes and compare the costs
> and benefits of climate policies.

This repository translates the original GAMS code (a commercial mathematical
programming language common in economics) into Python so that it can be used,
modified, and taught without a GAMS licence.

---

## 2. Model overview — the big picture

The model covers the **entire world as one unit** over **81 periods of 5 years**
(2020 to 2425). At each period, it asks: given the current state of the economy
and climate, what is the best level of emission control and savings?

The causal chain runs like this:

```
Savings rate S(t)  ──►  Investment  ──►  Capital K(t+1)
                                              │
                                              ▼
Emission control        ──►  CO₂ emissions ECO₂(t)  ──►  Atmospheric CO₂ MAT(t)
rate MIU(t)                                                        │
    │                                                              ▼
    └──►  Abatement cost                              Radiative forcing FORC(t)
                                                               │
                                                               ▼
                                                     Temperature TATM(t)
                                                               │
                                                               ▼
                                  GDP gross YGROSS(t)  ──►  Damages DAMAGES(t)
                                              │
                                              ▼
                                  GDP net Y(t)  ──►  Consumption C(t)  ──►  Welfare
```

The two **free choice variables** are MIU(t) and S(t). Everything else follows
deterministically from these choices and the model equations.

---

## 3. The equations, explained

For each equation the README gives:
- **Plain language**: what it means intuitively
- **The equation**: formal mathematical statement
- **Code**: which file and function implements it

---

### 3.1 The economy

#### Production function

**Plain language:** World GDP is produced by combining capital (machines,
buildings), labour (population), and technology. The model uses a standard
Cobb-Douglas production function. Technology and population grow exogenously
over time — the model cannot choose them.

**Equation:**

$$Y_{\text{gross}}(t) = A(t) \cdot \left(\frac{L(t)}{1000}\right)^{1-\gamma} \cdot K(t)^{\gamma}$$

| Symbol | Meaning | Value / Source |
|--------|---------|----------------|
| $A(t)$ | Total factor productivity (TFP) | Grows at declining rate; computed in `precompute.py` |
| $L(t)$ | World population (millions) | Logistic growth to 10 825 million; in `precompute.py` |
| $\gamma$ | Capital share of output | 0.300 (`params.py: gama`) |
| $K(t)$ | Capital stock (trill 2019 USD) | State variable, starts at 295 |

**Code:** `forward.py` → `simulate()`, line `YGROSS[t] = aL[t] * (L[t]/1000)**(1-P.gama) * K[t]**P.gama`

---

#### Capital accumulation

**Plain language:** Capital depreciates each year (buildings wear out, machines
break) and is replenished by investment. Investment equals the savings rate
times total output.

**Equations:**

$$I(t) = S(t) \cdot Y(t)$$

$$K(t+1) = (1 - \delta_K)^{\Delta t} \cdot K(t) + \Delta t \cdot I(t)$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| $S(t)$ | Savings rate (fraction of output invested) | Choice variable; fixed at 0.28 after 2205 |
| $\delta_K$ | Annual capital depreciation rate | 0.100 (`params.py: dk`) |
| $\Delta t$ | Years per period | 5 |

**Code:** `forward.py` → `simulate()`:
```python
I_inv[t] = s[t] * Y[t]
K[t+1]   = max((1 - P.dk)**tstep * K[t] + tstep * I_inv[t], 1.0)
```

---

#### Output accounting

**Plain language:** Gross output is reduced by climate damages and by the cost
of cutting emissions. What remains is split between investment and consumption.

**Equations:**

$$Y_{\text{net}}(t) = Y_{\text{gross}}(t) \cdot \bigl(1 - \text{DAMFRAC}(t)\bigr)$$

$$Y(t) = Y_{\text{net}}(t) - \text{ABATECOST}(t)$$

$$C(t) = Y(t) - I(t) \qquad \text{(consumption)}$$

$$\text{CPC}(t) = \frac{1000 \cdot C(t)}{L(t)} \qquad \text{(per-capita consumption, k\$/yr)}$$

**Code:** `forward.py` → `simulate()`:
```python
YNET[t] = YGROSS[t] * (1 - DAMFRAC[t])
Y[t]    = YNET[t] - ABATECOST[t]
C[t]    = max(Y[t] - I_inv[t], 2.0)
CPC[t]  = max(1000.0 * C[t] / L[t], 0.01)
```

---

### 3.2 CO₂ emissions

**Plain language:** Total CO₂ emissions depend on how much output the economy
produces, how carbon-intensive that production is (σ), and how much of those
emissions are cut (MIU). Land-use change (deforestation) adds a separate
exogenous source.

**Equations:**

$$E_{\text{CO}_2}(t) = \bigl[\sigma(t) \cdot Y_{\text{gross}}(t) + E_{\text{land}}(t)\bigr] \cdot \bigl(1 - \text{MIU}(t)\bigr)$$

$$\sigma(t+1) = \sigma(t) \cdot e^{5 \cdot g_\sigma(t)}, \quad g_\sigma(t) = \min\!\bigl(g_{\sigma,0} \cdot \delta_g^{t-1},\; g_{\sigma,\infty}\bigr)$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\sigma(t)$ | CO₂ intensity of output (GtCO₂ per trill USD) | Declines over time; computed in `precompute.py` |
| $E_{\text{land}}(t)$ | Land-use CO₂ emissions (GtCO₂/yr) | $5.9 \times (1-0.1)^{t-1}$; in `precompute.py` |
| $\text{MIU}(t)$ | Emission control rate | Choice variable; $\in [0,\, \text{miuup}(t)]$ |
| $g_{\sigma,0}$ | Initial decarbonisation rate | −0.015/yr (`params.py: gsigma1`) |

Cumulative emissions are tracked for the carbon cycle:

$$\text{CCATOT}(t+1) = \text{CCATOT}(t) + E_{\text{CO}_2}(t) \cdot \frac{5}{3.666}$$

(The factor 3.666 = 44/12 converts GtCO₂ to GtC.)

**Code:** `forward.py` → `simulate()`:
```python
ECO2[t]   = (sigma[t] * YGROSS[t] + eland[t]) * (1 - miu[t])
CCATOT[t+1] = CCATOT[t] + ECO2[t] * (tstep / 3.666)
```
The $\sigma(t)$ array is built in `precompute.py` → `build()`.

---

### 3.3 The carbon cycle (FAIR model)

**Plain language:** CO₂ does not simply accumulate in the atmosphere. Oceans
and land absorb a large fraction, but at different speeds. The FAIR model
represents this with four reservoirs — from a permanent geological sink to a
fast biosphere exchange. The absorption rate also slows as more carbon
accumulates (ocean saturation), captured by the scaling factor **α**.

#### Reservoir dynamics

**Equations:** For each reservoir $i \in \{0,1,2,3\}$:

$$R_i(t+1) = \underbrace{\varepsilon_i \cdot \tau_i \cdot \alpha(t+1) \cdot \frac{E_{\text{CO}_2}(t+1)}{3.667} \cdot \left(1 - e^{-\Delta t / (\tau_i \cdot \alpha(t+1))}\right)}_{\text{new carbon absorbed from emissions}} + \underbrace{R_i(t) \cdot e^{-\Delta t / (\tau_i \cdot \alpha(t+1))}}_{\text{carbon remaining from prior periods}}$$

| Symbol | Meaning | Values |
|--------|---------|--------|
| $\varepsilon_i$ | Emission share flowing into reservoir $i$ | 0.2173, 0.224, 0.2824, 0.2763 |
| $\tau_i$ | Decay time constant (years) | $10^6$, 394.4, 36.53, 4.304 |
| $\alpha(t)$ | Carbon-decay scaling factor (> 1 as saturation grows) | Solved implicitly (see below) |

Atmospheric CO₂ is the sum of all reservoirs plus the pre-industrial baseline:

$$\text{MAT}(t+1) = M_{\text{eq}} + R_0(t+1) + R_1(t+1) + R_2(t+1) + R_3(t+1)$$

**Code:** `forward.py` → `_climate_from_alpha()`:
```python
def _res(es, tau, r_prev):
    at = alpha * tau
    return es * at * eco2_gtc * (1 - np.exp(-P.tstep / at)) + r_prev * np.exp(-P.tstep / at)
```

---

#### The implicit equation for α — the hardest part of the model

**Plain language:** The scaling factor α tells us how much the ocean's
absorption capacity has been reduced by cumulative carbon build-up and warming.
It is defined implicitly: α is the value that makes the model's 100-year
impulse response (how much of a pulse of CO₂ remains in the atmosphere after
100 years) consistent with observed ocean/biosphere behaviour.

**Equation:** α satisfies:

$$\underbrace{\sum_{i=0}^{3} \alpha \cdot \varepsilon_i \cdot \tau_i \cdot \left(1 - e^{-100/(\alpha \cdot \tau_i)}\right)}_{\text{LHS: IRF}_{100} \text{ implied by } \alpha} = \underbrace{\text{IRF}_0 + \text{irC} \cdot \text{CACC}(t) + \text{irT} \cdot \text{TATM}(t)}_{\text{RHS: IRF}_{100} \text{ from carbon and warming}}$$

where $\text{CACC}(t) = \text{CCATOT}(t) - (\text{MAT}(t) - M_{\text{eq}})$ is the cumulative carbon absorbed by sinks.

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\text{IRF}_0$ | Pre-industrial impulse response at 100 yr | 32.4 yr |
| $\text{irC}$ | IRF increase per GtC of cumulative uptake | 0.019 yr/GtC |
| $\text{irT}$ | IRF increase per degree of warming | 4.165 yr/K |

**Key point:** Both sides depend on α (through MAT → CACC and through
temperature). This means we cannot solve for α directly — we must find the
root of $\text{LHS}(\alpha) - \text{RHS}(\alpha) = 0$ numerically at every
time step.

**Code:** `forward.py` → `_irflhs()` computes the LHS; `_solve_alpha()`
uses `scipy.optimize.brentq` to find the root:
```python
def _irflhs(alpha):
    return sum(alpha * es * tau * (1 - np.exp(-100.0 / (alpha * tau)))
               for es, tau in [(P.emshare0, P.tau0), ...])

alpha_next = brentq(residual, 0.1, 100.0)
```

---

### 3.4 Radiative forcing and temperature

**Plain language:** More atmospheric CO₂ traps more heat from the sun (the
greenhouse effect). The extra trapped energy, called radiative forcing, then
gradually warms both the upper ocean and the deep ocean at different speeds.
The temperature we care about (TATM) is the sum of both boxes.

#### Radiative forcing

**Equation:**

$$F(t) = F_{2\times\text{CO}_2} \cdot \log_2\!\left(\frac{\text{MAT}(t)}{M_{\text{eq}}}\right) + F_{\text{misc}}(t) + F_{\text{GHGabate}}(t)$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| $F_{2\times\text{CO}_2}$ | Forcing from doubling CO₂ | 3.93 W/m² |
| $M_{\text{eq}}$ | Pre-industrial atmospheric CO₂ | 588 GtC |
| $F_{\text{misc}}(t)$ | Non-abatable forcing (aerosols, etc.) | Exogenous; in `precompute.py` |
| $F_{\text{GHGabate}}(t)$ | Forcing from abatable non-CO₂ GHGs | State variable |

**Code:** `forward.py` → `simulate()`:
```python
FORC[t] = P.fco22x * np.log(MAT[t] / P.mateq) / np.log(2) + F_Misc[t] + F_GHGabate[t]
```

---

#### Temperature dynamics (two-box model)

**Plain language:** Heat spreads from the atmosphere into the ocean slowly.
The upper ocean responds within years; the deep ocean responds over centuries.
The two-box structure captures this lag. Crucially, past emissions keep warming
the planet for decades even after emissions stop — this is **committed warming**.

**Equations:**

$$\text{TBOX}_1(t+1) = \text{TBOX}_1(t) \cdot e^{-\Delta t/d_1} + \theta_1 \cdot F(t+1) \cdot \left(1 - e^{-\Delta t/d_1}\right)$$

$$\text{TBOX}_2(t+1) = \text{TBOX}_2(t) \cdot e^{-\Delta t/d_2} + \theta_2 \cdot F(t+1) \cdot \left(1 - e^{-\Delta t/d_2}\right)$$

$$\text{TATM}(t+1) = \text{TBOX}_1(t+1) + \text{TBOX}_2(t+1)$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| $d_1$ | Deep ocean thermal response time | 236 years |
| $d_2$ | Upper ocean thermal response time | 4.07 years |
| $\theta_1, \theta_2$ | Thermal equilibration parameters | 0.324, 0.44 m²/KW |

**Code:** `forward.py` → `_climate_from_alpha()`:
```python
tbox1 = tbox1_prev * np.exp(-P.tstep/P.d1) + P.teq1 * forc * (1 - np.exp(-P.tstep/P.d1))
tbox2 = tbox2_prev * np.exp(-P.tstep/P.d2) + P.teq2 * forc * (1 - np.exp(-P.tstep/P.d2))
tatm  = max(min(tbox1 + tbox2, 20.0), 0.5)
```

---

### 3.5 Damages

**Plain language:** Higher temperatures reduce economic output — crops fail,
infrastructure is damaged, heat reduces labour productivity. DICE captures this
with a simple quadratic function of temperature. A 3 °C warming costs about
3 % of world GDP every year, permanently.

**Equation:**

$$\text{DAMFRAC}(t) = a_1 \cdot \text{TATM}(t) + a_2 \cdot \text{TATM}(t)^{a_3}$$

$$\text{DAMAGES}(t) = Y_{\text{gross}}(t) \cdot \text{DAMFRAC}(t)$$

| Symbol | Meaning | Default value |
|--------|---------|---------------|
| $a_1$ | Linear damage coefficient | 0 (no linear term) |
| $a_2$ | Quadratic damage coefficient | **0.003467** (`params.py: a2base`) |
| $a_3$ | Damage exponent | 2.00 |

**Code:** `forward.py` → `simulate()`:
```python
DAMFRAC[t] = P.a1 * TATM[t] + _a2base * TATM[t]**P.a3
DAMAGES[t] = YGROSS[t] * DAMFRAC[t]
```

> **Why this is contested:** At 3 °C the model says damages are 3 %. Many
> economists and climate scientists argue this severely underestimates risk,
> especially for warming above 3–4 °C. The `altdam` scenario triples $a_2$ to
> 0.01 to explore this sensitivity.

---

### 3.6 Abatement costs

**Plain language:** Cutting emissions requires switching to cleaner but more
expensive energy, retrofitting industry, or deploying carbon capture. The cost
rises steeply as you try to eliminate the last units of emissions (convex cost
function). The model anchors costs to a **backstop technology** — a catch-all
clean alternative (e.g. direct air capture + green hydrogen) whose price falls
over time.

**Equations:**

$$\text{ABATECOST}(t) = Y_{\text{gross}}(t) \cdot \Theta_1(t) \cdot \text{MIU}(t)^{\theta_2}$$

where the cost scaling factor is:

$$\Theta_1(t) = \frac{p_{\text{back}}(t) \cdot \sigma_{\text{tot}}(t)}{\theta_2 \cdot 1000}$$

The **marginal abatement cost** (carbon price consistent with MIU) is:

$$\text{CPRICE}(t) = p_{\text{back}}(t) \cdot \text{MIU}(t)^{\theta_2 - 1}$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\theta_2$ | Exponent of control cost function | 2.6 (`params.py: expcost2`) |
| $p_{\text{back}}(t)$ | Backstop price (2019 USD/tCO₂) | $515 in 2050, declining |
| $\sigma_{\text{tot}}(t)$ | Total GHG-output ratio (incl. non-CO₂) | From `precompute.py` |

**Code:** `forward.py` → `simulate()`:
```python
ABATECOST[t] = YGROSS[t] * cost1tot[t] * miu[t]**P.expcost2
CPRICE[t]    = pbacktime[t] * miu[t]**(P.expcost2 - 1)
```
`cost1tot` is precomputed in `precompute.py` → `build()`.

> **The Social Cost of Carbon** at the welfare optimum equals CPRICE. This is
> because the optimiser sets the marginal benefit of abatement (avoided damages)
> equal to the marginal cost — which is CPRICE. At the optimum: **SCC = CPRICE**.

---

### 3.7 The welfare function

**Plain language:** The model maximises the sum of discounted, population-
weighted utility over all periods and all generations. Think of it as the total
"well-being" of all humans alive over the next 400 years, weighted by how many
there are and discounted because future welfare counts slightly less than
present welfare (the "time preference").

**Equations:** Instantaneous utility of per-capita consumption:

$$\text{PERIODU}(t) = \frac{\text{CPC}(t)^{1-\eta} - 1}{1 - \eta} - 1$$

Period welfare (population-weighted and discounted):

$$\text{TOTPERIODU}(t) = \text{PERIODU}(t) \cdot L(t) \cdot R(t)$$

Total welfare (the objective to maximise):

$$\mathcal{W} = \Delta t \cdot s_1 \cdot \sum_{t=1}^{81} \text{TOTPERIODU}(t) + s_2$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\eta$ | Elasticity of marginal utility of consumption | 0.95 (`params.py: elasmu`) |
| $R(t)$ | Discount factor (see §3.8) | Computed in `precompute.py` |
| $s_1, s_2$ | Scaling constants (so that $\mathcal{W} \approx$ PV consumption) | 0.00891061, −6275.91 |

**Code:** `forward.py` → `simulate()`:
```python
PERIODU[t]    = (CPC[t]**(1 - _elasmu) - 1) / (1 - _elasmu) - 1
TOTPERIODU[t] = PERIODU[t] * L[t] * RR[t]
...
UTILITY = P.tstep * P.scale1 * np.sum(TOTPERIODU) + P.scale2
```

> **Note on $\eta$:** When $\eta$ is high, the marginal utility of consumption
> falls fast — meaning richer future generations count less in welfare per dollar
> of consumption. When $\eta$ is close to 0, welfare is nearly linear in
> consumption and all generations are treated almost equally per dollar.

---

### 3.8 Discount rates

**Plain language:** A dollar of welfare in the future is worth less than a
dollar today — both because people are impatient (pure time preference) and
because people expect to be richer in the future (growth effect). DICE also
adds a **precautionary premium** to account for uncertainty about future
consumption growth.

**Equations:**

The risk-adjusted pure rate of time preference:

$$\rho^* = e^{\bar{\rho} + \beta_{\text{clim}} \cdot \pi} - 1$$

The **precautionary discount factor** accounts for consumption growth uncertainty:

$$r_{\text{prec}}(t) = -\frac{1}{2} \cdot \text{Var}[\Delta \ln C] \cdot \eta^2, \quad \text{where} \quad \text{Var}(t) = \min\!\bigl(\sigma_{gc}^2 \cdot 5(t-1),\; \sigma_{gc}^2 \cdot 5 \cdot 47\bigr)$$

The full discount factor:

$$R(t) = \underbrace{\frac{1}{(1+\rho^*)^{5(t-1)}}}_{\text{pure time preference}} \cdot \underbrace{(1 + r_{\text{prec}}(t))^{-5(t-1)}}_{\text{precautionary term}}$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\bar{\rho}$ | Pure rate of social time preference | 0.001/yr (`params.py: prstp`) |
| $\beta_{\text{clim}}$ | Climate beta (risk loading) | 0.5 |
| $\pi$ | Capital risk premium | 0.05 |
| $\sigma_{gc}$ | Annual std dev of consumption growth | 0.01 |

**Code:** `precompute.py` → `build()`:
```python
rartp    = np.exp(_prstp + P.betaclim * P.pi) - 1
varpcc   = np.minimum(P.siggc1**2 * 5 * (tv-1), P.siggc1**2 * 5 * 47)
rprecaut = -0.5 * varpcc * _elasmu**2
RR1      = 1.0 / (1 + rartp)**(P.tstep * (tv-1))
RR       = RR1 * (1 + rprecaut)**(-P.tstep * (tv-1))
```

> **The discount rate debate:** Nordhaus uses $\bar{\rho} \approx 1\%$/yr →
> moderate abatement. Stern uses $\bar{\rho} \approx 0.1\%$/yr → aggressive
> abatement. The five `disc` scenarios let you see the full sensitivity.

---

## 4. The 11 scenarios

| Key | Name | What changes | Question answered |
|-----|------|-------------|-------------------|
| `opt` | **Optimal** | Nothing — pure welfare maximisation | What is the socially optimal emission path? |
| `base` | **Baseline** | Carbon price capped at current policy level; full decarbonisation forced after 2305 | What happens under today's weak policies? |
| `T2` | **2 °C** | Hard ceiling TATM ≤ 2 °C | Cost of meeting the Paris Agreement upper target |
| `T15` | **1.5 °C** | Hard ceiling TATM ≤ 1.5 °C | Cost of meeting the Paris Agreement stretch target |
| `altdam` | **Alt Damages** | $a_2$ tripled from 0.003467 to 0.01 | What if damages are far worse than assumed? |
| `paris` | **Paris** | MIU constrained to a Paris-NDC-consistent ramp | What does actual Paris policy achieve? |
| `disc1` | **1% Discount** | $\bar{\rho}$ = 1 %, $k_0$ = 420 | Low discount → high SCC, more action |
| `disc2` | **2% Discount** | $\bar{\rho}$ = 2 %, $k_0$ = 409 | ↕ |
| `disc3` | **3% Discount** | $\bar{\rho}$ = 3 %, $k_0$ = 370 | ↕ |
| `disc4` | **4% Discount** | $\bar{\rho}$ = 4 %, $k_0$ = 326 | ↕ |
| `disc5` | **5% Discount** | $\bar{\rho}$ = 5 %, $k_0$ = 290 | High discount → low SCC, less action |

---

## 5. Installation

Requires **Python 3.9 or later**.

```bash
pip install numpy scipy          # required
pip install matplotlib pandas    # for visualisation
```

No GAMS licence required. No other dependencies.

---

## 6. How to run

```bash
cd "DICE Python"

# Run all 11 scenarios  (~10-30 min depending on your machine)
python run_dice.py

# Run specific scenarios only (much faster for exploration)
python run_dice.py opt
python run_dice.py opt T2 paris disc1 disc5
```

Output is written to **`DICE2023-Python.csv`** in the same folder.
A summary table is also printed to the console.

---

## 7. Expected output

After running `python run_dice.py opt` you should see values close to:

| Variable | 2020 | 2050 | 2100 |
|----------|------|------|------|
| TATM (°C above pre-industrial) | 1.25 | ~1.7 | ~2.8 |
| MAT — atmospheric CO₂ (GtC) | 887 | ~1 000 | ~1 050 |
| CO₂ concentration (ppm) | ~416 | ~470 | ~490 |
| MIU — emission control rate | 0.05 | ~0.45 | ~0.85 |
| SCC — social cost of carbon ($/tCO₂) | ~5–15 | ~30–60 | ~100–200 |
| CPC — per-capita consumption (k$/yr) | ~17 | ~25 | ~50 |
| YGROSS — world gross output (trill $) | ~136 | ~200 | ~400 |

> **For first-time users:** Start with `opt` and `base`. The gap in temperature
> shows the value of optimal climate policy. Then compare `disc1` vs `disc5` —
> this single parameter change (discount rate) shifts the SCC by a factor of 5
> or more, illustrating why economists disagree so sharply on the "right" carbon
> price.

---

## 8. How to visualise the results

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('DICE2023-Python.csv')

def get_var(df, variable):
    """Extract one variable for all scenarios as a DataFrame indexed by year."""
    sub = df[df['Variable'] == variable].drop(columns='Variable')
    sub = sub.set_index('Scenario').T
    sub.index = range(2020, 2425, 5)
    return sub.apply(pd.to_numeric, errors='coerce')

# ── Chart 1: Temperature trajectories ────────────────────────────────────────
tatm = get_var(df, 'TATM')
fig, ax = plt.subplots(figsize=(10, 5))
for label in ['Optimal', '2 Deg C', '1.5 Deg C', 'Baseline']:
    if label in tatm.columns:
        ax.plot(tatm.index, tatm[label], label=label)
ax.axhline(2.0, color='orange', linestyle='--', lw=1, label='2 °C target')
ax.axhline(1.5, color='red',    linestyle='--', lw=1, label='1.5 °C target')
ax.set(xlabel='Year', ylabel='Temperature anomaly (°C)',
       title='Global Mean Temperature — DICE2023', xlim=(2020, 2200))
ax.legend(); plt.tight_layout(); plt.savefig('temperature.png', dpi=150); plt.show()

# ── Chart 2: Social Cost of Carbon ───────────────────────────────────────────
scc = get_var(df, 'SCC')
fig, ax = plt.subplots(figsize=(10, 5))
for label in ['Optimal', '2 Deg C', 'Alt Damages', '1pct Discount', '5pct Discount']:
    if label in scc.columns:
        ax.plot(scc.index, scc[label], label=label)
ax.set(xlabel='Year', ylabel='SCC (2019 $/tCO₂)',
       title='Social Cost of Carbon — DICE2023', xlim=(2020, 2150))
ax.legend(); plt.tight_layout(); plt.savefig('scc.png', dpi=150); plt.show()

# ── Chart 3: Emission control rate ───────────────────────────────────────────
miu = get_var(df, 'MIU')
fig, ax = plt.subplots(figsize=(10, 5))
for label in miu.columns:
    ax.plot(miu.index, miu[label], label=label, alpha=0.8)
ax.set(xlabel='Year', ylabel='MIU (0 = no control, 1 = full decarbonisation)',
       title='Emission Control Rate — All Scenarios',
       xlim=(2020, 2200), ylim=(0, 1.15))
ax.legend(fontsize=8, ncol=2); plt.tight_layout()
plt.savefig('miu.png', dpi=150); plt.show()

# ── Chart 4: SCC vs discount rate (bar chart, year 2020) ─────────────────────
disc_labels = ['1pct Discount','2pct Discount','3pct Discount',
               '4pct Discount','5pct Discount']
disc_rates  = [0.01, 0.02, 0.03, 0.04, 0.05]
scc_2020    = [scc.loc[2020, l] for l in disc_labels if l in scc.columns]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(disc_rates[:len(scc_2020)], scc_2020, width=0.006, color='steelblue')
ax.set(xlabel='Pure rate of time preference (prstp)',
       ylabel='SCC in 2020 ($/tCO₂)',
       title='Social Cost of Carbon vs Discount Rate')
plt.tight_layout(); plt.savefig('scc_discount.png', dpi=150); plt.show()
```

Replace `'TATM'`, `'SCC'`, or `'MIU'` with any variable listed in
`dice2023/output.py` → `TIME_SERIES_VARS`.

---

## 9. Code structure — file by file

```
DICE Python/
├── run_dice.py              ← Entry point
└── dice2023/
    ├── params.py            ← All scalar constants          (§ 3.1–3.8)
    ├── precompute.py        ← Exogenous time-series arrays  (§ 3.1, 3.2, 3.8)
    ├── forward.py           ← Model equations               (§ 3.1–3.7)
    ├── optimize.py          ← The NLP solver
    ├── scenarios.py         ← Scenario catalogue
    └── output.py            ← CSV writer
```

---

### `params.py`

**What it contains:** Every scalar constant from the three GAMS source files,
verbatim. Nothing is computed here.

**What equations live here:** All parameter values for §3.1–3.8 above.
If you want to change any fundamental assumption (damage coefficient, discount
rate, backstop cost…) this is where you do it.

```python
a2base    = 0.003467   # damage quadratic coefficient — eq. DAMFRAC
pback2050 = 515.0      # backstop cost in 2050 — eq. CPRICE / ABATECOST
prstp     = 0.001      # pure rate of time preference — eq. R(t)
elasmu    = 0.95       # elasticity of marginal utility — eq. PERIODU
fco22x    = 3.93       # CO2 forcing sensitivity — eq. FORC
```

---

### `precompute.py` — function `build(**kwargs)`

**What it does:** Computes all time-series arrays that do not depend on policy
choices (MIU, S). Must be called before `simulate()`.

**Equations implemented here:**

| Array built | Equation section | Key formula |
|-------------|-----------------|-------------|
| `L(t)` | §3.1 | $L(t+1) = L(t) \cdot (L_\infty / L(t))^{p_{\text{adj}}}$ |
| `aL(t)`, `gA(t)` | §3.1 | $A(t+1) = A(t)/(1-g_A(t))$, $g_A(t) = g_{A,0} \cdot e^{-\delta_A \cdot 5(t-1)}$ |
| `sigma(t)`, `gsig(t)` | §3.2 | $\sigma(t+1) = \sigma(t) \cdot e^{5 g_\sigma(t)}$ |
| `eland(t)` | §3.2 | $E_{\text{land}}(t) = E_{\text{land},0} \cdot (1-0.1)^{t-1}$ |
| `pbacktime(t)` | §3.6 | Backstop price trajectory |
| `miuup(t)` | §3.2 | Technical ceiling on MIU, piecewise |
| `RR(t)`, `RR1(t)` | §3.8 | Full discount factor with precautionary term |
| `cost1tot(t)` | §3.6 | $\Theta_1(t) = p_{\text{back}}(t) \cdot \sigma_{\text{tot}}(t) / (\theta_2 \cdot 1000)$ |

Keyword arguments override defaults for scenario variations:
- `a2base=0.01` — alternative damages
- `prstp=0.03, elasmu=0.001, k0=370, no_precaution=True` — 3% discount scenario
- `miuup_paris=True` — Paris-ramp MIU ceiling
- `base_scenario=True` — baseline carbon-price constraint on MIU

---

### `forward.py` — function `simulate(miu, srate, par)`

**What it does:** Given MIU and S arrays and the precomputed parameter dict,
propagates all 40+ state variables forward in time and returns them as a
dictionary of numpy arrays.

**Equations implemented here:** All equations in §3.1–3.7.

**The simulation loop** (pseudocode):
```
for t = 0 to 80:
    1.  YGROSS[t]   ← production function §3.1
    2.  DAMFRAC[t]  ← damage function §3.5
    3.  ABATECOST[t]← abatement cost §3.6
    4.  Y[t], C[t]  ← output accounting §3.1
    5.  ECO2[t]     ← emissions §3.2
    6.  FORC[t]     ← radiative forcing §3.4
    7.  PERIODU[t]  ← welfare §3.7

    if t < 80:
        K[t+1]      ← capital accumulation §3.1
        CCATOT[t+1] ← cumulative emissions §3.2
        solve α[t+1] via brentq (§3.3) using ECO2[t+1] preview
        → R0..R3[t+1], MAT[t+1], TATM[t+1]  (§3.3, §3.4)
```

**The α solve** is the only non-trivial numerical step. `_solve_alpha()` calls
`_climate_from_alpha()` inside a Brent root-finding loop until
$|\text{LHS}(\alpha) - \text{RHS}(\alpha)| < 10^{-8}$.

---

### `optimize.py` — function `run(par, ...)`

**What it does:** Wraps the forward simulation as a minimisation objective and
finds the MIU and S arrays that maximise welfare.

**The optimisation problem:**

$$\max_{\text{MIU}(t),\, S(t)} \mathcal{W}(\text{MIU}, S)$$

$$\text{subject to:} \quad 0 \le \text{MIU}(t) \le \text{miuup}(t), \quad 0.01 \le S(t) \le 0.99 \quad (t \le 37)$$

The 117 free variables are $[\text{MIU}(2), \ldots, \text{MIU}(81), S(1), \ldots, S(37)]$.
($\text{MIU}(1) = 0.05$ is fixed at its historical value; $S(t > 37) = 0.28$.)

Uses **SLSQP** from scipy with three optimisation passes from the best solution
found (mirrors GAMS's triple-solve for robustness).

For temperature-constrained scenarios, the penalty term
$-\lambda \sum_t \max(0, \text{TATM}(t) - T_{\max})^2$ is added to the
objective, with $\lambda = 50\,000$.

---

### `scenarios.py`

**What it does:** Defines all 11 scenarios as a dictionary. Each entry maps to
a GAMS `Include/def-*.gms` file and specifies:
- `build_kwargs` → overrides for `precompute.build()`
- `tatm_max` → temperature ceiling (None = unconstrained)
- `miu_fixed_after` → GAMS t.val after which MIU is fixed (base scenario)

`run_scenario(name)` and `run_all()` orchestrate the full pipeline.

---

### `output.py`

**What it does:** Writes a CSV file with one row per (variable, scenario)
pair and years as columns — matching the format of the original GAMS output
file. Also prints a summary table to the console.

---

## 10. How to add a new scenario

**Step 1** — Add an entry to `SCENARIOS` in `dice2023/scenarios.py`:

```python
'stern': {
    'label': 'Stern Review',
    'build_kwargs': {
        'prstp':         0.001,   # 0.1 %/yr — Stern's near-zero rate
        'elasmu':        1.0,     # unit elasticity (log utility)
        'no_precaution': True,
    },
    'tatm_max':        None,
    'miu_fixed_after': None,
},
```

**Step 2** — Optionally add `'stern'` to `RUN_ORDER`.

**Step 3** — Run it:
```bash
python run_dice.py stern
```

**Parameters available in `build_kwargs`:**

| Parameter | Default | Equation affected |
|-----------|---------|-------------------|
| `prstp` | 0.001 | Discount factor $R(t)$ — §3.8 |
| `elasmu` | 0.95 | Utility function PERIODU — §3.7 |
| `k0` | 295 | Initial capital $K(0)$ — §3.1 |
| `a2base` | 0.003467 | Damage fraction DAMFRAC — §3.5 |
| `no_precaution` | `False` | Removes precautionary term from $R(t)$ — §3.8 |
| `miuup_paris` | `False` | Paris-ramp ceiling on MIU — §3.2 |
| `base_scenario` | `False` | Carbon-price ceiling on MIU — §3.2 |

Set `tatm_max` to a temperature in °C to add a hard temperature ceiling.

---

## 11. Limitations

| Limitation | Practical consequence |
|------------|----------------------|
| **One world region** | No distributional analysis across countries or income groups |
| **Deterministic** | No climate uncertainty, no tipping points, no tail risks — likely underestimates the value of aggressive action |
| **Quadratic damage function** | Poorly constrained above 3–4 °C; most estimates suggest damages accelerate nonlinearly at high temperatures |
| **Reduced-form FAIR climate** | Calibrated to CMIP5 ensemble medians — cannot reproduce regional effects or extreme events |
| **SCC proxy** | At the welfare optimum SCC = CPRICE (exact). For constrained scenarios (T2, T15, Paris) CPRICE is the marginal abatement cost, not the welfare-based SCC |
| **Optimiser sensitivity** | SLSQP is a local solver; results may vary slightly with starting point. The 3-pass approach reduces but does not eliminate this sensitivity |

---

## 12. References

**Primary source:**
Nordhaus, W.D. (2023). *DICE2023 Model*. Yale University.

**Original DICE paper:**
Nordhaus, W.D. (1992). An optimal transition path for controlling greenhouse gases.
*Science*, 258(5086), 1315–1319.

**Updated calibration and SCC estimates:**
Nordhaus, W.D. (2017). Revisiting the social cost of carbon.
*PNAS*, 114(7), 1518–1523.

**The FAIR carbon-cycle model (used verbatim in this code):**
Millar, R.J. et al. (2017). A modified impulse-response representation of the
global near-surface air temperature and atmospheric concentration response to
carbon dioxide emissions.
*Atmospheric Chemistry and Physics*, 17, 7213–7228.

**The discount rate debate:**
Stern, N. (2007). *The Economics of Climate Change: The Stern Review*.
Cambridge University Press.

Weitzman, M.L. (2007). A review of the Stern Review on the economics of climate
change. *Journal of Economic Literature*, 45(3), 703–724.

Nordhaus, W.D. (2007). A review of the Stern Review on the economics of climate
change. *Journal of Economic Literature*, 45(3), 686–702.
