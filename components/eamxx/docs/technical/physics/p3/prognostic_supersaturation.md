# Prognostic Supersaturation

This document describes the prognostic supersaturation treatment in EAMxx: an
optional replacement for the instantaneous liquid saturation adjustment that
EAMxx normally performs. The treatment spans two components, SHOC and P3, and
is controlled by three runtime switches.

## Purpose

By default, EAMxx assumes instantaneous liquid saturation adjustment, as
described by Caldwell et al. (2021) [@Caldwell_et21]. SHOC's assumed PDF
diagnoses cloud liquid $q_c$ by removing any supersaturation with respect to
liquid within the time step, and P3 inherits an already-saturated state. Two
consequences follow:

1. Supersaturation with respect to liquid can never exist, so the microphysics
   cannot represent the competition between droplet activation, condensational
   growth, and the depletion of vapor by pre-existing droplets.
2. Because $q_c$ is diagnosed rather than predicted, aerosol activation cannot
   respond to the local supersaturation.

The prognostic supersaturation treatment instead lets supersaturation exist and
relaxes it at a finite rate. Condensation and evaporation of cloud liquid move
from SHOC into P3, where they are computed from the local supersaturation and
the supersaturation relaxation timescale.

## Runtime Switches

| Switch | Component | Default | Meaning |
| --- | --- | --- | --- |
| `shoc_enable_condensation` | SHOC | `true` | When `false`, SHOC no longer condenses or evaporates cloud liquid. |
| `p3_super_sat` | P3 | `false` | When `true`, P3 activates droplets from the local supersaturation and condenses/evaporates cloud liquid explicitly. |
| `p3_WBFoff` | P3 | `false` | When `true`, the Bergeron (WBF) sink of cloud liquid to ice is switched off. |

The defaults reproduce the standard EAMxx configuration bit-for-bit. The
intended prognostic-supersaturation configuration is

```yaml
shoc:
  shoc_enable_condensation: false
p3:
  p3_super_sat: true
  p3_WBFoff: true
```

`p3_WBFoff` is recommended together with `p3_super_sat` because the prognostic
treatment already transfers liquid to ice through the explicit
condensation/evaporation term; leaving the Bergeron sink active double counts
that transfer.

Because `p3_super_sat` makes P3 read the vertical velocity and the turbulent
kinetic energy, enabling it adds `omega` and `tke` to P3's required fields.
Both are read-only as far as P3 is concerned.

## SHOC Side: Disabling Condensation

With `shoc_enable_condensation = false`, `shoc_assumed_pdf` skips the entire
SGS condensation branch. Specifically:

- $q_c$ is **not** modified, so the value produced by the previous processes
  persists.
- Liquid cloud fraction is diagnosed all-or-nothing,

  $$
  C_l =
  \begin{cases}
  1 & q_c > 10^{-18}\ \mathrm{kg/kg} \\
  0 & \text{otherwise,}
  \end{cases}
  $$

- All outputs that depend on sub-grid variability or on condensation are set to
  zero: the cloud-liquid variance $q_{c}^{\prime 2}$ (`shoc_ql2`), the liquid
  water flux $\overline{w^\prime q_l^\prime}$ (`wqls`), the buoyancy flux
  $\overline{w^\prime \theta_v^\prime}$ (`wthv_sec`), and the `shoc_cond` /
  `shoc_evap` diagnostics.

Because SHOC no longer sets $q_c$, the SHOC post-process must keep the effect
of vertical diffusion on $q_c$. In the default configuration it restores $q_c$
from `qc_copy`, a copy taken specifically to *exclude* vertical diffusion. With
condensation disabled it instead keeps the diffused $q_c$ and applies a lower
bound of zero.

Zeroing the buoyancy flux removes SHOC's SGS-condensation contribution to
buoyancy production of TKE, so this option is intended to be used with the
1.5-TKE closure (`shoc_1p5tke`).

## P3 Side: Activation

With `p3_super_sat = true`, droplet activation in `p3_main_part1` moves out of
the prescribed-CCN branch and becomes supersaturation-dependent. Activation is
applied wherever the grid-scale vapor exceeds liquid saturation,

$$
q_v > q_{v,\mathrm{sat},l},
$$

and the droplet number is raised to the SPA-derived activated concentration

$$
N_c \leftarrow \max\!\left(N_c,\ \alpha
\left(\frac{N_{\mathrm{CCN}}}{C_l}\right)^{\beta}\right),
$$

where $\alpha$ and $\beta$ are the `spa_ccn_to_nc_factor` and
`spa_ccn_to_nc_exponent` runtime options. Unlike the default treatment, the
newly activated droplets carry mass. Each is assumed to be a sphere of radius
$r_a = 1\ \mu\mathrm{m}$, so

$$
\Delta q_c = \max(\Delta N_c, 0)\ \frac{4}{3}\pi r_a^3 \rho_w,
$$

and that mass is removed from the vapor field with the corresponding latent
heating,

$$
q_c \leftarrow q_c + \Delta q_c, \qquad
q_v \leftarrow q_v - \Delta q_c, \qquad
\theta \leftarrow \theta + \frac{L_v}{c_p}\,\Pi^{-1}\,\Delta q_c,
$$

with $\Pi^{-1}$ the inverse Exner function (`inv_exner`), consistent with every
other latent-heating update in P3.

Activation is only applied when prescribed CCN are available
(`do_prescribed_ccn`), since $N_{\mathrm{CCN}}$ is otherwise unset.

## P3 Side: Condensation and Evaporation

With `p3_super_sat = true`, `p3_main_part2` calls
`prognostic_supersat_cond_evap` for every in-cloud lane
($q_{c,\mathrm{incld}} \ge Q_{\mathrm{small}}$) instead of relying on the
saturation adjustment done upstream.

### Updraft speed

The supersaturation source is set by the vertical velocity, taken as the
resolved part from $\omega$ plus a sub-grid part from TKE. Assuming isotropic
turbulence, $\overline{w^{\prime 2}} = \tfrac{2}{3}e$, so

$$
w = -\frac{\omega}{\rho g} + \sqrt{\max\!\left(0, \tfrac{2}{3}e\right)}.
$$

### Supersaturation budget

Let $s = q_v - q_{v,\mathrm{sat},l}$ be the supersaturation with respect to
liquid. Adiabatic ascent cools the parcel at $\mathrm{d}T/\mathrm{d}t = -wg/c_p$
and therefore produces supersaturation at the rate

$$
A = -\frac{\partial q_{v,\mathrm{sat},l}}{\partial T}
\frac{\mathrm{d}T}{\mathrm{d}t}
= \frac{\partial q_{v,\mathrm{sat},l}}{\partial T}\ \frac{w g}{c_p}.
$$

Pre-existing droplets deplete supersaturation on the relaxation timescale
$\tau = 1/\epsilon_c$, where $\epsilon_c$ is the cloud-liquid relaxation-rate
coefficient computed by `calc_liq_relaxation_timescale`. The supersaturation
then obeys

$$
\frac{\mathrm{d}s}{\mathrm{d}t} = A - \frac{s}{\tau},
$$

whose analytic solution, averaged over a time step $\Delta t$ and divided by
the latent-heating correction factor $a_b$, gives the step-mean condensation
rate

$$
P_{cc} = \frac{1}{a_b}\left[
A\,\epsilon_c \tau
+ \left(s_0 - A\tau\right)
\frac{\epsilon_c \tau}{\Delta t}
\left(1 - e^{-\Delta t/\tau}\right)
\right].
$$

$P_{cc} > 0$ is condensation and $P_{cc} < 0$ is evaporation. Evaporation is
limited so that it can never remove more liquid than is present,

$$
P_{cc} \leftarrow \max\!\left(-\frac{q_{c,\mathrm{incld}}}{\Delta t},\ P_{cc}\right).
$$

The tendency is applied to the grid-mean fields weighted by the liquid cloud
fraction,

$$
q_c \leftarrow q_c + C_l P_{cc}\Delta t, \qquad
q_v \leftarrow q_v - C_l P_{cc}\Delta t, \qquad
\theta \leftarrow \theta + \frac{L_v}{c_p}\Pi^{-1} C_l P_{cc} \Delta t,
$$

after which the in-cloud mixing ratios, the cloud droplet size distribution,
and $\epsilon_c$ are all recomputed from the updated $q_c$ before the rest of
the microphysics runs.

## P3 Side: Switching Off Bergeron

With `p3_WBFoff = true`, `ice_deposition_sublimation` leaves the Bergeron
tendency at its initialized value of zero:

$$
\left(\frac{\partial q_i}{\partial t}\right)_{\mathrm{berg}} = 0.
$$

Vapor deposition, sublimation, and the associated ice-number sink are not
affected.

## Variable Definitions

- $q_c$: cloud liquid mass mixing ratio [kg/kg]
- $q_v$: water vapor mass mixing ratio [kg/kg]
- $q_{v,\mathrm{sat},l}$: saturation vapor mixing ratio w.r.t. liquid [kg/kg]
- $s$: supersaturation w.r.t. liquid, $q_v - q_{v,\mathrm{sat},l}$ [kg/kg]
- $N_c$: cloud droplet number mixing ratio [1/kg]
- $N_{\mathrm{CCN}}$: prescribed CCN concentration from SPA [1/kg]
- $C_l$: liquid cloud fraction [-]
- $e$: turbulent kinetic energy, `tke` [m2/s2]
- $\omega$: vertical pressure velocity, `omega` [Pa/s]
- $w$: vertical velocity used to drive the supersaturation [m/s]
- $\epsilon_c$: cloud-liquid relaxation-rate coefficient, `epsc` [1/s]
- $\tau$: supersaturation relaxation timescale, $1/\epsilon_c$ [s]
- $a_b$: latent-heating correction factor, `ab` [-]
- $\Pi^{-1}$: inverse Exner function, `inv_exner` [-]
- $\rho_w$: density of liquid water [kg/m3]
- $r_a$: assumed radius of a newly activated droplet, $10^{-6}$ m

## Implementation Details

- SHOC condensation switch:
  `components/eamxx/src/physics/shoc/impl/shoc_assumed_pdf_impl.hpp`
- SHOC post-process $q_c$ handling:
  `components/eamxx/src/physics/shoc/eamxx_shoc_process_interface.hpp`
- P3 activation:
  `components/eamxx/src/physics/p3/impl/p3_main_impl_part1.hpp`
- P3 condensation/evaporation rate:
  `components/eamxx/src/physics/p3/impl/p3_prognostic_supersat_cond_evap_impl.hpp`
- P3 call site:
  `components/eamxx/src/physics/p3/impl/p3_main_impl_part2.hpp`
- Bergeron switch:
  `components/eamxx/src/physics/p3/impl/p3_ice_deposition_sublimation_impl.hpp`
- Defaults:
  `components/eamxx/cime_config/namelist_defaults_eamxx.xml`

`prognostic_supersat_cond_evap` writes both of its outputs only inside the
supplied context mask, and both are initialized to zero on entry. $\epsilon_c$
and $a_b$ are only meaningful inside that mask, so the routine substitutes one
for them on the inactive lanes of the pack before forming the reciprocals. This
keeps the inactive lanes finite, which matters for the floating-point-exception
builds.

## Property Tests

The unit tests live in
`components/eamxx/src/physics/p3/tests/p3_prognostic_supersat_cond_evap_unit_tests.cpp`
and
`components/eamxx/src/physics/p3/tests/p3_ice_deposition_sublimation_tests.cpp`.

`prognostic_supersat_cond_evap_property` covers:

- saturated air with no vertical motion produces exactly zero tendency
- supersaturated air condenses, subsaturated air evaporates
- the evaporation limiter is exactly $-q_{c,\mathrm{incld}}/\Delta t$ when the
  air is driven completely dry
- the updraft speed matches $-\omega/(\rho g) + \sqrt{2e/3}$
- ascent produces condensation, stronger ascent produces more of it, and
  subsidence produces evaporation
- masked lanes with $\epsilon_c = a_b = 0$ return finite values, and the
  inactive lanes are left at exactly zero

`ice_deposition_sublimation_wbf_switch` covers:

- a sub-freezing, liquid-supersaturated state produces a non-zero Bergeron
  tendency with `p3_WBFoff = false`
- `p3_WBFoff = true` zeroes that tendency
- deposition, sublimation, and the ice-number sink are bit-for-bit unchanged by
  the switch
