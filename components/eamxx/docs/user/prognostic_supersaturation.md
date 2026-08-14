# Prognostic supersaturation

By default, EAMxx assumes instantaneous liquid saturation adjustment: SHOC's
assumed PDF removes any supersaturation with respect to liquid within the time
step, and P3 inherits an already-saturated state. The prognostic
supersaturation option instead lets supersaturation exist and relaxes it at a
finite rate, moving condensation and evaporation of cloud liquid out of SHOC
and into P3.

For the equations and the implementation, see the
[technical guide page](../technical/physics/p3/prognostic_supersaturation.md).

## Runtime flags

Three flags control the treatment. All three default to the values that
reproduce standard EAMxx behavior bit-for-bit, so the option is off unless you
turn it on.

| Flag | Default | Meaning |
| --- | --- | --- |
| `shoc_enable_condensation` | `true` | When `false`, SHOC no longer condenses or evaporates cloud liquid, and diagnoses liquid cloud fraction all-or-nothing. |
| `p3_super_sat` | `false` | When `true`, P3 activates droplets from the local supersaturation and condenses/evaporates cloud liquid explicitly. |
| `p3_WBFoff` | `false` | When `true`, the Bergeron (Wegener-Bergeron-Findeisen) sink of cloud liquid to ice is switched off. |

## Example setup

The three flags are meant to be set together:

```shell
    ./atmchange shoc_enable_condensation=false
    ./atmchange p3_super_sat=true
    ./atmchange p3_WBFoff=true
```

`p3_WBFoff=true` is recommended whenever `p3_super_sat=true`. The prognostic
treatment already transfers liquid to ice through its explicit
condensation/evaporation term, so leaving the Bergeron sink on double counts
that transfer.

Setting the flags individually is supported but is not a tested configuration:

- `shoc_enable_condensation=false` on its own leaves nothing condensing cloud
  liquid.
- `p3_super_sat=true` on its own means P3 condenses on top of a state that SHOC
  has already saturation-adjusted.

Turning SHOC condensation off also zeroes SHOC's sub-grid buoyancy flux, which
removes the SGS-condensation contribution to buoyancy production of TKE. This
option is therefore intended to be used with the 1.5-TKE closure:

```shell
    ./atmchange shoc_1p5tke=true
```

## Additional inputs required by P3

With `p3_super_sat=true`, P3 reads two fields it does not otherwise need, in
order to estimate the vertical velocity that drives the supersaturation:

- `omega`, the vertical pressure velocity, for the resolved updraft
- `tke`, the turbulent kinetic energy, for the sub-grid updraft

Both are read-only as far as P3 is concerned, and neither is added to P3's
field list when `p3_super_sat` is `false`.

## Diagnostics

With SHOC condensation disabled, the SHOC `shoc_cond` and `shoc_evap`
diagnostics are zero by construction, as are `shoc_ql2`, `wqls`, and
`wthv_sec`. Condensation and evaporation are then P3 processes, so use the P3
extra diagnostics instead:

```shell
    ./atmchange extra_p3_diags=true
```
