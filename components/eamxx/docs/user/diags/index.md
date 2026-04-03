# Online diagnostics

EAMxx has facilities to output optional diagnostics
that are computed during runtime. These diagnostics
are designed generically and composably, and are requestable by users.

## Quick reference

The table below summarizes every diagnostic available in EAMxx
output YAML files. Each diagnostic is triggered by a naming
pattern in the `field_names` list.

### Built-in derived fields

These are standalone diagnostics that compute a specific quantity.
Request them by their exact name.

| YAML name | Description |
| --------- | ----------- |
| `PotentialTemperature` | Total potential temperature |
| `LiqPotentialTemperature` | Liquid potential temperature |
| `VirtualTemperature` | Virtual temperature |
| `AtmosphereDensity` | Atmospheric density |
| `Exner` | Exner function |
| `DryStaticEnergy` | Dry static energy |
| `SeaLevelPressure` | Sea level pressure |
| `RelativeHumidity` | Relative humidity |
| `wind_speed` | Horizontal wind speed |
| `AerosolOpticalDepth550nm` | Aerosol optical depth at 550 nm |
| `dz` | Layer thickness |
| `z_mid`, `z_int` | Height at midpoints / interfaces |
| `geopotential_mid`, `geopotential_int` | Geopotential at midpoints / interfaces |
| `height_mid`, `height_int` | Height at midpoints / interfaces |

### Parameterized diagnostics

These diagnostics are triggered by a naming pattern.
Replace `X` with a field name.

| Pattern | Example | Description |
| ------- | ------- | ----------- |
| `{Kind}WaterPath` | `LiqWaterPath` | Water path (Kind: Ice, Liq, Rain, Rime, Vap) |
| `{Kind}NumberPath` | `IceNumberPath` | Number path (Kind: Ice, Liq, Rain) |
| `{Dir}VapFlux` | `ZonalVapFlux` | Vapor flux (Dir: Meridional, Zonal) |
| `precip_{type}_surf_mass_flux` | `precip_liq_surf_mass_flux` | Precip surface mass flux (type: liq, ice, total) |
| `X_at_lev_{N}` | `T_mid_at_lev_5` | Field at model level N |
| `X_at_model_top` | `T_mid_at_model_top` | Field at model top |
| `X_at_model_bot` | `T_mid_at_model_bot` | Field at model bottom |
| `X_at_{P}hPa` | `T_mid_at_500hPa` | Field interpolated to pressure level |
| `X_at_{H}m_above_sealevel` | `T_mid_at_100m_above_sealevel` | Field interpolated to height |
| `X_atm_backtend` | `T_mid_atm_backtend` | Atmospheric tendency |

### Arithmetic diagnostics

| Pattern | Example | Description | Details |
| ------- | ------- | ----------- | ------- |
| `X_{op}_Y` | `T_mid_plus_p_mid` | Binary arithmetic | [Binary ops](binary_ops.md) |
| `{op}_of_X` | `sqrt_of_T_mid` | Unary function | [Unary ops](unary_ops.md) |
| `expr_{expression}` | `expr_sqrt(U**2+V**2)` | Free-form expression | [Expressions](expressions.md) |

### Reduction and sampling diagnostics

| Pattern | Example | Description | Details |
| ------- | ------- | ----------- | ------- |
| `X_horiz_avg` | `T_mid_horiz_avg` | Horizontal average | [Field contraction](field_contraction.md) |
| `X_vert_avg` | `T_mid_vert_avg` | Vertical average | [Field contraction](field_contraction.md) |
| `X_vert_sum_dp_weighted` | `T_mid_vert_sum_dp_weighted` | Weighted vertical sum | [Field contraction](field_contraction.md) |
| `X_zonal_avg_{N}_bins` | `T_mid_zonal_avg_36_bins` | Zonal average | [Field contraction](field_contraction.md) |
| `X_histogram_{bins}` | `T_mid_histogram_250_270_290` | Histogram | [Field contraction](field_contraction.md) |
| `X_where_Y_{op}_{val}` | `T_mid_where_qv_gt_0.01` | Conditional sampling | [Conditional sampling](conditional_sampling.md) |
| `X_pvert_derivative` | `T_mid_pvert_derivative` | Pressure vertical derivative | [Vertical derivative](vert_derivative.md) |
| `X_zvert_derivative` | `T_mid_zvert_derivative` | Height vertical derivative | [Vertical derivative](vert_derivative.md) |

## Composability

Diagnostics can be composed by chaining suffixes. For example:

```yaml
field_names:
  # vertical sum, then horizontal average
  - T_mid_vert_sum_dz_weighted_horiz_avg
  # conditional sample, then vertical average
  - T_mid_where_qv_gt_0.01_vert_avg
```

Composition works because the output of one diagnostic becomes
the input to the next. However, not all combinations are valid.
See the individual diagnostic pages for caveats.

## Choosing the right diagnostic

| I want to... | Use |
| ------------ | --- |
| Output a single known field | Just put the field name in `field_names` |
| Add/subtract/multiply/divide two fields | [Binary ops](binary_ops.md): `X_plus_Y` |
| Apply sqrt/abs/log/exp to a field | [Unary ops](unary_ops.md): `sqrt_of_X` |
| Compute a complex formula | [Expressions](expressions.md): `expr_sqrt(U**2+V**2)` |
| Average over columns or levels | [Field contraction](field_contraction.md) |
| Filter by a condition | [Conditional sampling](conditional_sampling.md) |
| Compute vertical gradients | [Vertical derivative](vert_derivative.md) |

## Detailed guides

- [Field contraction](field_contraction.md) (horizontal/vertical reductions, zonal averages, histograms)
- [Conditional sampling](conditional_sampling.md)
- [Binary operations](binary_ops.md)
- [Unary operations](unary_ops.md)
- [Expression diagnostics](expressions.md)
- [Vertical derivative](vert_derivative.md)
