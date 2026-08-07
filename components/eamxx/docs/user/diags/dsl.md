# Requesting diagnostics

Diagnostics are requested by writing an expression in the output YAML. The
syntax follows Python and [xarray](https://docs.xarray.dev), so if you know how
to write `da.mean(dim="lev")` you already know most of it.

```yaml
fields:
  physics_pg2:
    field_names:
      - T_vavg := T_mid.weighted('dp').mean(dim='lev')
      - Tbot   := T_mid.isel(lev=-1)
      - T500   := T_mid.interp(plev=500, units='hPa')
      - cloudy := T_mid.where(qc > 1e-5)
```

Plain model fields need no expression: write `T_mid` and you get `T_mid`.

## Naming the output

An expression is not a valid netCDF variable name, so anything other than a
bare name must be given one with `:=`:

```yaml
      - T_vavg := T_mid.weighted('dp').mean(dim='lev')
```

The name on the left is what appears in the file. See
[IO aliases](../io_aliases.md) for the full aliasing mechanism, including the
`aliases:` section for intermediate quantities you need but do not want
written out.

## Operations

### Selecting a level

| | |
|---|---|
| `X.isel(lev=10)` | the 11th level, counting from the model top |
| `X.isel(lev=0)` | the top level |
| `X.isel(lev=-1)` | the bottom level (Python's negative index) |
| `X.interp(plev=500, units='hPa')` | interpolated to a pressure level |
| `X.interp(plev=50000)` | the same, in Pa (the default) |
| `X.interp(z=10, reference='surface')` | interpolated to a height above the surface |
| `X.interp(z=10, reference='sealevel')` | ...or above sea level |

`isel` selects an existing level by index; `interp` interpolates between
levels. Pressure units may be `Pa`, `hPa`, or `mb`; heights are in metres.

### Reductions

| | |
|---|---|
| `X.mean(dim='col')` | average over columns |
| `X.mean(dim='lev')` | average over the column |
| `X.sum(dim='lev')` | vertical sum |
| `X.weighted('dp').mean(dim='lev')` | pressure-thickness weighted average |
| `X.weighted('dz').sum(dim='lev')` | height-weighted sum |

As in xarray, `.weighted()` has no meaning on its own — it modifies the
reduction that follows it. Weighting applies to vertical reductions only.

### Arithmetic

Ordinary operators, with ordinary precedence:

```text
qc + qr
T_mid / p_mid
Rgas * T_mid
(A - B) / C
```

Parentheses decide what groups: `A - B / C` is `A - (B/C)`, and `(A - B) / C`
is what it looks like. An operand may name a physical constant (`Rgas`,
`gravit`, ...) instead of a field.

`X / dt` divides by the timestep and is recognized specially — `dt` is a
reserved name, not a field.

### Everything else

| | |
|---|---|
| `X.where(qc > 1e-5)` | sample where a condition holds |
| `X.shift(time=1)` | the value at the previous timestep |
| `X.tend()` | backward tendency, i.e. `(X - X.shift(time=1)) / dt` |
| `X.differentiate('p')` | vertical derivative with respect to pressure |
| `X.differentiate('z')` | ...or height |
| `X.histogram(bins=[0,1,2])` | histogram over the given bin edges |
| `X.zonal_mean(bins=20)` | zonal average into latitude bins |

Comparisons in `.where()` may use `>`, `>=`, `==`, `!=`, `<=`, `<`, and the
right-hand side may be a field or a number. Only a single comparison is
supported; chain two `.where()` calls for more.

### Named diagnostics

Some diagnostics are requested by name rather than built from an expression:

```text
LiqWaterPath   IceWaterPath   RainWaterPath   RimeWaterPath   VapWaterPath
LiqNumberPath  IceNumberPath  RainNumberPath
MeridionalVapFlux   ZonalVapFlux
PotentialTemperature   LiqPotentialTemperature
precip_liq_surf_mass_flux   precip_ice_surf_mass_flux   precip_total_surf_mass_flux
z_mid   z_int   height_mid   height_int   geopotential_mid   geopotential_int   dz
RelativeHumidity   SeaLevelPressure   Exner   wind_speed   AerosolOpticalDepth550nm
```

These compose like anything else: `LiqWaterPath.mean(dim='col')` is fine.

## Composing

Operations chain, and the result of one is the input to the next:

```yaml
      - T_prof := T_mid.where(qc > 1e-5).weighted('dp').mean(dim='lev')
      - dTdt   := T_mid.tend().mean(dim='col')
```

Intermediate quantities are computed once and shared, so writing both
`T_mid.mean(dim='lev')` and `T_mid.mean(dim='lev').isel(lev=0)` does not
compute the average twice.

## When something is wrong

Errors point at the character that caused them:

```text
Parser errors:
  - line 1, column 9: Illegal token in input: '@'
      T_mid + @foo
              ^
```

and unrecognized operations say what is available:

```text
In '(T_mid.man((dim='lev')))': unrecognized method '.man()'.
 - available: .mean, .sum, .isel, .interp, .where, .shift,
              .differentiate, .histogram, .zonal_mean,
              .weighted, .tend
```

## Not available yet

These parse but have no diagnostic behind them yet, and say so when you try:

- `.min()`, `.max()`, `.std()`, `.var()` over `dim='lev'`
- `abs()`, `log()`, `exp()`, `sqrt()`, and negation (`-X`)
- `**`
- compound conditions inside `.where()`

## Older name syntax

Names like `T_mid_at_500hPa` and `T_mid_vert_avg_dp_weighted` still work and
still produce identically-named output. They are deprecated in favour of the
expression syntax — see [Legacy names](parsing_precedence.md) for the full
mapping and for the precedence quirks they carry.
