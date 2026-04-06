# Nudging in EAMxx

Nudging is supported in EAMxx.
Currently, it is possible to nudge EAMxx to the output from a different EAMxx run or to reanalysis. Nudging data can be on your model grid or an arbitrary coarser grid. Inline interpolating of finer-grid nudging data to a coarser model resolution isn't implemented yet but may be in the future.

## Data to nudge towards

The user is expected to prepapre (and then use `atmchange` to point to) nudging data files that are compliant with EAMxx specification.
In practice, this means that the data files must contain variable names known to EAMxx (only U, V, T_mid, and qv are supported now).
The files can be specified with an explicit list or via pattern matching.
The files must contain an arbitary global attribute `case_t0`, and it is recommended to be the same as the time dimension unit (the files must be time-dimensioned).
The time dimension should be UNLIMITED.
Finally, the dimension order must be such that `lev` is the last dimension, so most likely, the user must transpose the dimensions.

## Pressure in the nudging data

Pressure can be explicitly provided in the nudging data as time-varying `p_mid` corresponding to the option `TIME_DEPENDENT_3D_PROFILE` for `source_pressure_type`.
Alternatively, the data can contain a time-invariant pressure variable `p_lev` corresponding to the option `STATIC_1D_VERTICAL_PROFILE` for `source_pressure_type`.

## Weighted nudging for RRM applications

In regionally refined model applications, it is possible to use weighted nudging, for example, to avoid nudging the refined region.
To achieve that, the user can use `atmchange` to set `use_nudging_weights` (boolean) and provide `nudging_weights_file` that has the weight to apply for nudging (for example, zeros in the refined region).
Currently, weighted nudging is only supported if the user provides the nudging data at the target grid.

## Example setup (current as of April 2024)

To enable nudging as a process, one must declare it in the `atm_procs_list` runtime parameter.

```shell
./atmchange physics::atm_procs_list="mac_aero_mic,rrtmgp,cosp,nudging"
```

The following options are needed to specify the nudging.

```shell
./atmchange nudging::nudging_filenames_patterns="/pathto/nudging_files/*.nc" # can provide file name explicitly here instead (or multiple patterns)
./atmchange nudging::source_pressure_type=TIME_DEPENDENT_3D_PROFILE # see section on pressure
./atmchange nudging::nudging_fields=U,V # can include qv, T_mid as well
./atmchange nudging::nudging_timescale=21600 # in seconds
```

To gain a deeper understanding of these parameters and options, please refer to code implementation of the nudging process.

## Enhanced Nudging Options

The following optional features are based on Zhang et al. (2026, GMD) and can be enabled at runtime via `atmchange`. All default to OFF, preserving backward compatibility.

### Per-Variable Timescales and Coefficients

Each nudged field can have its own relaxation timescale and strength coefficient:

```shell
# Per-variable timescales (seconds) — override the global nudging_timescale
./atmchange nudging::nudging_timescales::T_mid=43200    # 12h for temperature
./atmchange nudging::nudging_timescales::qv=86400       # 24h for humidity

# Per-variable coefficients [0,1] — scale nudging strength per field
./atmchange nudging::nudging_coefficients::T_mid=0.5
./atmchange nudging::nudging_coefficients::qv=0.3
```

### Unified Weight Function w_m (Eq. 4)

A physics-based weight function that smoothly transitions nudging strength across vertical regimes. It combines upper-atmosphere ramp-off and PBL exclusion into a single piecewise function:

- `P_m <= P_top`: Zero nudging (above reanalysis ceiling)
- `P_top < P_m <= P_0`: Linear ramp `P_m / P_0` (upper atmosphere)
- `P_m > P_0` and `Z_m <= Z_b`: Tanh PBL transition (near surface)
- Otherwise: Full nudging (free troposphere)

```shell
./atmchange nudging::nudging_weight_function=true
./atmchange nudging::nudging_wfn_p_top=100.0       # Pa (1 hPa), top cutoff
./atmchange nudging::nudging_wfn_p0_default=3000.0  # Pa (30 hPa), default upper ramp
./atmchange nudging::nudging_wfn_z_b=150.0          # m, PBL height threshold

# Per-variable P_0 overrides (different ramp pressures per field)
./atmchange nudging::nudging_wfn_p0::T_mid=1000.0    # 10 hPa for temperature
./atmchange nudging::nudging_wfn_p0::qv=10000.0      # 100 hPa for humidity
```

Note: This requires `z_mid` (geopotential height) to be available from the dynamics.

### Horizontal Window Function

A tanh-based 2D lat/lon Heaviside window for regional nudging (e.g., RRM applications):

```shell
./atmchange nudging::nudging_horiz_window=true
./atmchange nudging::nudging_hwin_lat0=38.0
./atmchange nudging::nudging_hwin_lon0=254.0
./atmchange nudging::nudging_hwin_latwidth=34.0
./atmchange nudging::nudging_hwin_lonwidth=44.0
./atmchange nudging::nudging_hwin_latdelta=3.8
./atmchange nudging::nudging_hwin_londelta=3.8
./atmchange nudging::nudging_hwin_invert=true   # nudge OUTSIDE the window
```

### Vertical Window Function

A tanh-based window to restrict nudging to specific model level ranges:

```shell
./atmchange nudging::nudging_vert_window=true
./atmchange nudging::nudging_vwin_lindex=1.0     # low transition level
./atmchange nudging::nudging_vwin_hindex=60.0    # high transition level
./atmchange nudging::nudging_vwin_ldelta=1.0
./atmchange nudging::nudging_vwin_hdelta=1.0
```

### Advanced Thermodynamic Nudging

Alternative strategies for temperature and humidity nudging that reduce interference with model physics:

```shell
# Temperature: 0=direct T, 1=virtual temperature Tv, 2=RH-based adjustment
./atmchange nudging::nudging_t_option=1

# Humidity: 0=direct q, 1=RH-based adjustment
./atmchange nudging::nudging_q_option=1
```

### Tendency Diagnostic Output

Nudging tendency diagnostics use EAMxx's built-in `compute_tendencies` mechanism. To output how nudging modifies specific fields:

```shell
./atmchange nudging::compute_tendencies=T_mid,qv
```

This will produce tendency fields showing the rate of change due to nudging for each specified field.

### Field Name Mapping

When nudging data files use different variable names than EAMxx, use `nudging_names` to map them:

```shell
./atmchange nudging::nudging_fields=U,V,T_mid,qv
./atmchange nudging::nudging_names::T_mid=temperature
./atmchange nudging::nudging_names::qv=specific_humidity
```

This reads `temperature` from the data file to nudge `T_mid`, and `specific_humidity` to nudge `qv`. Fields without explicit mappings use their EAMxx name.

### Generic Field Nudging

Any 3D scalar field in the EAMxx field manager can be nudged, not just U, V, T_mid, and qv. For example, to nudge ozone:

```shell
./atmchange nudging::nudging_fields=U,V,T_mid,o3
```

### Horizontal Remapping

Nudging data can be on any grid (coarser or finer than the model grid). Provide a map file connecting the data grid to the model grid:

```shell
./atmchange nudging::nudging_refine_remap_mapfile="/path/to/map_file.nc"
```

The remapper automatically handles both refining (coarse data → fine model) and coarsening (fine data → coarse model) directions. The map file must have `n_a` and `n_b` dimensions matching the data and model grids (in either order).

### Option Simplification Notes

- The legacy `nudging_refine_remap_vert_cutoff` parameter is automatically absorbed into the Eq. 4 weight function when set. You don't need to enable both.
- If both `nudging_weight_function` and `nudging_vert_window` are enabled, a warning is logged since they provide overlapping vertical control. Use one or the other unless you specifically need both.
