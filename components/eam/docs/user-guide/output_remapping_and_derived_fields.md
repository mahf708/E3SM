# Online Output Remapping, Vertical Coarsening, and Derived Fields

> **Development status:** This functionality was developed as an out-of-cycle
> extension for EAM, specifically for the AI Group (aigroup) efforts. It is not
> part of the official E3SM release and has not gone through the standard E3SM
> review and testing process. Use at your own discretion and report issues to
> the aigroup team.

EAM supports three composable online output processing features that avoid the
need for offline post-processing:

1. **Horizontal remapping** — remap output from the native physics grid
   (e.g., ne30pg2, ne120pg2) to a regular lat-lon grid via a precomputed
   ESMF/TempestRemap mapping file.
2. **Vertical coarsening** — reduce the vertical dimension by averaging
   model levels into coarser pressure-bounded layers using pdel-weighted
   averaging.
3. **Derived fields** — define new output fields as sums of existing 3D
   state variables or constituents (e.g., `TOTAL_CLD_WATER = CLDLIQ + CLDICE`).

These features are fully composable: derived fields can be vertically
coarsened, and both native and processed fields can be output on remapped
grids. All processing happens in double precision before writing; the output
precision is controlled by the standard `ndens` namelist parameter.

---

## 1. Horizontal Remapping via Map Files

### Overview

Remap history output from the native unstructured physics grid to a regular
lat-lon grid using a precomputed sparse matrix mapping file. This is
analogous to the `horiz_remap_file` capability in EAMxx (SCREAM).

The remapping applies to an entire history tape — all fields on a tape with
`horiz_remap_file` set will be output on the target lat-lon grid with
dimensions `(lon, lat, ...)` instead of `(ncol, ...)`.

### Namelist Parameters (cam_history_nl)

| Parameter | Type | Description |
|-----------|------|-------------|
| `horiz_remap_file(k)` | `char*256` | Path to ESMF map file for history tape k. Empty string (default) = no remapping. |

### Constraints

- **Mutually exclusive** with `interpolate_output` on the same tape. Setting
  both will cause a runtime error.
- Not supported for satellite history files or initial-condition files (will
  be silently disabled with a warning).
- **Scalar fields only** — no vector wind rotation is performed. U and V wind
  components are remapped as independent scalars, which is appropriate for
  analysis but not for computing derived vector quantities on the target grid.
- Both **coarsening** (target grid smaller than source) and **refining**
  (target grid larger than source) are supported and have been tested.
- Works with any field on the physics grid, including 2D (horiz_only) fields
  like PS, and 3D fields like T(lev, ncol). Also works with the vertically
  coarsened scalar fields (T_1, T_2, etc.).

### Basic Example

```fortran
! In user_nl_eam:
fincl2 = 'T', 'Q', 'U', 'V', 'PS'
horiz_remap_file(2) = '/path/to/map_ne30pg2_to_gaussian_90x180.nc'
```

This outputs T, Q, U, V, PS on tape h1 (fincl2 = h1) on a 90x180 Gaussian
grid instead of the native ne30pg2 grid.

### Generating Map Files

Map files are generated using TempestRemap and NCO. These tools are available
in the `xgns` micromamba environment on NERSC Perlmutter, or can be installed
via conda/pip.

#### Step 1: Generate the source grid SCRIP file

If you don't already have a SCRIP file for your physics grid:

```bash
NE=30  # or 4, 120, etc. — must match your run's ATM_GRID

# Generate cubed-sphere mesh
GenerateCSMesh --alt --res ${NE} --file ne${NE}.g

# Generate pg2 volumetric mesh
GenerateVolumetricMesh --in ne${NE}.g --out ne${NE}pg2.g --np 2 --uniform

# Convert to SCRIP format
ConvertMeshToSCRIP --in ne${NE}pg2.g --out ne${NE}pg2_scrip.nc

# Fix grid_imask type (required for ncremap compatibility)
ncap2 --overwrite -s 'grid_imask=int(grid_imask)' ne${NE}pg2_scrip.nc ne${NE}pg2_scrip.nc
```

For ne4pg2 (384 columns), this produces a SCRIP file of about 50 KB.
For ne30pg2 (21,600 columns), about 3 MB.

#### Step 2: Generate the target Gaussian grid

```bash
# Gaussian grid with 90 latitudes x 180 longitudes (16,200 points)
ncremap -g gaussian_grid_90x180.nc -G latlon=90,180#lat_typ=gss

# Smaller grid for coarsening tests (10 x 20 = 200 points)
ncremap -g gaussian_grid_10x20.nc -G latlon=10,20#lat_typ=gss
```

If you need the Gaussian grid shifted by 0.5 degrees in longitude (e.g., to
match an FV3-generated grid), apply the shift in Python:

```python
import xarray as xr
ds = xr.open_dataset("gaussian_grid.nc")
ds["grid_center_lon"] = (ds["grid_center_lon"] + 0.5) % 360
ds["grid_corner_lon"] = (ds["grid_corner_lon"] + 0.5) % 360
ds.to_netcdf("gaussian_grid_shifted.nc")
```

#### Step 3: Generate the mapping file

```bash
# Area-averaged conservative remapping (recommended)
ncremap -a aave \
    -s ne${NE}pg2_scrip.nc \
    -g gaussian_grid_90x180.nc \
    -m map_ne${NE}pg2_to_gaussian_90x180.nc
```

#### Validating the map file

After generating, verify the map file has the required structure:

```bash
ncdump -h map_ne30pg2_to_gaussian_90x180.nc | grep -E "n_s|n_a|n_b|dst_grid"
```

Expected output:
```
    n_s = 156336 ;       # number of sparse matrix entries
    n_a = 21600 ;        # source grid size (must match ATM_NX)
    n_b = 16200 ;        # target grid size (nlat * nlon)
    dst_grid_rank = 2 ;  # must be 2 for lat-lon grids
    int dst_grid_dims(dst_grid_rank) ;
```

Check that `dst_grid_dims` contains `[nlon, nlat]`:
```bash
ncdump -v dst_grid_dims map_ne30pg2_to_gaussian_90x180.nc
# Should show: dst_grid_dims = 180, 90 ;
```

### Map File Format Requirements

The mapping file must be in standard ESMF/TempestRemap format:

| NetCDF Entity | Type | Description |
|---------------|------|-------------|
| `n_s` | dimension | Number of sparse matrix entries |
| `n_a` | dimension | Source grid size (must match physics grid `ATM_NX`) |
| `n_b` | dimension | Target grid size (`nlat * nlon`) |
| `dst_grid_rank` | dimension | Must be exactly 2 for lat-lon target grids |
| `dst_grid_dims(dst_grid_rank)` | int variable | `[nlon, nlat]` (Fortran order) |
| `row(n_s)` | int variable | Target grid indices (1-indexed) |
| `col(n_s)` | int variable | Source grid indices (1-indexed) |
| `S(n_s)` | double variable | Interpolation weights |
| `xc_b(n_b)` | double variable | Target grid longitudes (degrees) |
| `yc_b(n_b)` | double variable | Target grid latitudes (degrees) |

### Output File Structure

When `horiz_remap_file` is active for a tape, the output file will have:
- `lat(lat)` and `lon(lon)` coordinate dimensions instead of `ncol`
- 3D fields dimensioned as `(time, lev, lat, lon)` instead of `(time, lev, ncol)`
- 2D fields dimensioned as `(time, lat, lon)` instead of `(time, ncol)`
- Standard CF-compliant coordinate attributes

### Implementation Details

The remapping module (`horiz_remap_mod.F90`) works as follows:

1. **Initialization** (`horiz_remap_init`): Reads the mapping file, partitions
   target grid rows contiguously across MPI ranks, extracts each rank's local
   sparse matrix entries, and sets up an `MPI_Alltoallv` communication pattern
   to exchange source column data between ranks.

2. **Field remapping** (`horiz_remap_field`): For each output field, packs
   local source columns into a send buffer, performs `MPI_Alltoallv` to
   gather needed source data from other ranks, then applies the local sparse
   matrix-vector multiply.

3. **Output** (`horiz_remap_write`): Writes the remapped field using PIO
   with a DOF-based decomposition matching the lat-lon grid structure.

The output grid is registered with `cam_grid_support` using grid IDs in the
300+ range (to avoid conflicts with the existing `interpolate_output` grids
in the 200+ range).

---

## 2. Vertical Coarsening

### Overview

Reduce the vertical dimension by computing pdel-weighted (pressure-thickness-
weighted) averages of model levels within user-defined pressure layers.

Each coarsened layer produces a 2D (horizontal-only) output field. For a
field named `T` with 4 coarsened layers, the output fields are `T_1`, `T_2`,
`T_3`, `T_4`.

The coarsening handles partial overlap correctly: if a model level spans
part of a target pressure layer, only the overlapping portion contributes
to the weighted average. This is important near the surface where model
levels may not fully span the bottom target layer.

### Namelist Parameters (derived_fields_nl)

| Parameter | Type | Description |
|-----------|------|-------------|
| `vcoarsen_pbounds` | `real(51)` | Pressure boundaries (Pa) from model top to surface. N boundaries define N-1 layers. Unused entries should be -1 (default). |
| `vcoarsen_flds` | `char*34(100)` | List of field names to vertically coarsen. |

### Supported Input Fields

The vertical coarsening and derived field system operates on 3D fields
accessible from `physics_state` at runtime. This is a strict requirement —
only fields that exist in the `physics_state` type or as registered
constituents can be used. Arbitrary history fields (e.g., `CLOUD`, `RELHUM`,
`FREQL`) and physics buffer (`pbuf`) fields are **not** supported.

**Hardcoded state variables** (accessed directly from `state%t`, `state%u`, etc.):
- `T` — air temperature
- `U` — eastward wind
- `V` — northward wind
- `Q` — specific humidity (water vapor, constituent index 1)
- `OMEGA` — vertical pressure velocity
- `Z3` — geopotential height at midpoints

**Registered constituents** (accessed via `cnst_get_ind` from `state%q(:,:,idx)`):
- `CLDLIQ` — cloud liquid water mixing ratio
- `CLDICE` — cloud ice mixing ratio
- `NUMLIQ`, `NUMICE` — cloud droplet/ice number concentrations
- `RAINQM`, `SNOWQM` — rain/snow mixing ratios
- Any other species registered with the constituent system

**Derived field names** (from `derived_fld_defs`, see Section 3):
- e.g., `TOTAL_CLD_WATER`, `SPECIFIC_TOTAL_WATER`

**Not supported:**
- 2D surface fields (`PS`, `TS`, `FLUT`, `LHFLX`, etc.) — these have no
  vertical dimension and don't need coarsening
- Diagnostic fields computed in parameterizations (`CLOUD`, `RELHUM`, etc.)
- Physics buffer fields

If a field name is misspelled or unsupported, the model will error at
initialization with a clear message identifying the invalid name.

### Output Field Naming

Output fields are named `FIELDNAME_K` where K is the layer index (1 = topmost
layer, N = bottommost layer). These are registered as 2D `horiz_only` fields
and must be explicitly requested on a history tape via `fincl`.

### Example

```fortran
! Define 4 layers: 0-250 hPa, 250-500 hPa, 500-750 hPa, 750-1013.25 hPa
vcoarsen_pbounds = 0, 25000, 50000, 75000, 101325
vcoarsen_flds = 'T', 'Q'

! Request coarsened fields on tape 1
fincl1 = 'T_1', 'T_2', 'T_3', 'T_4',
         'Q_1', 'Q_2', 'Q_3', 'Q_4'
```

### Notes on Verification

When verifying vertical coarsening offline, note that:

- **Instantaneous output** can be verified exactly by reconstructing interface
  pressures from `PS`, `hyai`, and `hybi` (which are in the output file) and
  computing the pdel-weighted average manually.
- **Time-averaged output** cannot be verified exactly from the averaged fields
  because `avg(coarsen(T, pint))` != `coarsen(avg(T), avg(pint))` — the
  pressure weights vary in time, making the coarsening and averaging operations
  non-commutative.
- **Single-precision output** (default `ndens=2`) introduces ~1e-7 relative
  errors compared to double-precision reference values. Use `ndens=1` for
  bit-for-bit verification.
- **Near-surface layers** may have columns where the surface pressure is below
  the bottom pressure boundary, producing zero-weight entries. The module
  handles this correctly but verification scripts should mask these points.

---

## 3. Derived (Combined) Fields

### Overview

Define new output fields that are the sum of existing 3D state variables or
constituents. This avoids the need to add custom `outfld` calls in the physics
code for common combined quantities.

### Namelist Parameters (derived_fields_nl)

| Parameter | Type | Description |
|-----------|------|-------------|
| `derived_fld_defs` | `char*256(50)` | Definitions in the form `OUTPUT_NAME=INPUT1+INPUT2+...` |

### Example

```fortran
derived_fld_defs = 'TOTAL_CLD_WATER=CLDLIQ+CLDICE'

! Request on any tape
fincl1 = 'TOTAL_CLD_WATER'
```

The derived field `TOTAL_CLD_WATER` is registered as a 3D field on `lev` and
can be requested on any tape, including remapped tapes.

### Composing with Vertical Coarsening

Derived fields can be vertically coarsened by including the derived field name
in `vcoarsen_flds`. The module first computes the sum, then applies the
pdel-weighted vertical coarsening:

```fortran
derived_fld_defs = 'TOTAL_CLD_WATER=CLDLIQ+CLDICE'
vcoarsen_pbounds = 0, 25000, 50000, 75000, 101325
vcoarsen_flds = 'T', 'Q', 'TOTAL_CLD_WATER'

! Request the full 3D field AND the per-layer coarsened fields
fincl1 = 'TOTAL_CLD_WATER',
         'TOTAL_CLD_WATER_1', 'TOTAL_CLD_WATER_2',
         'TOTAL_CLD_WATER_3', 'TOTAL_CLD_WATER_4'
```

### Notes on Precision

When verifying `TOTAL_CLD_WATER == CLDLIQ + CLDICE` from output files, the
comparison will show ~1e-7 relative errors with default `ndens=2` (single
precision) output. This is because the model computes the sum in double
precision before writing, but when you read back the individually-written
single-precision `CLDLIQ` and `CLDICE` and sum them, the truncation differs.
The derived field computation itself is exact in double precision.

---

## 4. Composing All Features

All three features are fully composable:

- Derived fields can be vertically coarsened
- Both native and processed fields can be output on remapped grids
- Different tapes can use different remapping targets
- Each tape independently controls frequency, averaging, and field selection

### Production Test Configuration

The following `user_nl_eam` configuration was used for validation testing on
Perlmutter with the `SMS_P128.ne4pg2_oQU480.F2010` test case:

```fortran
! Clear default h0 fields to reduce output noise
empty_htapes = .true.

! Output frequency and averaging
! h0 = hourly avg (native, verification baseline)
! h1 = hourly instant, remapped to 90x180 Gaussian (refining)
! h2 = 6-hourly avg, remapped to 10x20 Gaussian (coarsening)
! h3 = hourly instant (native, reference for h1 offline comparison)
! h4 = 6-hourly avg (native, reference for h2 offline comparison)
nhtfrq = -1, -1, -6, -1, -6
avgflag_pertape = 'A', 'I', 'A', 'I', 'A'

! Derived fields
derived_fld_defs = 'TOTAL_CLD_WATER=CLDLIQ+CLDICE'

! Vertical coarsening: 4 pressure layers
vcoarsen_pbounds = 0, 25000, 50000, 75000, 101325
vcoarsen_flds = 'T', 'Q', 'TOTAL_CLD_WATER'

! Tape 1 (h0): native grid with all verification fields
fincl1 = 'T', 'Q', 'PS', 'CLDLIQ', 'CLDICE', 'PDELDRY',
         'T_1', 'T_2', 'T_3', 'T_4',
         'Q_1', 'Q_2', 'Q_3', 'Q_4',
         'TOTAL_CLD_WATER',
         'TOTAL_CLD_WATER_1', 'TOTAL_CLD_WATER_2',
         'TOTAL_CLD_WATER_3', 'TOTAL_CLD_WATER_4'

! Tape 2 (h1): remapped to 90x180 Gaussian (refining, 16200 > 384 src pts)
fincl2 = 'T', 'Q', 'U', 'V', 'PS',
         'T_1', 'T_2', 'T_3', 'T_4',
         'Q_1', 'Q_2', 'Q_3', 'Q_4',
         'TOTAL_CLD_WATER',
         'TOTAL_CLD_WATER_1', 'TOTAL_CLD_WATER_2',
         'TOTAL_CLD_WATER_3', 'TOTAL_CLD_WATER_4'
horiz_remap_file(2) = '/path/to/map_ne4pg2_to_gaussian_90x180.nc'

! Tape 3 (h2): remapped to 10x20 Gaussian (coarsening, 200 < 384 src pts)
fincl3 = 'T', 'Q', 'U', 'V', 'PS',
         'T_1', 'T_2', 'T_3', 'T_4',
         'TOTAL_CLD_WATER',
         'TOTAL_CLD_WATER_1', 'TOTAL_CLD_WATER_2',
         'TOTAL_CLD_WATER_3', 'TOTAL_CLD_WATER_4'
horiz_remap_file(3) = '/path/to/map_ne4pg2_to_gaussian_10x20.nc'

! Tape 4 (h3): native reference for tape 2 (same fields/freq/avg)
fincl4 = 'T', 'Q', 'U', 'V', 'PS',
         'T_1', 'T_2', 'T_3', 'T_4',
         'Q_1', 'Q_2', 'Q_3', 'Q_4',
         'TOTAL_CLD_WATER',
         'TOTAL_CLD_WATER_1', 'TOTAL_CLD_WATER_2',
         'TOTAL_CLD_WATER_3', 'TOTAL_CLD_WATER_4'

! Tape 5 (h4): native reference for tape 3 (same fields/freq/avg)
fincl5 = 'T', 'Q', 'U', 'V', 'PS',
         'T_1', 'T_2', 'T_3', 'T_4',
         'TOTAL_CLD_WATER',
         'TOTAL_CLD_WATER_1', 'TOTAL_CLD_WATER_2',
         'TOTAL_CLD_WATER_3', 'TOTAL_CLD_WATER_4'
```

### Verification Strategy

The test configuration above supports four verification checks:

| Test | Online | Reference | Method |
|------|--------|-----------|--------|
| Horiz remap (refining) | h1 (90x180) | h3 (native) | `ncremap -m map.nc h3.nc offline.nc`, diff against h1 |
| Horiz remap (coarsening) | h2 (10x20) | h4 (native) | `ncremap -m map.nc h4.nc offline.nc`, diff against h2 |
| Vertical coarsening | h0: `T_1`..`T_4` | h0: `T`, `PS` | Manual pdel-weighted avg using `PS` + `hyai`/`hybi` |
| Derived field | h0: `TOTAL_CLD_WATER` | h0: `CLDLIQ` + `CLDICE` | Direct sum comparison |

**Expected precision:**
- With `ndens=2` (default, single precision): ~1e-7 relative errors
- With `ndens=1` (double precision): bit-for-bit for instantaneous fields;
  ~1e-14 for averaged fields due to floating-point accumulation order

The reference tapes (h3/h4) must use the same averaging type and frequency
as the remapped tapes (h1/h2) for a fair comparison. Instantaneous tapes
(`'I'`) give the cleanest comparison since there is no averaging.

For the remapping comparison, note that the online remap operates on the
full double-precision history buffer (`hbuf`), while the offline `ncremap`
reads from the single-precision native file. This explains the ~1e-7
differences with `ndens=2`.

### Plotting Native Unstructured Data

When plotting native ne4pg2/ne30pg2 output for visual comparison, use
Matplotlib's `tripcolor` with Cartopy for proper triangulated interpolation:

```python
import cartopy.crs as ccrs

def fix_lon(x):
    return (x + 180) % 360 - 180

fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.tripcolor(fix_lon(ds.lon.values), ds.lat.values, field_data,
             transform=ccrs.PlateCarree(), cmap="RdYlBu_r",
             shading="gouraud")
ax.set_global()
ax.coastlines(resolution="110m", linewidth=0.5)
```

Do **not** use `scatter` for native unstructured data — it produces unfilled
point plots that are hard to interpret visually.

---

## 5. Common Pitfalls

### `PINT` is not an output field

EAM does not register `PINT` (interface pressure) via `addfld`. If you need
interface pressures for verification, reconstruct them from `PS` and the
hybrid coefficients `hyai`/`hybi` which are coordinate variables in every
output file:

```python
pint[k] = hyai[k] * P0 + hybi[k] * PS
```

### `interpolate_output` conflict

`horiz_remap_file` and `interpolate_output` cannot be used on the same tape.
`interpolate_output` uses HOMME's native Lagrange polynomial interpolation
and only works on the GLL grid (not the physics grid pg2). The new
`horiz_remap_file` works on the physics grid via sparse matrix application,
making it the correct choice for pg2 grids.

### Field name collisions with `_N` suffix

The vertical coarsening appends `_1`, `_2`, etc. to field names. Avoid
defining fields whose names already end with `_N` patterns, as this could
create ambiguity.

### `empty_htapes`

By default, EAM outputs hundreds of fields on tape h0. Use
`empty_htapes = .true.` to start with a clean slate and only output what
you explicitly request via `fincl`. This is strongly recommended for
testing and production runs where you know exactly what you need.

### Restart support

The `horiz_remap_file` paths are saved in the restart file, so restarted
runs will continue using the same mapping files. If you change the map file
path between runs, the new path will be picked up from the namelist.

---

## 6. Relationship to Offline Data Processing Pipeline

These online features were motivated by the offline data processing pipeline
used by the AI Group for preparing ML training datasets (see
`scripts/data_process/compute_dataset_e3smv2.py` in the ACE repository).
The offline pipeline performs the following steps, and this section documents
which are now handled online and which still require post-processing.

### Operations now handled online

| Offline operation | Online equivalent | Notes |
|-------------------|-------------------|-------|
| `ncremap` (horizontal regridding) | `horiz_remap_file` | Sparse matrix remap, identical approach |
| `compute_vertical_coarsening` (pdel-weighted averaging) | `vcoarsen_pbounds` + `vcoarsen_flds` | Same algorithm: pdel-weighted mean within pressure layers |
| `compute_specific_total_water` (sum of water species) | `derived_fld_defs` | e.g., `SPECIFIC_TOTAL_WATER=Q+CLDLIQ+CLDICE+RAINQM+SNOWQM` |
| Vertical coarsening index specification | `vcoarsen_pbounds` | Offline uses model-level index pairs `[start, end]`; online uses pressure bounds in Pa. Pressure bounds are more portable across resolutions. |

### Operations still requiring offline post-processing

| Offline operation | Why not online | Workaround |
|-------------------|---------------|------------|
| `compute_tendencies` (time derivatives) | Requires data from adjacent timesteps — cannot compute within a single physics step | Compute offline from output time series: `tendency = diff(field) / diff(time)` |
| `compute_rad_fluxes` (e.g., `FLNS+FLDS`) | These are sums/differences of **2D surface fields**, not 3D state variables. The derived field system only accesses `physics_state` 3D fields. | Output the component fluxes (`FLNS`, `FLDS`, `FSDS`, `FSNS`, `FSNTOA`, `SOLIN`) directly and compute sums offline. These are cheap 2D operations. |
| `compute_surface_precipitation_rate` | Simple unit conversion of existing output fields (`PRECT * 1000 kg/m3`) | Trivial offline: `precip_mass_flux = PRECT * 1000` |
| `compute_column_moisture_integral` | Column integral `sum(q * dp) / g` — could be added to derived system but currently not implemented | Output `Q` and `PS`; compute offline using hybrid coefficients |
| `compute_coarse_ak_bk` | Coordinate metadata, not a field transformation | Extract from output file's `hyai`/`hybi` variables at the model-level indices corresponding to the pressure bounds |
| `sfc_phis_to_hgt` | Simple division `PHIS / 9.80665` | Output `PHIS` directly; convert offline |
| `roundtrip_filter` (spherical harmonic filtering) | Spectral transform library not available in EAM Fortran runtime | Apply offline using `xtorch_harmonics` |
| Field renaming | Cosmetic, no computational benefit to doing online | Apply offline with `ncrename` or in xarray |

### Example: Reproducing the full ACE/aigroup pipeline

Given the v3 8-layer AMIP configuration, the online settings would be:

```fortran
! Derived fields: specific total water (all water species)
derived_fld_defs = 'SPECIFIC_TOTAL_WATER=Q+CLDLIQ+CLDICE+RAINQM'

! 8-layer vertical coarsening matching the ACE index-based layers
! These pressure bounds approximate the v3 [0,25,38,46,52,56,61,69,80] level indices
vcoarsen_pbounds = 0, 300, 3000, 10000, 22000, 35000, 51000, 74000, 110000
vcoarsen_flds = 'T', 'U', 'V', 'SPECIFIC_TOTAL_WATER'

! Tape 1: 6-hourly instantaneous on Gaussian grid (replaces offline ncremap)
nhtfrq = -6
avgflag_pertape = 'I'
empty_htapes = .true.

fincl1 = 'PS', 'TS', 'PHIS', 'OCNFRAC', 'LANDFRAC', 'ICEFRAC',
         'LHFLX', 'SHFLX', 'FLNS', 'FLDS', 'FSNS', 'FSDS',
         'FSNTOA', 'SOLIN', 'FLUT', 'PRECT', 'TMQ', 'TGCLDLWP', 'TGCLDIWP',
         'T_1','T_2','T_3','T_4','T_5','T_6','T_7','T_8',
         'U_1','U_2','U_3','U_4','U_5','U_6','U_7','U_8',
         'V_1','V_2','V_3','V_4','V_5','V_6','V_7','V_8',
         'SPECIFIC_TOTAL_WATER_1','SPECIFIC_TOTAL_WATER_2',
         'SPECIFIC_TOTAL_WATER_3','SPECIFIC_TOTAL_WATER_4',
         'SPECIFIC_TOTAL_WATER_5','SPECIFIC_TOTAL_WATER_6',
         'SPECIFIC_TOTAL_WATER_7','SPECIFIC_TOTAL_WATER_8'
horiz_remap_file(1) = '/path/to/map_ne30pg2_to_gaussian_180x360.nc'
```

The remaining offline steps (radiation flux sums, precipitation unit
conversion, time tendencies, spherical harmonic filtering, renaming) are
lightweight 2D operations that take seconds compared to the minutes/hours
saved by eliminating the horizontal remapping and vertical coarsening from
post-processing.

### Converting offline vertical coarsening indices to pressure bounds

The offline pipeline specifies coarsening layers as model-level index pairs
(e.g., `[0, 25]` means levels 0 through 24). To convert to pressure bounds
for the online `vcoarsen_pbounds`, look up the interface pressure at each
boundary index using the model's hybrid coefficients:

```python
import xarray as xr
# Read from any EAM output file
ds = xr.open_dataset("eam_output.nc")
hyai = ds["hyai"].values  # hybrid A coefficient at interfaces
P0 = ds["P0"].values      # reference pressure (typically 100000 Pa)

# Offline indices (from YAML config)
indices = [0, 25, 38, 46, 52, 56, 61, 69, 80]

# Convert to approximate pressure bounds (using hyai*P0, ignoring hybi*PS)
pbounds_pa = [hyai[i] * P0 for i in indices]
print("vcoarsen_pbounds =", ", ".join(f"{p:.0f}" for p in pbounds_pa))
```

Note: The online approach uses fixed pressure bounds, while the offline
approach uses model-level indices. The pressure-based approach is more
portable across resolutions but may not exactly replicate the level-based
coarsening. For exact equivalence, choose pressure bounds that align with
the interface pressures at the desired level indices.

---

## 7. Source Files

| File | Description |
|------|-------------|
| `components/eam/src/control/horiz_remap_mod.F90` | Horizontal remapping module: reads map files, manages MPI communication, applies sparse matrix multiply |
| `components/eam/src/control/cam_history_derived.F90` | Vertical coarsening and derived field computation |
| `components/eam/src/control/cam_history.F90` | History output integration: namelist, h_define, dump_field, restart |
| `components/eam/src/control/runtime_opts.F90` | Namelist reading integration for derived_fields_nl |
| `components/eam/bld/namelist_files/namelist_definition.xml` | Namelist parameter definitions |
| `components/eam/docs/user-guide/output_remapping_and_derived_fields.md` | This documentation |

### Related EAMxx implementation

The EAMxx (SCREAM) atmosphere model has a similar horizontal remapping
capability configured via `horiz_remap_file` in YAML output specifications.
The EAM implementation follows the same design principles (sparse matrix from
ESMF map files, CRS storage, MPI-parallel application) but is implemented in
Fortran and integrates with EAM's `cam_history` infrastructure rather than
EAMxx's `AtmosphereOutput` class.
