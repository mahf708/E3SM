# Online Output Remapping and Derived Fields

EAM supports online horizontal remapping and vertical coarsening of history
output fields, avoiding the need for offline post-processing with tools like
`ncremap`. These features are configured via namelist parameters in
`user_nl_eam`.

## Horizontal Remapping via Map Files

Remap history output from the native physics grid (e.g., ne30pg2, ne120pg2)
to a regular lat-lon grid using a precomputed ESMF/TempestRemap mapping file.
This is analogous to the `horiz_remap_file` capability in EAMxx.

### Namelist (cam_history_nl)

| Parameter | Type | Description |
|-----------|------|-------------|
| `horiz_remap_file(k)` | `char*256` | Path to ESMF map file for history tape k. Empty = no remapping. |

- Mutually exclusive with `interpolate_output` on the same tape.
- Not supported for satellite or initial-condition history files.
- The map file must contain `dst_grid_rank` and `dst_grid_dims` variables so
  that `nlat` and `nlon` can be inferred automatically.
- Scalar fields only (no vector wind rotation in this implementation).

### Example

```fortran
! In user_nl_eam:
! Tape 1: native grid output (default)
! Tape 2: remapped to 90x180 Gaussian grid
fincl2 = 'T', 'Q', 'U', 'V', 'PS'
horiz_remap_file(2) = '/path/to/map_ne30pg2_to_gaussian_90x180.nc'
```

### Generating Map Files

Map files can be generated using TempestRemap and NCO tools. The source grid
SCRIP file must match the model's physics grid.

**Step 1: Generate the source grid SCRIP file** (if not already available):

```bash
NE=30  # or 4, 120, etc.

# Generate cubed-sphere mesh
GenerateCSMesh --alt --res ${NE} --file ne${NE}.g

# Generate pg2 volumetric mesh
GenerateVolumetricMesh --in ne${NE}.g --out ne${NE}pg2.g --np 2 --uniform

# Convert to SCRIP format
ConvertMeshToSCRIP --in ne${NE}pg2.g --out ne${NE}pg2_scrip.nc

# Fix grid_imask type for compatibility
ncap2 --overwrite -s 'grid_imask=int(grid_imask)' ne${NE}pg2_scrip.nc ne${NE}pg2_scrip.nc
```

**Step 2: Generate the target Gaussian grid:**

```bash
# Gaussian grid (e.g., 90 lats x 180 lons)
ncremap -g gaussian_grid_90x180.nc -G latlon=90,180#lat_typ=gss
```

**Step 3: Generate the mapping file:**

```bash
ncremap -a aave \
    -s ne${NE}pg2_scrip.nc \
    -g gaussian_grid_90x180.nc \
    -m map_ne${NE}pg2_to_gaussian_90x180.nc
```

### Map File Requirements

The mapping file must be in standard ESMF/TempestRemap format with:

| NetCDF Entity | Description |
|---------------|-------------|
| `n_s` (dim) | Number of sparse matrix entries |
| `n_a` (dim) | Source grid size (must match physics grid ncols) |
| `n_b` (dim) | Target grid size (nlat * nlon) |
| `dst_grid_rank` (dim) | Must be 2 for lat-lon target grids |
| `dst_grid_dims` (var) | Array of 2 integers: `[nlon, nlat]` |
| `row` (var) | Target indices (1-indexed), shape `(n_s)` |
| `col` (var) | Source indices (1-indexed), shape `(n_s)` |
| `S` (var) | Interpolation weights, shape `(n_s)` |
| `xc_b` (var) | Target grid longitudes (degrees), shape `(n_b)` |
| `yc_b` (var) | Target grid latitudes (degrees), shape `(n_b)` |

### Implementation Details

The remapping module (`horiz_remap_mod.F90`) partitions the target grid across
MPI ranks and uses `MPI_Alltoallv` to exchange source column data between ranks
before applying the sparse matrix-vector multiply. Output is written via PIO
with a DOF-based decomposition matching the lat-lon grid structure.

---

## Vertical Coarsening

Reduce the vertical dimension by averaging model levels into coarser pressure
layers using pdel-weighted averaging. Output fields are 2D (horizontal only),
with one field per coarsened layer.

### Namelist (derived_fields_nl)

| Parameter | Type | Description |
|-----------|------|-------------|
| `vcoarsen_pbounds` | `real(51)` | Pressure boundaries (Pa) from model top to surface. N boundaries define N-1 layers. Default: -1 (disabled). |
| `vcoarsen_flds` | `char*34(100)` | Field names to vertically coarsen. |

Output field names are `FIELDNAME_1`, `FIELDNAME_2`, ..., `FIELDNAME_N` for
N coarsened layers. These must be explicitly requested on a history tape via
`fincl`.

### Example

```fortran
! 4 layers: 0-250 hPa, 250-500 hPa, 500-750 hPa, 750-1013.25 hPa
vcoarsen_pbounds = 0, 25000, 50000, 75000, 101325
vcoarsen_flds = 'T', 'Q'

! Request on tape 1
fincl1 = 'T_1', 'T_2', 'T_3', 'T_4', 'Q_1', 'Q_2', 'Q_3', 'Q_4'
```

---

## Derived (Combined) Fields

Define new output fields that are the sum of existing 3D state variables or
constituents. Useful for outputting combined quantities without modifying
physics code.

### Namelist (derived_fields_nl)

| Parameter | Type | Description |
|-----------|------|-------------|
| `derived_fld_defs` | `char*256(50)` | Definitions in the form `OUTPUT_NAME=INPUT1+INPUT2+...` |

### Example

```fortran
derived_fld_defs = 'TOTAL_CLD_WATER=CLDLIQ+CLDICE'

! Request the full 3D field on tape 1
fincl1 = 'TOTAL_CLD_WATER'
```

### Composing Derived Fields with Vertical Coarsening

Derived fields can also be vertically coarsened by including the derived field
name in `vcoarsen_flds`. The module computes the sum first, then applies the
vertical coarsening.

```fortran
derived_fld_defs = 'TOTAL_CLD_WATER=CLDLIQ+CLDICE'
vcoarsen_pbounds = 0, 25000, 50000, 75000, 101325
vcoarsen_flds = 'T', 'Q', 'TOTAL_CLD_WATER'

! Request per-layer coarsened output (and optionally the full 3D field)
fincl1 = 'TOTAL_CLD_WATER',
         'TOTAL_CLD_WATER_1', 'TOTAL_CLD_WATER_2',
         'TOTAL_CLD_WATER_3', 'TOTAL_CLD_WATER_4'
```

---

## Complete Example

Testing all features together with multiple remapping targets:

```fortran
! === Derived fields ===
derived_fld_defs = 'TOTAL_CLD_WATER=CLDLIQ+CLDICE'

! === Vertical coarsening ===
vcoarsen_pbounds = 0, 25000, 50000, 75000, 101325
vcoarsen_flds = 'T', 'Q', 'TOTAL_CLD_WATER'

! === Tape 1: native grid + coarsened/derived fields ===
fincl1 = 'T_1', 'T_2', 'T_3', 'T_4',
         'Q_1', 'Q_2', 'Q_3', 'Q_4',
         'TOTAL_CLD_WATER',
         'TOTAL_CLD_WATER_1', 'TOTAL_CLD_WATER_2',
         'TOTAL_CLD_WATER_3', 'TOTAL_CLD_WATER_4'

! === Tape 2: remapped to fine lat-lon grid (refining) ===
fincl2 = 'T', 'Q', 'U', 'V', 'PS'
horiz_remap_file(2) = '/path/to/map_ne30pg2_to_gaussian_90x180.nc'

! === Tape 3: remapped to coarse lat-lon grid (coarsening) ===
fincl3 = 'T', 'Q', 'U', 'V', 'PS'
horiz_remap_file(3) = '/path/to/map_ne30pg2_to_gaussian_10x20.nc'
```

## Source Files

| File | Description |
|------|-------------|
| `components/eam/src/control/horiz_remap_mod.F90` | Horizontal remapping module |
| `components/eam/src/control/cam_history_derived.F90` | Vertical coarsening and derived fields |
| `components/eam/src/control/cam_history.F90` | History output integration |
| `components/eam/bld/namelist_files/namelist_definition.xml` | Namelist definitions |
