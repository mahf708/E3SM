# Field Perturbations

EAMxx supports adding random perturbations to fields during initialization and restart.
This capability is useful for creating multiple trajectories in ensemble simulations
or navigating tricky instabilities during model runs.

The perturbation functionality is controlled by parameters under the `initial_conditions` section:

- `perturbed_fields` (`initial_conditions`, `array(string)`):
    - List of field names (with level dimension) to apply random perturbation to.
    - Only fields on the GLL grid with LEV or ILEV as the last dimension are supported.
    - By default, this is empty except for certain compsets (e.g., DP-EAMxx uses T_mid).
    - Example: `perturbed_fields = T_mid, qv`

- `perturbation_limit` (`initial_conditions`, `real`):
    - Defines the range [1-x, 1+x] from which perturbation values are randomly generated.
    - Default: 0.001 (0.1% perturbation)
    - Example: with limit=0.001, each grid point is multiplied by a random value between 0.999 and 1.001.

- `perturbation_minimum_pressure` (`initial_conditions`, `real`):
    - Minimum pressure (in millibars) above which perturbation is applied, relative to a reference level pressure profile.
    - Default: 900.0 mb
    - Perturbations are applied only at levels where the reference pressure exceeds this threshold (i.e., lower atmosphere/near surface).
    - For example, with default 900 mb, perturbations affect the lower troposphere but not the upper atmosphere.

- `perturbation_random_seed` (`initial_conditions`, `integer`):
    - Random seed used for perturbation generation.
    - Default: 0
    - Can be overridden by `generate_perturbation_random_seed`.

- `generate_perturbation_random_seed` (`initial_conditions`, `logical`):
    - When true, generates a random seed based on the current time.
    - Default: false
    - Mutually exclusive with explicitly setting `perturbation_random_seed`.

- `perturb_on_restart` (`initial_conditions`, `logical`):
    - When true, applies perturbations when restarting from a restart file.
    - Default: false
    - **Important**: When enabled, perturbations will be applied at every restart, creating compounded perturbations over multiple restart cycles. Use with caution and ensure this is the desired behavior for your use case.
    - This option is particularly helpful for navigating tricky instabilities or creating diverging trajectories in ensemble runs.

## Usage Examples

To perturb temperature field in initial conditions:
```shell
./atmchange initial_conditions::perturbed_fields=T_mid
./atmchange initial_conditions::perturbation_limit=0.001
```

To enable perturbations on restart (e.g., for handling instabilities):
```shell
./atmchange initial_conditions::perturb_on_restart=true
./atmchange initial_conditions::perturbed_fields=T_mid
```

To use a specific random seed for reproducibility:
```shell
./atmchange initial_conditions::perturbation_random_seed=12345
```

To generate a random seed automatically:
```shell
./atmchange initial_conditions::generate_perturbation_random_seed=true
```
