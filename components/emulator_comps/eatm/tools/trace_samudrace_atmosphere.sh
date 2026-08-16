#!/bin/bash
# Prepare the SamudrACE-E3SMv3 atmosphere for EATM: extract the atmosphere
# stepper from the published checkpoint, trace it to TorchScript, and build a
# matching EATM initial-condition file.
#
# Run this on a GPU node -- the traced model has to be built for the device it
# will run on, and EATM runs it on a GPU:
#
#   salloc --nodes 1 --qos interactive --time 01:00:00 \
#          --constraint gpu --account=<your gpu account>
#   components/emulator_comps/eatm/tools/trace_samudrace_atmosphere.sh
#
# Everything is driven by environment variables so it can be pointed at a
# different checkout or staging area without editing the script.

set -euo pipefail

# --- inputs ------------------------------------------------------------------
# ACE repository checkout (needs the `fme` package importable, e.g. via uv)
ACE_REPO="${ACE_REPO:-${PSCRATCH}/ace}"
# The corrector- and ocean-aware tracing script from the ACE repository
TRACE_SCRIPT="${EATM_TRACE_SCRIPT:-${PSCRATCH}/test_ace_repo/trace.py}"
# Hugging Face checkout of allenai/SamudrACE-E3SMv3
HF_DIR="${HF_DIR:-${PSCRATCH}/SamudrACE-E3SMv3}"
# Where the artifacts EATM consumes are written
STAGE="${EATM_STAGE:-${HF_DIR}/eatm}"
# Which of the three published initial conditions to start from (0, 1 or 2)
IC_INDEX="${IC_INDEX:-0}"

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${STAGE}"
cd "${ACE_REPO}"

RUN="uv run python"
if [[ -x "${ACE_REPO}/.venv/bin/python" ]]; then
    RUN="${ACE_REPO}/.venv/bin/python"
fi

# --- 1. get a single-component atmosphere checkpoint --------------------------
ATM_CKPT="${HF_DIR}/SamudrACE-E3SMv3-atmosphere.tar"
# a previous run of this script may already have extracted one
if [[ ! -s "${ATM_CKPT}" ]] || [[ $(stat -c%s "${ATM_CKPT}") -lt 1000000 ]]; then
    if [[ -s "${STAGE}/SamudrACE-E3SMv3-atmosphere.tar" ]] && \
       [[ $(stat -c%s "${STAGE}/SamudrACE-E3SMv3-atmosphere.tar") -gt 1000000 ]]; then
        ATM_CKPT="${STAGE}/SamudrACE-E3SMv3-atmosphere.tar"
    fi
fi
if [[ ! -s "${ATM_CKPT}" ]] || [[ $(stat -c%s "${ATM_CKPT}") -lt 1000000 ]]; then
    echo "==> ${ATM_CKPT} is absent or still a git-lfs pointer"
    if [[ -s "${HF_DIR}/SamudrACE-E3SMv3.tar" ]]; then
        echo "==> extracting the atmosphere from the coupled checkpoint"
        ATM_CKPT="${STAGE}/SamudrACE-E3SMv3-atmosphere.tar"
        ${RUN} "${ACE_REPO}/scripts/coupled/create_decoupled_checkpoint.py" \
            --component atmosphere \
            --input_path "${HF_DIR}/SamudrACE-E3SMv3.tar" \
            --output_path "${ATM_CKPT}"
    else
        echo "ERROR: no checkpoint found.  Fetch one with:" >&2
        echo "  cd ${HF_DIR} && git lfs pull --include='SamudrACE-E3SMv3-atmosphere.tar'" >&2
        exit 1
    fi
fi
echo "==> atmosphere checkpoint: ${ATM_CKPT}"

# --- 2. trace it --------------------------------------------------------------
# check_trace is deliberately left off: the atmosphere is a NoiseConditionedSFNO
# and draws fresh noise every forward pass, so a trace check would always fail.
EATM_TRACE_SCRIPT="${TRACE_SCRIPT}" ${RUN} "${TOOLS}/trace_eatm_model.py" \
    "${ATM_CKPT}" "${STAGE}/samudrace_atm_traced" \
    --emulator SamudrACE-E3SMv3 \
    --device cuda

# --- 3. build the EATM initial condition ---------------------------------------
${RUN} "${TOOLS}/make_eatm_ic.py" \
    --emulator SamudrACE-E3SMv3 \
    --source "${HF_DIR}/initial_conditions/SamudrACE-E3SMv3-ICx3-train_atmosphere_ic.nc" \
    --forcing "${HF_DIR}/forcing_data/atmosphere-forcing-1yr.nc" \
    --time-index "${IC_INDEX}" \
    --output "${STAGE}/samudrace_atm_ic_${IC_INDEX}.nc"

cat <<EOF

================================================================================
Done.  In your case:

  ./xmlchange EATM_EMULATOR=SamudrACE-E3SMv3

  cat >> user_nl_eatm <<'NL'
  eatm_model_file = '${STAGE}/samudrace_atm_traced_cuda.pt'
  eatm_ic_file    = '${STAGE}/samudrace_atm_ic_${IC_INDEX}.nc'
  NL
================================================================================
EOF
