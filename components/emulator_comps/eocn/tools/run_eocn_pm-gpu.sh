#!/bin/bash
# Create, build and run one of the EOCN compsets on Perlmutter GPU.
#
#   CASE=emul   E2000-EATM-EOCN   gauss180x360_gauss180x360
#               both halves of SamudrACE, everything else stubbed.  One task
#               per component; the whole thing is two neural nets and a coupler.
#
#   CASE=f2010  F2010-ELM-EOCN    ne30pg2_gauss180x360
#               prognostic EAM and ELM over the emulated ocean.  Known to abort
#               on the first physics step at the pole -- see VERIFICATION.md.
#
# Usage:
#   ./run_eocn_pm-gpu.sh                          # emulated pair, 11 days
#   CASE=f2010 ./run_eocn_pm-gpu.sh
#   STOP_N=110 CASE_NAME=eocn-long ./run_eocn_pm-gpu.sh
#   SUBMIT=false ./run_eocn_pm-gpu.sh             # build only
#
# With an interactive allocation, run it inside `salloc`; with none, it submits
# to the batch queue.
set -euo pipefail

# --- toolchain shim, required since the 2026-08-17 NERSC CPE roll --------------
# The system default libmpi_gtl_cuda.so.0 is a CUDA 13 build while E3SM's pm-gpu
# configuration pins cudatoolkit/12.9, and the Cray wrappers link it into every
# binary they produce -- including the throwaway one CMake compiles to check the
# CUDA headers, which is why the failure surfaces inside find_package(FTorch)
# looking like a CUDA_HOME problem.  See eatm/REVIEW.md #74.
if [ "${EOCN_SKIP_ENV_SHIM:-0}" != "1" ]; then
  if command -v module >/dev/null 2>&1; then
    module reset >/dev/null 2>&1 || true
    module load cray-python >/dev/null 2>&1 || true
  fi
  export LD_LIBRARY_PATH="/opt/cray/pe/mpich/8.1.30/gtl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/lib64:${LD_LIBRARY_PATH:-}"
fi

CASE="${CASE:-emul}"
case "$CASE" in
  emul)
    COMPSET="${COMPSET:-E2000-EATM-EOCN}"
    RESOLUTION="${RESOLUTION:-gauss180x360_gauss180x360}"
    PECOUNT="${PECOUNT:-1x1}"
    NTASKS="${NTASKS:-1}"
    ;;
  f2010)
    COMPSET="${COMPSET:-F2010-ELM-EOCN}"
    RESOLUTION="${RESOLUTION:-ne30pg2_gauss180x360}"
    PECOUNT="${PECOUNT:-16x1}"
    NTASKS="${NTASKS:-16}"
    ;;
  *)
    echo "ERROR: CASE must be 'emul' or 'f2010', got '$CASE'" >&2
    exit 1
    ;;
esac

CASE_NAME="${CASE_NAME:-eocn-${CASE}}"
MACHINE="${MACHINE:-pm-gpu}"
PROJECT="${PROJECT:-e3sm}"
CASE_ROOT="${CASE_ROOT:-${PSCRATCH}/e3sm-repo/${CASE_NAME}}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

STOP_OPTION="${STOP_OPTION:-ndays}"
STOP_N="${STOP_N:-11}"
REST_OPTION="${REST_OPTION:-ndays}"
REST_N="${REST_N:-5}"
HIST_OPTION="${HIST_OPTION:-ndays}"
HIST_N="${HIST_N:-1}"
SUBMIT="${SUBMIT:-true}"

# EOCN's traced model and initial condition.  Produce them with
# tools/trace_eocn_model.py and tools/make_eocn_input.py; the namelist defaults
# already point here, so these are only for overriding.
EOCN_MODEL="${EOCN_MODEL:-}"
EOCN_IC="${EOCN_IC:-}"

# The emulated pair only makes sense with SamudrACE's atmosphere, which is not
# the EATM default (that is ACE2-EAMv3).
EATM_MODEL="${EATM_MODEL:-${PSCRATCH}/SamudrACE-E3SMv3/eatm/samudrace_atm_traced_cuda.pt}"
EATM_IC="${EATM_IC:-${PSCRATCH}/SamudrACE-E3SMv3/eatm/samudrace_atm_ic_0.nc}"

echo "=== ${CASE_NAME}: ${COMPSET} on ${RESOLUTION} (${STOP_N} ${STOP_OPTION}) ==="

cd "${CODE_ROOT}/cime/scripts"
./create_newcase --case "${CASE_ROOT}" --compset "${COMPSET}" --res "${RESOLUTION}" \
    --mach "${MACHINE}" --project "${PROJECT}" --pecount "${PECOUNT}" \
    --handle-preexisting-dirs r

cd "${CASE_ROOT}"

./xmlchange "NTASKS=${NTASKS},NTASKS_OCN=1,NTHRDS=1,ROOTPE=0"
./xmlchange "STOP_OPTION=${STOP_OPTION},STOP_N=${STOP_N}"
./xmlchange "REST_OPTION=${REST_OPTION},REST_N=${REST_N}"
./xmlchange "HIST_OPTION=${HIST_OPTION},HIST_N=${HIST_N}"
./xmlchange DOUT_S=FALSE

if [ -n "${EOCN_MODEL}" ]; then
  echo "eocn_model_file = '${EOCN_MODEL}'" >> user_nl_eocn
fi
if [ -n "${EOCN_IC}" ]; then
  echo "eocn_ic_file = '${EOCN_IC}'" >> user_nl_eocn
fi

if [ "${CASE}" = "emul" ]; then
  ./xmlchange EATM_EMULATOR=SamudrACE-E3SMv3
  ./xmlchange ATM_NCPL=48,OCN_NCPL=48
  cat >> user_nl_eatm <<EOF
eatm_model_file = '${EATM_MODEL}'
eatm_ic_file    = '${EATM_IC}'
eatm_frzprec_units = 'm/s'
EOF
fi

./case.setup --reset
./case.build

if [ "${SUBMIT}" != "true" ]; then
  echo "SUBMIT=false; built only.  Case is at ${CASE_ROOT}"
  exit 0
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
  ./case.submit --no-batch
else
  ./case.submit
fi
