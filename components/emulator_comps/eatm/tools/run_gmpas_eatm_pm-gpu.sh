#!/bin/bash
# Create, build and submit a GMPAS-EATM case on Perlmutter GPU:
# emulated atmosphere (EATM) coupled to a prognostic MPAS-Ocean and MPAS-Seaice.
#
#   COMPSET  GMPAS-EATM = 2000_EATM_SLND_MPASSI_MPASO%DATMFORCED_DROF%JRA-1p5-2023_SGLC_SWAV
#   GRID     gauss180x360_IcoswISC30E3r5  (1 deg emulator grid, ~30 km ocean/ice)
#
# Land is a stub and runoff is a data model; ocean and sea ice are prognostic.
#
# Defaults to a 5-year run submitted as five 1-year segments (STOP_N=1,
# RESUBMIT=4) on 8 nodes, which is also the pm-gpu debug queue's node limit so
# the same layout can be smoke-tested there.  A 2-year shakedown of this
# compset ran at 5.6 SYPD on 11 nodes; on 8 nodes expect roughly 4-4.5 SYPD,
# i.e. ~6 h per 1-year segment.
#
# Usage:
#   ./run_gmpas_eatm_pm-gpu.sh                        # build and submit
#   CASE_NAME=my-run STOP_N=1 RESUBMIT=4 ./run_gmpas_eatm_pm-gpu.sh
#   SUBMIT=false ./run_gmpas_eatm_pm-gpu.sh           # build only
#
#   # 3-day smoke test in the debug queue (<=8 nodes, <=30 min)
#   CASE_NAME=smoke STOP_OPTION=ndays STOP_N=3 RESUBMIT=0 \
#     QUEUE=debug WALLCLOCK=00:30:00 ./run_gmpas_eatm_pm-gpu.sh
#
set -euo pipefail

# --- toolchain shim, required since the 2026-08-17 NERSC CPE roll --------------
# The system default /opt/cray/pe/lib64/libmpi_gtl_cuda.so.0 is now a CUDA 13
# build, while E3SM's pm-gpu configuration pins cudatoolkit/12.9 (FTorch is
# built against libtorch 2.10.0+cu128).  The Cray wrappers link that GTL shim
# whenever craype-accel-nvidia80 is loaded, so every binary they produce -- down
# to the throwaway one CMake compiles to check the CUDA header version --
# acquires a libcudart.so.13 dependency a 12.9 environment cannot satisfy, and
# the build dies inside find_package(FTorch) with a message that reads like a
# CUDA_HOME misconfiguration and is not one.  Putting the CUDA-12 GTL and the
# 12.9 runtime ahead of the system default fixes it.  See eatm/REVIEW.md #74.
#
# `module reset` matters on its own: an inherited interactive environment
# carries /usr/local/cuda-13.2/compat on LD_LIBRARY_PATH, which CIME neither
# manages nor removes.  Set EATM_SKIP_ENV_SHIM=1 once the machine files pin the
# GTL search path themselves.
if [ "${EATM_SKIP_ENV_SHIM:-0}" != "1" ] && [ "${MACHINE:-pm-gpu}" = "pm-gpu" ]; then
  if command -v module >/dev/null 2>&1; then
    module reset >/dev/null 2>&1 || true
    module load cray-python >/dev/null 2>&1 || true
  fi
  export LD_LIBRARY_PATH="/opt/cray/pe/mpich/8.1.30/gtl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/lib64:${LD_LIBRARY_PATH:-}"
fi

# --- what to run -------------------------------------------------------------
EMULATOR="${EMULATOR:-SamudrACE-E3SMv3}"   # or ACE2-EAMv3
CASE_NAME="${CASE_NAME:-GMPAS-EATM-${EMULATOR}-5yr}"
COMPSET="${COMPSET:-GMPAS-EATM}"
RESOLUTION="${RESOLUTION:-gauss180x360_IcoswISC30E3r5}"

# --- where ------------------------------------------------------------------
MACHINE="${MACHINE:-pm-gpu}"
COMPILER="${COMPILER:-gnugpu}"
PROJECT="${PROJECT:-e3sm_g}"
CASE_ROOT="${CASE_ROOT:-${PSCRATCH}/E3SMv3/${CASE_NAME}}"

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# --- run length --------------------------------------------------------------
# five 1-year segments; each segment is one batch job
STOP_OPTION="${STOP_OPTION:-nyears}"
STOP_N="${STOP_N:-1}"
RESUBMIT="${RESUBMIT:-4}"
WALLCLOCK="${WALLCLOCK:-08:00:00}"
QUEUE="${QUEUE:-regular}"
REST_OPTION="${REST_OPTION:-nmonths}"
REST_N="${REST_N:-1}"
HIST_OPTION="${HIST_OPTION:-nmonths}"
HIST_N="${HIST_N:-1}"

SUBMIT="${SUBMIT:-true}"

# --- emulator artifacts -------------------------------------------------------
# Left empty for ACE2-EAMv3, which has working defaults in
# bld/namelist_files/namelist_defaults_eatm.xml.
STAGE="${EATM_STAGE:-${PSCRATCH}/SamudrACE-E3SMv3/eatm}"
IC_INDEX="${IC_INDEX:-0}"
EATM_MODEL_FILE="${EATM_MODEL_FILE:-}"
EATM_IC_FILE="${EATM_IC_FILE:-}"

if [[ "${EMULATOR}" == "SamudrACE-E3SMv3" ]]; then
    EATM_MODEL_FILE="${EATM_MODEL_FILE:-${STAGE}/samudrace_atm_traced_cuda.pt}"
    EATM_IC_FILE="${EATM_IC_FILE:-${STAGE}/samudrace_atm_ic_${IC_INDEX}.nc}"
    for f in "${EATM_MODEL_FILE}" "${EATM_IC_FILE}"; do
        if [[ ! -s "${f}" ]]; then
            cat >&2 <<EOF
ERROR: missing ${f}

The SamudrACE-E3SMv3 atmosphere has to be traced to TorchScript before EATM can
run it.  On a GPU node:

  salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --account=${PROJECT}
  ${CODE_ROOT}/components/emulator_comps/eatm/tools/trace_samudrace_atmosphere.sh

Then re-run this script.  To use the already-traced ACE2-EAMv3 model instead:

  EMULATOR=ACE2-EAMv3 \$0
EOF
            exit 1
        fi
    done
fi

echo "=============================================================="
echo " case      : ${CASE_NAME}"
echo " compset   : ${COMPSET}"
echo " grid      : ${RESOLUTION}"
echo " emulator  : ${EMULATOR}"
echo " length    : ${STOP_N} ${STOP_OPTION} x $((RESUBMIT + 1)) segments"
echo " code root : ${CODE_ROOT}"
echo " case root : ${CASE_ROOT}"
echo "=============================================================="

umask 022

"${CODE_ROOT}/cime/scripts/create_newcase" \
    --case "${CASE_ROOT}" \
    --mach "${MACHINE}" \
    --res "${RESOLUTION}" \
    --compset "${COMPSET}" \
    --compiler "${COMPILER}" \
    --project "${PROJECT}"

cd "${CASE_ROOT}"

# --- PE layout ---------------------------------------------------------------
# EATM is serial and runs the emulator on one GPU, so it gets a node of its own
# at global rank 0; everything else shares the remaining 7 nodes starting at
# rank 64.  8 nodes total, which is the pm-gpu debug queue's limit.
./xmlchange MAX_MPITASKS_PER_NODE=64
./xmlchange NTASKS=-7
./xmlchange NTASKS_ATM=1
./xmlchange NTASKS_ESP=1
./xmlchange NTASKS_IAC=1
./xmlchange ROOTPE=64
./xmlchange ROOTPE_ATM=0
./xmlchange ROOTPE_WAV=1
./xmlchange ROOTPE_GLC=1
./xmlchange PSTRID_ATM=16
./xmlchange EXCL_STRIDE_ATM=16

# --- run control -------------------------------------------------------------
./xmlchange DEBUG=false
./xmlchange JOB_QUEUE="${QUEUE}"
./xmlchange JOB_WALLCLOCK_TIME="${WALLCLOCK}"
./xmlchange STOP_OPTION="${STOP_OPTION}",STOP_N="${STOP_N}"
./xmlchange RESUBMIT="${RESUBMIT}"
./xmlchange REST_OPTION="${REST_OPTION}",REST_N="${REST_N}"
./xmlchange HIST_OPTION="${HIST_OPTION}",HIST_N="${HIST_N}"
# no short-term archiving: it queues a second dependent job per segment and
# moves output out from under you mid-run.  Output stays in RUNDIR.
./xmlchange DOUT_S=FALSE

# --- emulator ----------------------------------------------------------------
./xmlchange EATM_EMULATOR="${EMULATOR}"

if [[ -n "${EATM_MODEL_FILE}" || -n "${EATM_IC_FILE}" ]]; then
    {
        echo ""
        echo "! set by run_gmpas_eatm_pm-gpu.sh"
        [[ -n "${EATM_MODEL_FILE}" ]] && echo "eatm_model_file = '${EATM_MODEL_FILE}'"
        [[ -n "${EATM_IC_FILE}"    ]] && echo "eatm_ic_file    = '${EATM_IC_FILE}'"
        echo "eatm_model_device = 'gpu'"
    } >> user_nl_eatm
fi

./case.setup
./case.build

if [[ "${SUBMIT}" == "true" ]]; then
    ./case.submit
    echo ""
    echo "Submitted.  Watch it with:"
    echo "  cd ${CASE_ROOT} && ./case.qstatus"
    echo "  tail -f \$(./xmlquery -value RUNDIR)/atm.log.*"
else
    echo ""
    echo "Built but not submitted (SUBMIT=false).  Submit with:"
    echo "  cd ${CASE_ROOT} && ./case.submit"
fi
