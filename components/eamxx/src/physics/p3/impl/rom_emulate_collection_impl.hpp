#ifndef ROM_EMULATE_COLLECTION_IMPL_HPP
#define ROM_EMULATE_COLLECTION_IMPL_HPP

#include "p3_functions.hpp" // for ETI only but harmless for GPU

#include <Python.h>
#include <pybind11/pybind11.h>
#include "share/atm_process/atmosphere_process_pyhelpers.hpp"
#include "share/core/eamxx_pysession.hpp"

namespace py = pybind11;

namespace scream {
namespace p3 {

/*
 * Implementation of SDM ROM microphysics emulator.
 * 
 */

template<typename S, typename D>
void Functions<S,D>
::rom_emulate_collection(
const Pack& qc_incld, const Pack& nc_incld, const Pack& qr_incld, const Pack& nr_incld, const Pack& rho, const Pack& mu_c, const Pack& mu_r, const Scalar& dtimestep, Pack& rom_emulate_qctend, Pack& rom_emulate_nctend, Pack& rom_emulate_qrtend, Pack& rom_emulate_nrtend, const Mask& context)
{

  // The lifetime of rom_module is owned by PySession, which clears it
  // before Py_Finalize() runs (eamxx_pysession.hpp). So if we got here
  // and the interpreter is still up, we are safe to call into Python.
  // A simple Py_IsInitialized() guard is enough; Py_IsFinalizing() is
  // 3.13-only public API and we don't need it now that the lifetime
  // bug is fixed at the source.
  if (!Py_IsInitialized()) {
    return;
  }

  PyGILState_STATE gstate = PyGILState_Ensure();
  //py::gil_scoped_acquire gil;

  auto& pysession = scream::PySession::get();
  if (!pysession.rom_module.has_value()) {
      printf("rom_module not initialized!\n");
      PyGILState_Release(gstate);
      return;
    }

  const auto& py_mod = std::any_cast<const py::module&>(pysession.rom_module);

  constexpr Scalar QSMALL  = C::QSMALL;
  const     Scalar QSMALL2 = 1.0e-8;
  const auto qc_large_enough = qc_incld >= QSMALL2 && context;

  const auto qc_not_small = qc_incld >= QSMALL && context;
  const auto qr_not_small = qr_incld >= QSMALL && context;

  try { 

    if (qc_large_enough.any() || (qc_not_small.any() && qr_not_small.any()) || qr_not_small.any()) {
       for(int i = 0; i < Pack::n; ++i) {
            if (qc_not_small[i] || qr_not_small[i]) {
                
                //qc_val = qc_incld[i] * rho[i];  // unit: kg/m3
                //qr_val = qr_incld[i] * rho[i];  // unit: 1/m3
                //nc_val = nc_incld[i] * rho[i];  // unit: kg/m3
                //nr_val = nr_incld[i] * rho[i];  // unit: 1/m3
                //printf("rank qc_incld = %d\n", qc_incld[i].rank());

                // Import SDM_interface python module and call function with scalars
                //printf("BEFORE ROM_interface call\n");
                py::tuple result = py_mod.attr("ROM_interface")( 
                          qc_incld[i] * rho[i], nc_incld[i] * rho[i], 
                          qr_incld[i] * rho[i], nr_incld[i] * rho[i], 
                          mu_c[i], mu_r[i], QSMALL, dtimestep);

                //printf("AFTER ROM_interface call\n");
                // Unpack tuple: retrieve tendency rate terms
                double qctend = result[0].cast<double>()/rho[i];  // unit: kg/kg/s
                double nctend = result[1].cast<double>()/rho[i];  // unit: 1/kg/s
                double qrtend = result[2].cast<double>()/rho[i];  // unit: kg/kg/s
                double nrtend = result[3].cast<double>()/rho[i];  // unit: 1/kg/s

                //printf("ALICE: qctend = %-+.16E\n",qctend);
                //printf("ALICE: ncttend = %-+.16E\n",nctend);
                //printf("ALICE: qrtend = %-+.16E\n",qrtend);
                //printf("ALICE: nrtend = %-+.16E\n",nrtend);

                // 20250714. YPS. Important to care about the sign. be consistent with default model!!
                // from SDM ROM, qctend is negative. qrtend is positive. nctend is negative. nrtend could be negative/positive. This inconsistency once cause NaN values in p3. No error in COMBLE case. NaNs in GoAmazon cases.
                rom_emulate_qctend[i] = (-1)*qctend;
                rom_emulate_nctend[i] = (-1)*nctend;
                rom_emulate_qrtend[i] = qrtend;
                rom_emulate_nrtend[i] = (-1)*nrtend;

            } else {

                rom_emulate_qctend[i] = 0;
                rom_emulate_nctend[i] = 0;
                rom_emulate_qrtend[i] = 0;
                rom_emulate_nrtend[i] = 0;

            }

        }  // for loop

    } // if cond

  } catch (py::error_already_set& e) {
           printf("ALICE: Python call error: %s\n",  e.what());
           PyGILState_Release(gstate);
           throw;
         }

  PyGILState_Release(gstate);

} //rom_emulate_collection

} // namespace p3
} // namespace scream

#endif
