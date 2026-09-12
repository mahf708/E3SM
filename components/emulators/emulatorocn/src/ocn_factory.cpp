/**
 * @file ocn_factory.cpp
 * @brief emulator_create_ocn, called by ocn_comp_mct.F90.
 */

#include "component_factory.hpp"
#include "ocn.hpp"

extern "C" {

void *emulator_create_ocn(const EmulatorCreateConfig *cfg) {
  return emulator::create_component<emulator::EmulatorOcn>("ocn", cfg);
}

} // extern "C"
