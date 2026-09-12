/**
 * @file atm_factory.cpp
 * @brief emulator_create_atm, called by atm_comp_mct.F90.
 */

#include "atm.hpp"
#include "component_factory.hpp"

extern "C" {

void *emulator_create_atm(const EmulatorCreateConfig *cfg) {
  return emulator::create_component<emulator::EmulatorAtm>("atm", cfg);
}

} // extern "C"
