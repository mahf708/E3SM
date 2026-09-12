/**
 * @file ice_factory.cpp
 * @brief emulator_create_ice, called by ice_comp_mct.F90.
 */

#include "component_factory.hpp"
#include "ice.hpp"

extern "C" {

void *emulator_create_ice(const EmulatorCreateConfig *cfg) {
  return emulator::create_component<emulator::EmulatorIce>("ice", cfg);
}

} // extern "C"
