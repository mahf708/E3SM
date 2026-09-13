/**
 * @file ice_factory.cpp
 * @brief emulator_create_ice, called by ice_comp_mct.F90.
 */

#include "component_factory.hpp"
#include "sea_ice_operators.hpp"

extern "C" {

void *emulator_create_ice(const EmulatorCreateConfig *cfg) {
  emulator::ice::register_ice_operators();
  return emulator::create_component("ice", cfg, [] {
    return std::make_unique<emulator::EmulatorComponent>(
        emulator::EmulatorType::ICE_COMP, "emulatorice");
  });
}

} // extern "C"
