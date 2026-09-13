/**
 * @file ocn_factory.cpp
 * @brief emulator_create_ocn, called by ocn_comp_mct.F90.
 */

#include "component_factory.hpp"
#include "ocean_operators.hpp"

extern "C" {

void *emulator_create_ocn(const EmulatorCreateConfig *cfg) {
  emulator::ocn::register_ocn_operators();
  return emulator::create_component("ocn", cfg, [] {
    return std::make_unique<emulator::EmulatorComponent>(
        emulator::EmulatorType::OCN_COMP, "emulatorocn");
  });
}

} // extern "C"
