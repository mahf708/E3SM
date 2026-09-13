/**
 * @file atm_factory.cpp
 * @brief emulator_create_atm, called by atm_comp_mct.F90.
 */

#include "component_factory.hpp"
#include "ace_operators.hpp"

extern "C" {

void *emulator_create_atm(const EmulatorCreateConfig *cfg) {
  emulator::atm::register_atm_operators();
  return emulator::create_component("atm", cfg, [] {
    return std::make_unique<emulator::EmulatorComponent>(
        emulator::EmulatorType::ATM_COMP, "emulatoratm");
  });
}

} // extern "C"
