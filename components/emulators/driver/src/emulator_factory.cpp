/**
 * @file emulator_factory.cpp
 * @brief emulator_create(kind): one entry point over the per-component
 *        factories, for tools and tests that choose a kind at run time.
 *
 * The caps call emulator_create_<kind> directly; this library is not linked
 * into the model executable.
 */

#include "emulator_c_api.hpp"

#include <cstring>

extern "C" {

void *emulator_create(const char *kind, const EmulatorCreateConfig *cfg) {
  if (kind == nullptr) {
    return nullptr;
  }
  if (std::strcmp(kind, "atm") == 0) {
    return emulator_create_atm(cfg);
  }
  if (std::strcmp(kind, "ocn") == 0) {
    return emulator_create_ocn(cfg);
  }
  if (std::strcmp(kind, "ice") == 0) {
    return emulator_create_ice(cfg);
  }
  return nullptr;
}

} // extern "C"
