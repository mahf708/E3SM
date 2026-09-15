/**
 * @file fpe_guard.cpp
 * @brief Implementation of FpeGuard.
 */

#include "fpe_guard.hpp"

#include <cfenv>

// feenableexcept and friends are glibc extensions.  Everywhere else there is
// nothing to guard against, so FpeGuard becomes a no-op rather than an
// #error: trapping is what E3SM debug builds do on Linux.
#if defined(__GLIBC__)
#define EMULATOR_HAVE_FEENABLEEXCEPT 1
#endif

namespace emulator {
namespace inference {

FpeGuard::FpeGuard() {
#ifdef EMULATOR_HAVE_FEENABLEEXCEPT
  m_saved_excepts = fegetexcept();
  if (m_saved_excepts > 0) {
    fedisableexcept(m_saved_excepts);
  }
#endif
}

FpeGuard::~FpeGuard() {
#ifdef EMULATOR_HAVE_FEENABLEEXCEPT
  if (m_saved_excepts > 0) {
    feclearexcept(FE_ALL_EXCEPT);
    feenableexcept(m_saved_excepts);
  }
#endif
}

} // namespace inference
} // namespace emulator
