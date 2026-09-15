/**
 * @file fpe_guard.hpp
 * @brief RAII suspension of floating-point exception traps.
 */

#ifndef E3SM_EMULATOR_FPE_GUARD_HPP
#define E3SM_EMULATOR_FPE_GUARD_HPP

namespace emulator {
namespace inference {

/**
 * @brief Disable FPE traps for the lifetime of the object, then restore
 *        exactly the set that was enabled before.
 *
 * A no-op where `feenableexcept` does not exist (it is a glibc extension);
 * there is nothing to guard against on those platforms, and an #error would
 * be worse than a no-op.
 */
class FpeGuard {
public:
  FpeGuard();
  ~FpeGuard();
  FpeGuard(const FpeGuard &) = delete;
  FpeGuard &operator=(const FpeGuard &) = delete;

private:
  int m_saved_excepts = 0;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_FPE_GUARD_HPP
