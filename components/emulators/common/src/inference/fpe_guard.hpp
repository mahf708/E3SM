/**
 * @file fpe_guard.hpp
 * @brief RAII suspension of floating-point exception traps.
 *
 * An E3SM debug build enables trapping on invalid, divide-by-zero and
 * overflow.  Both machine-learning backends need it off around the model
 * call, for the same reason and independently of each other: importing numpy
 * raises benign FPEs, and libtorch's kernels raise them too -- a softmax or
 * a normalization that divides by a zero it is about to mask is perfectly
 * ordinary inside a network and fatal under a trap.
 *
 * This lives in its own header, compiled unconditionally, because it started
 * out inside the embedded-Python backend and was therefore compiled only
 * when Python was enabled.  A libtorch-only build in debug would have had no
 * guard at all, and nothing to catch the trap when it fired.
 *
 * EAMxx's PySession does the same thing for the same reason.
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
