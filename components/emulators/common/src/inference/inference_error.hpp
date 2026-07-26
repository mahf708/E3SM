/**
 * @file inference_error.hpp
 * @brief Error type and check macro shared by the inference layer.
 */

#ifndef E3SM_EMULATOR_INFERENCE_ERROR_HPP
#define E3SM_EMULATOR_INFERENCE_ERROR_HPP

#include <sstream>
#include <stdexcept>
#include <string>

namespace emulator {
namespace inference {

/**
 * @brief Exception thrown for unrecoverable inference errors.
 *
 * Used for programming/configuration errors (bad shapes, missing models,
 * unknown backends, ...).  Per-step failures that a caller may reasonably
 * want to handle without unwinding are reported through the boolean return
 * value of InferenceBackend::infer() instead.
 */
class InferenceError : public std::runtime_error {
public:
  explicit InferenceError(const std::string &what)
      : std::runtime_error("[emulator::inference] " + what) {}
};

} // namespace inference
} // namespace emulator

/**
 * @brief Throw an InferenceError with a streamed message unless cond holds.
 *
 * Usage: `EMULATOR_INFER_REQUIRE(n > 0, "bad size: " << n);`
 */
#define EMULATOR_INFER_REQUIRE(cond, msg)                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::ostringstream _emu_oss;                                             \
      _emu_oss << msg;                                                         \
      throw ::emulator::inference::InferenceError(_emu_oss.str());             \
    }                                                                          \
  } while (false)

#endif // E3SM_EMULATOR_INFERENCE_ERROR_HPP
