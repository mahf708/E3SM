/**
 * @file inference_config.hpp
 * @brief Backend-agnostic inference configuration.
 */

#ifndef E3SM_EMULATOR_INFERENCE_CONFIG_HPP
#define E3SM_EMULATOR_INFERENCE_CONFIG_HPP

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Configuration for an inference backend.
 *
 * Holds the handful of settings every backend needs (which backend, which
 * model file, what the inputs/outputs look like) plus a free-form string
 * option map for everything backend specific.  New backends therefore need
 * no changes here, and a component can pass options straight through from
 * its namelist without this header knowing about them.
 *
 * Values can be built programmatically or read from a small line-oriented
 * text format (see from_string()), deliberately close to the `key: value`
 * parsing the emulator components already do on their `*_in` files.  A
 * richer format (YAML via ekat) can be layered on top later without
 * touching backends.
 */
struct InferenceConfig {
  // -------------------------------------------------------------------------
  // Core settings
  // -------------------------------------------------------------------------

  /// Registered backend key, e.g. "stub", "python", "torch", "onnx".
  std::string backend = "stub";

  /// Path to the model artifact (TorchScript .pt, .onnx, ...), if any.
  std::string model_path;

  /// Declared inputs.  May be empty if the backend introspects the model.
  std::vector<TensorSpec> inputs;

  /// Declared outputs.  May be empty if the backend introspects the model.
  std::vector<TensorSpec> outputs;

  /**
   * @brief Input features per batch element for the flat-array convenience
   *        path (InferenceBackend::infer(const double*, double*, int)).
   *
   * Ignored when `inputs` is populated.
   */
  int input_channels = 0;

  /// Output features per batch element for the flat-array path.
  int output_channels = 0;

  /// Emit diagnostic output on construction/initialization/inference.
  bool verbose = false;

  /// Backend-specific options (see the backend headers for keys).
  std::map<std::string, std::string> options;

  // -------------------------------------------------------------------------
  // Option accessors
  // -------------------------------------------------------------------------

  /// @brief True if the option is present.
  bool has(const std::string &key) const;

  /// @brief Raw option value, or `fallback` when absent.
  std::string get(const std::string &key,
                  const std::string &fallback = "") const;

  /// @brief Option parsed as int; throws InferenceError if unparseable.
  int get_int(const std::string &key, int fallback = 0) const;

  /// @brief Option parsed as double; throws InferenceError if unparseable.
  double get_double(const std::string &key, double fallback = 0.0) const;

  /**
   * @brief Option parsed as bool.
   *
   * Accepts true/false, yes/no, on/off, 1/0, .true./.false. (any case).
   */
  bool get_bool(const std::string &key, bool fallback = false) const;

  /// @brief Same as get(), but throws InferenceError when absent or empty.
  std::string get_required(const std::string &key,
                           const std::string &context = "") const;

  /// @brief Set an option (string).
  InferenceConfig &set(const std::string &key, const std::string &value);

  /// @brief Set an option from any streamable value (int, double, bool, ...).
  template <typename T>
  InferenceConfig &set(const std::string &key, const T &value) {
    std::ostringstream oss;
    oss << std::boolalpha << value;
    return set(key, oss.str());
  }

  /**
   * @brief Apply one `key: value` setting, as the file parser would.
   *
   * Recognized keys (`backend`, `model_path`, `input`, `output`,
   * `input_channels`, `output_channels`, `verbose`) update the corresponding
   * field; `input`/`output` append a spec.  Anything else becomes an option.
   * Use this to feed settings in one at a time — from a component namelist,
   * a command line, or a driver — without going through a text blob.
   *
   * @throws InferenceError if the value cannot be interpreted for that key
   */
  InferenceConfig &apply(const std::string &key, const std::string &value);

  // -------------------------------------------------------------------------
  // Parsing / reporting
  // -------------------------------------------------------------------------

  /**
   * @brief Parse the line-oriented configuration format.
   *
   * Blank lines and lines starting with `#` or `!` are ignored.  Every other
   * line is `key: value`.  Recognized keys:
   *
   * ```
   * backend:         onnx                  # registered backend key
   * model_path:      /path/to/model.onnx
   * input:           T[-1,72]:float32      # repeatable, ordered
   * output:          dT[-1,72]:float32     # repeatable, ordered
   * input_channels:  4                     # flat-array convenience path
   * output_channels: 2
   * verbose:         true
   * ```
   *
   * Any other key becomes an entry in `options`; a leading `option.` prefix
   * is stripped, so `option.device: cuda` and `device: cuda` are equivalent.
   *
   * @throws InferenceError on a malformed line or tensor spec
   */
  static InferenceConfig from_string(const std::string &text);

  /// @brief Read from_string() content from a file.
  /// @throws InferenceError if the file cannot be opened
  static InferenceConfig from_file(const std::string &path);

  /**
   * @brief Extract an inference section out of a component namelist.
   *
   * Reads the same `key: value` syntax but only picks up keys carrying
   * `prefix` (default `inference.`), so an emulator's `atm_in` can carry
   * inference settings inline:
   *
   * ```
   * nx: 90
   * inference.backend: python
   * inference.python_module: my_emulator
   * ```
   *
   * Lines without the prefix are ignored, which lets a component hand its
   * whole namelist to the inference layer.
   */
  static InferenceConfig from_string_with_prefix(
      const std::string &text, const std::string &prefix = "inference.");

  /// @brief from_string_with_prefix() applied to the contents of a file.
  static InferenceConfig from_file_with_prefix(
      const std::string &path, const std::string &prefix = "inference.");

  /// @brief Multi-line human-readable dump (for logs).
  std::string to_string() const;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_CONFIG_HPP
