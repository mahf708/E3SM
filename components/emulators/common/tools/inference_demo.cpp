/**
 * @file inference_demo.cpp
 * @brief Drive any inference backend from a configuration file.
 *
 * A standalone harness for the inference layer: it builds a backend, reports
 * what the model expects, feeds it synthetic columns, and times the steps.
 * Useful for
 *  - checking that a model loads and runs before wiring it into a component,
 *  - comparing backends on the same model (export a model to both TorchScript
 *    and ONNX and run it through each),
 *  - measuring per-step cost against a component's time step.
 *
 * Build: configure with -DBUILD_EMULATOR_TESTS=ON, then run
 *   ./emulator_inference_demo --help
 */

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "create_inference_backend.hpp"

namespace {

using namespace emulator::inference;

void print_usage(const char *argv0) {
  std::cout
      << "usage: " << argv0 << " [options]\n\n"
      << "Options:\n"
      << "  --config FILE       inference configuration (key: value lines)\n"
      << "  --prefix STR        only read keys with this prefix from FILE,\n"
      << "                      e.g. --prefix inference. for a component "
         "namelist\n"
      << "  --backend KEY       backend to use (overrides the config)\n"
      << "  --model PATH        model artifact (overrides the config)\n"
      << "  --set KEY=VALUE     apply any config setting, as a config file\n"
      << "                      line would (repeatable), e.g.\n"
      << "                      --set input=T[-1,72]:float32 --set device=cuda\n"
      << "  --columns N         columns (batch size) per step [16]\n"
      << "  --steps N           number of steps to run [1]\n"
      << "  --print N           print the first N output values [4]\n"
      << "  --verbose           verbose backend output\n"
      << "  --list              list the available backends and exit\n"
      << "  --help              show this message\n\n"
      << "Example:\n"
      << "  " << argv0 << " --backend onnx --model atm.onnx --columns 384 "
      << "--steps 10\n";
}

/// Fill a tensor with a deterministic, non-degenerate pattern.
void fill_pattern(Tensor &tensor, std::int64_t seed) {
  for (std::int64_t i = 0; i < tensor.size(); ++i) {
    // Bounded, varying, and identical across runs so results are comparable.
    const double value =
        0.5 + 0.25 * static_cast<double>((i + seed) % 7) -
        0.1 * static_cast<double>((i + seed) % 3);
    switch (tensor.dtype()) {
    case DType::FLOAT32:
      tensor.flat<float>(i) = static_cast<float>(value);
      break;
    case DType::FLOAT64:
      tensor.flat<double>(i) = value;
      break;
    case DType::INT32:
      tensor.flat<std::int32_t>(i) = static_cast<std::int32_t>(i % 5);
      break;
    case DType::INT64:
      tensor.flat<std::int64_t>(i) = static_cast<std::int64_t>(i % 5);
      break;
    }
  }
}

void print_head(const Tensor &tensor, std::int64_t count) {
  std::cout << "    " << std::left << std::setw(28) << tensor.to_string()
            << std::right;
  const std::int64_t n = std::min(count, tensor.size());
  for (std::int64_t i = 0; i < n; ++i) {
    double value = 0.0;
    switch (tensor.dtype()) {
    case DType::FLOAT32:
      value = tensor.cflat<float>(i);
      break;
    case DType::FLOAT64:
      value = tensor.cflat<double>(i);
      break;
    case DType::INT32:
      value = tensor.cflat<std::int32_t>(i);
      break;
    case DType::INT64:
      value = static_cast<double>(tensor.cflat<std::int64_t>(i));
      break;
    }
    std::cout << " " << std::setw(12) << std::setprecision(6) << value;
  }
  if (tensor.size() > n) {
    std::cout << " ...";
  }
  std::cout << "\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string config_file;
  std::string prefix;
  std::string backend_override;
  std::string model_override;
  std::vector<std::pair<std::string, std::string>> overrides;
  std::int64_t columns = 16;
  int steps = 1;
  std::int64_t print_count = 4;
  bool verbose = false;

  const auto next_arg = [argc, argv](int &i, const char *what) {
    ++i;
    if (i >= argc) {
      std::cerr << "error: " << what << " needs a value\n";
      std::exit(2);
    }
    return std::string(argv[i]);
  };

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--list") {
      std::cout << "available backends: "
                << BackendRegistry::instance().available_string() << "\n";
      return 0;
    } else if (arg == "--config") {
      config_file = next_arg(i, "--config");
    } else if (arg == "--prefix") {
      prefix = next_arg(i, "--prefix");
    } else if (arg == "--backend") {
      backend_override = next_arg(i, "--backend");
    } else if (arg == "--model") {
      model_override = next_arg(i, "--model");
    } else if (arg == "--set") {
      const std::string kv = next_arg(i, "--set");
      const auto eq = kv.find('=');
      if (eq == std::string::npos) {
        std::cerr << "error: --set expects KEY=VALUE, got '" << kv << "'\n";
        return 2;
      }
      overrides.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    } else if (arg == "--columns") {
      columns = std::stoll(next_arg(i, "--columns"));
    } else if (arg == "--steps") {
      steps = std::stoi(next_arg(i, "--steps"));
    } else if (arg == "--print") {
      print_count = std::stoll(next_arg(i, "--print"));
    } else if (arg == "--verbose") {
      verbose = true;
    } else {
      std::cerr << "error: unknown option '" << arg << "'\n\n";
      print_usage(argv[0]);
      return 2;
    }
  }

  try {
    InferenceConfig config;
    if (!config_file.empty()) {
      config = prefix.empty()
                   ? InferenceConfig::from_file(config_file)
                   : InferenceConfig::from_file_with_prefix(config_file, prefix);
    }
    if (!backend_override.empty()) {
      config.backend = backend_override;
    }
    if (!model_override.empty()) {
      config.model_path = model_override;
    }
    for (const auto &kv : overrides) {
      config.apply(kv.first, kv.second);
    }
    config.verbose = config.verbose || verbose;

    std::cout << "=== configuration ===\n" << config.to_string();

    auto backend = create_backend(config);
    backend->initialize();
    std::cout << "\n=== backend ===\n" << backend->to_string();

    const auto in_specs = backend->input_specs();
    if (in_specs.empty()) {
      std::cerr << "\nerror: neither the configuration nor the model says what "
                   "the inputs are.\n       Add `input:`/`output:` lines (or "
                   "input_channels/output_channels)\n       to the "
                   "configuration.\n";
      return 1;
    }

    TensorMap inputs = backend->make_inputs(columns);
    TensorMap outputs = backend->make_outputs(columns);

    std::cout << "\n=== running " << steps << " step(s) with " << columns
              << " column(s) ===\n";

    double total_ms = 0.0;
    double first_ms = 0.0;
    for (int step = 0; step < steps; ++step) {
      std::int64_t seed = step;
      for (auto &tensor : inputs) {
        fill_pattern(tensor, seed++);
      }

      const auto t0 = std::chrono::steady_clock::now();
      const bool ok = backend->infer(inputs, outputs);
      const auto t1 = std::chrono::steady_clock::now();
      if (!ok) {
        std::cerr << "error: inference reported failure on step " << step
                  << "\n";
        return 1;
      }

      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      total_ms += ms;
      if (step == 0) {
        first_ms = ms;
        std::cout << "  inputs:\n";
        for (const auto &tensor : inputs) {
          print_head(tensor, print_count);
        }
        std::cout << "  outputs:\n";
        for (const auto &tensor : outputs) {
          print_head(tensor, print_count);
        }
      }
    }

    std::cout << "\n=== timing ===\n";
    std::cout << "  first step : " << std::fixed << std::setprecision(3)
              << first_ms << " ms (includes lazy runtime setup)\n";
    if (steps > 1) {
      const double rest = (total_ms - first_ms) / static_cast<double>(steps - 1);
      std::cout << "  later steps: " << rest << " ms/step ("
                << rest * 1000.0 / static_cast<double>(columns)
                << " us/column)\n";
    }
    std::cout << "  total      : " << total_ms << " ms for " << steps
              << " step(s)\n";

    backend->finalize();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "\nerror: " << e.what() << "\n";
    return 1;
  }
}
