/**
 * @file stub_inference_backend.cpp
 * @brief Dependency-free inference backend implementation.
 */

#include "stub_inference_backend.hpp"

#include <algorithm>
#include <iostream>

namespace emulator {
namespace inference {

namespace {

StubBackend::Mode parse_mode(const std::string &raw) {
  std::string key;
  std::transform(raw.begin(), raw.end(), std::back_inserter(key),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (key.empty() || key == "noop" || key == "none" || key == "passthrough") {
    return StubBackend::Mode::NOOP;
  }
  if (key == "zero" || key == "zeros") {
    return StubBackend::Mode::ZERO;
  }
  if (key == "constant" || key == "const" || key == "fill") {
    return StubBackend::Mode::CONSTANT;
  }
  if (key == "copy" || key == "identity") {
    return StubBackend::Mode::COPY;
  }
  if (key == "affine" || key == "linear") {
    return StubBackend::Mode::AFFINE;
  }
  EMULATOR_INFER_REQUIRE(false, "Unknown stub backend mode '"
                                    << raw
                                    << "'. Valid modes: noop, zero, constant, "
                                       "copy, affine.");
  return StubBackend::Mode::NOOP; // unreachable
}

/// Fill every element of `t` with `value`, whatever its element type.
void fill(Tensor &t, double value) {
  const std::int64_t n = t.size();
  switch (t.dtype()) {
  case DType::FLOAT32: {
    auto *p = t.data<float>();
    std::fill(p, p + n, static_cast<float>(value));
    break;
  }
  case DType::FLOAT64: {
    auto *p = t.data<double>();
    std::fill(p, p + n, value);
    break;
  }
  case DType::INT32: {
    auto *p = t.data<std::int32_t>();
    std::fill(p, p + n, static_cast<std::int32_t>(value));
    break;
  }
  case DType::INT64: {
    auto *p = t.data<std::int64_t>();
    std::fill(p, p + n, static_cast<std::int64_t>(value));
    break;
  }
  }
}

/// dst[i] = scale * src[i] + offset, converting element types as needed.
void affine(const Tensor &src, Tensor &dst, double scale, double offset) {
  EMULATOR_INFER_REQUIRE(src.size() == dst.size(),
                         "Stub backend (affine/copy): '"
                             << src.name() << "' has " << src.size()
                             << " elements but '" << dst.name() << "' has "
                             << dst.size() << ".");
  const std::int64_t n = dst.size();
  const auto read = [&src](std::int64_t i) -> double {
    switch (src.dtype()) {
    case DType::FLOAT32:
      return static_cast<double>(src.data<float>()[i]);
    case DType::FLOAT64:
      return src.data<double>()[i];
    case DType::INT32:
      return static_cast<double>(src.data<std::int32_t>()[i]);
    case DType::INT64:
      return static_cast<double>(src.data<std::int64_t>()[i]);
    }
    return 0.0;
  };

  switch (dst.dtype()) {
  case DType::FLOAT32: {
    auto *p = dst.data<float>();
    for (std::int64_t i = 0; i < n; ++i) {
      p[i] = static_cast<float>(scale * read(i) + offset);
    }
    break;
  }
  case DType::FLOAT64: {
    auto *p = dst.data<double>();
    for (std::int64_t i = 0; i < n; ++i) {
      p[i] = scale * read(i) + offset;
    }
    break;
  }
  case DType::INT32: {
    auto *p = dst.data<std::int32_t>();
    for (std::int64_t i = 0; i < n; ++i) {
      p[i] = static_cast<std::int32_t>(scale * read(i) + offset);
    }
    break;
  }
  case DType::INT64: {
    auto *p = dst.data<std::int64_t>();
    for (std::int64_t i = 0; i < n; ++i) {
      p[i] = static_cast<std::int64_t>(scale * read(i) + offset);
    }
    break;
  }
  }
}

} // namespace

StubBackend::StubBackend(const InferenceConfig &config)
    : InferenceBackend(config) {
  m_mode = parse_mode(config.get("mode"));
  m_value = config.get_double("value", 0.0);
  m_scale = config.get_double("scale", 1.0);
  m_offset = config.get_double("offset", 0.0);
}

StubBackend::~StubBackend() { StubBackend::finalize(); }

void StubBackend::init_impl() {
  if (m_config.verbose) {
    std::cout << "[stub inference] mode=" << m_config.get("mode", "noop")
              << " (no model is loaded)\n";
  }
}

void StubBackend::ensure_outputs(const TensorMap &inputs,
                                 TensorMap &outputs) const {
  const auto specs = output_specs();
  if (specs.empty()) {
    return;
  }
  // Batch size is taken from the leading extent of the first input, which is
  // the convention documented in tensor.hpp.
  const std::int64_t batch =
      (inputs.size() > 0 && inputs[0].rank() > 0) ? inputs[0].dim(0) : 1;
  for (const auto &spec : specs) {
    if (!outputs.has(spec.name)) {
      outputs.add(spec.make_tensor(batch));
    }
  }
}

bool StubBackend::infer_impl(const TensorMap &inputs, TensorMap &outputs) {
  if (m_mode == Mode::NOOP) {
    // Deliberately leaves the caller's buffers untouched: useful to confirm
    // that a coupling path does not depend on the emulator writing anything.
    return true;
  }

  ensure_outputs(inputs, outputs);

  switch (m_mode) {
  case Mode::ZERO:
    for (auto &t : outputs) {
      t.zero();
    }
    break;
  case Mode::CONSTANT:
    for (auto &t : outputs) {
      fill(t, m_value);
    }
    break;
  case Mode::COPY:
  case Mode::AFFINE: {
    EMULATOR_INFER_REQUIRE(inputs.size() >= outputs.size(),
                           "Stub backend in "
                               << (m_mode == Mode::COPY ? "copy" : "affine")
                               << " mode needs at least as many inputs ("
                               << inputs.size() << ") as outputs ("
                               << outputs.size() << ").");
    const double scale = (m_mode == Mode::COPY) ? 1.0 : m_scale;
    const double offset = (m_mode == Mode::COPY) ? 0.0 : m_offset;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      affine(inputs[i], outputs[i], scale, offset);
    }
    break;
  }
  case Mode::NOOP:
    break; // handled above
  }

  return true;
}

} // namespace inference
} // namespace emulator
