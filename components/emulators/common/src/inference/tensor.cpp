/**
 * @file tensor.cpp
 * @brief Implementation of the inference data container.
 */

#include "tensor.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <new>
#include <sstream>

namespace emulator {
namespace inference {

namespace {

/// Alignment of owned allocations: 64B covers AVX-512 and common cache lines.
constexpr std::size_t k_alignment = 64;

/// Round up to a multiple of the alignment (aligned new requires this).
std::size_t round_up(std::size_t n) {
  return ((n + k_alignment - 1) / k_alignment) * k_alignment;
}

std::string dims_to_string(const std::vector<std::int64_t> &dims) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    oss << (i ? "," : "") << dims[i];
  }
  oss << "]";
  return oss.str();
}

/// Cast-copy n elements from src (type S) to dst (type D).
template <typename D, typename S>
void cast_copy(void *dst, const void *src, std::int64_t n) {
  auto *d = static_cast<D *>(dst);
  const auto *s = static_cast<const S *>(src);
  for (std::int64_t i = 0; i < n; ++i) {
    d[i] = static_cast<D>(s[i]);
  }
}

/// Dispatch cast_copy on the source type for a known destination type.
template <typename D>
void cast_copy_from(void *dst, const void *src, DType src_type,
                    std::int64_t n) {
  switch (src_type) {
  case DType::FLOAT32:
    cast_copy<D, float>(dst, src, n);
    break;
  case DType::FLOAT64:
    cast_copy<D, double>(dst, src, n);
    break;
  case DType::INT32:
    cast_copy<D, std::int32_t>(dst, src, n);
    break;
  case DType::INT64:
    cast_copy<D, std::int64_t>(dst, src, n);
    break;
  }
}

} // namespace

// ===========================================================================
// DType helpers
// ===========================================================================

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
  case DType::FLOAT32:
    return sizeof(float);
  case DType::FLOAT64:
    return sizeof(double);
  case DType::INT32:
    return sizeof(std::int32_t);
  case DType::INT64:
    return sizeof(std::int64_t);
  }
  return 0;
}

const char *dtype_name(DType dtype) {
  switch (dtype) {
  case DType::FLOAT32:
    return "float32";
  case DType::FLOAT64:
    return "float64";
  case DType::INT32:
    return "int32";
  case DType::INT64:
    return "int64";
  }
  return "unknown";
}

DType dtype_from_string(const std::string &name) {
  std::string key;
  key.reserve(name.size());
  for (char c : name) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      key.push_back(static_cast<char>(
          std::tolower(static_cast<unsigned char>(c))));
    }
  }

  if (key == "float32" || key == "float" || key == "f32" || key == "single" ||
      key == "real4" || key == "r4") {
    return DType::FLOAT32;
  }
  if (key == "float64" || key == "double" || key == "f64" || key == "real8" ||
      key == "r8") {
    return DType::FLOAT64;
  }
  if (key == "int32" || key == "int" || key == "i32" || key == "integer") {
    return DType::INT32;
  }
  if (key == "int64" || key == "long" || key == "i64") {
    return DType::INT64;
  }

  EMULATOR_INFER_REQUIRE(false, "Unknown data type '"
                                    << name
                                    << "'. Valid names: float32, float64, "
                                       "int32, int64 (aliases: float, double, "
                                       "real4, real8, int, long).");
  return DType::FLOAT64; // unreachable
}

// ===========================================================================
// Tensor
// ===========================================================================

std::int64_t Tensor::compute_size(const std::vector<std::int64_t> &dims) {
  std::int64_t n = 1;
  for (std::int64_t d : dims) {
    EMULATOR_INFER_REQUIRE(d >= 0, "Tensor dimensions must be non-negative "
                                   "(got "
                                       << dims_to_string(dims)
                                       << "). Use TensorSpec for dynamic (-1) "
                                          "extents.");
    n *= d;
  }
  return n;
}

Tensor::Tensor(std::string name, std::vector<std::int64_t> dims, DType dtype)
    : m_name(std::move(name)), m_dims(std::move(dims)), m_dtype(dtype) {
  m_size = compute_size(m_dims);
  allocate(static_cast<std::size_t>(m_size) * dtype_size(m_dtype));
  m_owned = true;
  m_writable = true;
  zero();
}

Tensor Tensor::wrap(std::string name, void *data,
                    std::vector<std::int64_t> dims, DType dtype) {
  Tensor t;
  t.m_name = std::move(name);
  t.m_dims = std::move(dims);
  t.m_dtype = dtype;
  t.m_size = compute_size(t.m_dims);
  t.m_data = data;
  t.m_owned = false;
  t.m_writable = true;
  EMULATOR_INFER_REQUIRE(data != nullptr || t.m_size == 0,
                         "Tensor::wrap('" << t.m_name
                                          << "') got a null pointer for "
                                          << t.m_size << " elements.");
  return t;
}

Tensor Tensor::wrap(std::string name, const void *data,
                    std::vector<std::int64_t> dims, DType dtype) {
  Tensor t = wrap(std::move(name), const_cast<void *>(data), std::move(dims),
                  dtype);
  t.m_writable = false;
  return t;
}

Tensor::~Tensor() { release(); }

Tensor::Tensor(Tensor &&other) noexcept
    : m_name(std::move(other.m_name)), m_dims(std::move(other.m_dims)),
      m_dtype(other.m_dtype), m_data(other.m_data), m_size(other.m_size),
      m_capacity(other.m_capacity), m_owned(other.m_owned),
      m_writable(other.m_writable) {
  other.m_data = nullptr;
  other.m_size = 0;
  other.m_capacity = 0;
  other.m_owned = false;
  other.m_writable = false;
  other.m_dims.clear();
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    release();
    m_name = std::move(other.m_name);
    m_dims = std::move(other.m_dims);
    m_dtype = other.m_dtype;
    m_data = other.m_data;
    m_size = other.m_size;
    m_capacity = other.m_capacity;
    m_owned = other.m_owned;
    m_writable = other.m_writable;

    other.m_data = nullptr;
    other.m_size = 0;
    other.m_capacity = 0;
    other.m_owned = false;
    other.m_writable = false;
    other.m_dims.clear();
  }
  return *this;
}

void Tensor::allocate(std::size_t nbytes) {
  release();
  if (nbytes == 0) {
    m_capacity = 0;
    m_data = nullptr;
    return;
  }
  const std::size_t padded = round_up(nbytes);
  m_data = ::operator new(padded, std::align_val_t(k_alignment));
  m_capacity = padded;
}

void Tensor::release() {
  if (m_owned && m_data != nullptr) {
    ::operator delete(m_data, m_capacity, std::align_val_t(k_alignment));
  }
  m_data = nullptr;
  m_capacity = 0;
  m_owned = false;
}

Tensor Tensor::clone() const {
  Tensor t(m_name, m_dims, m_dtype);
  if (m_data != nullptr && m_size > 0) {
    std::memcpy(t.m_data, m_data, nbytes());
  }
  return t;
}

std::int64_t Tensor::dim(int i) const {
  EMULATOR_INFER_REQUIRE(i >= 0 && i < rank(),
                         "Tensor '" << m_name << "': dimension index " << i
                                    << " out of range for rank " << rank()
                                    << ".");
  return m_dims[static_cast<std::size_t>(i)];
}

void *Tensor::data() {
  EMULATOR_INFER_REQUIRE(m_data != nullptr,
                         "Tensor '" << m_name << "' has no data.");
  EMULATOR_INFER_REQUIRE(m_writable, "Tensor '"
                                         << m_name
                                         << "' is a read-only view; cannot "
                                            "get a writable pointer.");
  return m_data;
}

const void *Tensor::data() const {
  EMULATOR_INFER_REQUIRE(m_data != nullptr,
                         "Tensor '" << m_name << "' has no data.");
  return m_data;
}

void Tensor::check_dtype(DType expected) const {
  EMULATOR_INFER_REQUIRE(expected == m_dtype,
                         "Tensor '" << m_name << "' holds "
                                    << dtype_name(m_dtype)
                                    << " but was accessed as "
                                    << dtype_name(expected) << ".");
}

void Tensor::check_index(std::int64_t i) const {
  EMULATOR_INFER_REQUIRE(i >= 0 && i < m_size,
                         "Tensor '" << m_name << "': flat index " << i
                                    << " out of range for size " << m_size
                                    << ".");
}

void Tensor::reshape(const std::vector<std::int64_t> &dims) {
  const std::int64_t n = compute_size(dims);
  EMULATOR_INFER_REQUIRE(n == m_size,
                         "Tensor '" << m_name << "': cannot reshape "
                                    << dims_to_string(m_dims) << " (" << m_size
                                    << " elements) to "
                                    << dims_to_string(dims) << " (" << n
                                    << " elements).");
  m_dims = dims;
}

void Tensor::resize(const std::vector<std::int64_t> &dims) {
  resize(dims, m_dtype);
}

void Tensor::resize(const std::vector<std::int64_t> &dims, DType dtype) {
  EMULATOR_INFER_REQUIRE(m_owned || m_data == nullptr,
                         "Tensor '" << m_name
                                    << "' is a view and cannot be resized; "
                                       "use reshape() or set_batch_size().");
  const std::int64_t n = compute_size(dims);
  const std::size_t needed = static_cast<std::size_t>(n) * dtype_size(dtype);

  if (needed > m_capacity || (m_data == nullptr && needed > 0)) {
    allocate(needed); // drops old contents; documented behavior
    m_owned = true;
    m_writable = true;
  }
  m_dims = dims;
  m_dtype = dtype;
  m_size = n;
}

void Tensor::set_batch_size(std::int64_t batch) {
  EMULATOR_INFER_REQUIRE(rank() > 0, "Tensor '"
                                         << m_name
                                         << "' has rank 0; cannot set a batch "
                                            "size.");
  EMULATOR_INFER_REQUIRE(batch >= 0, "Tensor '" << m_name
                                                << "': negative batch size "
                                                << batch << ".");
  std::vector<std::int64_t> dims = m_dims;
  dims[0] = batch;
  if (m_owned || m_data == nullptr) {
    resize(dims, m_dtype);
  } else {
    // View: only the logical shape changes.  Growing a view is the caller's
    // responsibility (they own the memory and know how big it is).
    m_dims = dims;
    m_size = compute_size(dims);
  }
}

void Tensor::zero() {
  if (m_data == nullptr || m_size == 0) {
    return;
  }
  EMULATOR_INFER_REQUIRE(m_writable, "Tensor '" << m_name
                                                << "' is read-only; cannot "
                                                   "zero it.");
  std::memset(m_data, 0, nbytes());
}

void Tensor::copy_from(const Tensor &src) {
  EMULATOR_INFER_REQUIRE(m_writable && m_data != nullptr,
                         "Tensor '" << m_name
                                    << "' is not a writable destination.");
  EMULATOR_INFER_REQUIRE(src.m_data != nullptr,
                         "Tensor '" << src.m_name
                                    << "' has no data to copy from.");
  EMULATOR_INFER_REQUIRE(src.m_size == m_size,
                         "Cannot copy tensor '"
                             << src.m_name << "' "
                             << dims_to_string(src.m_dims) << " into '"
                             << m_name << "' " << dims_to_string(m_dims)
                             << ": element counts differ (" << src.m_size
                             << " vs " << m_size << ").");
  if (m_size == 0) {
    return;
  }

  if (src.m_dtype == m_dtype) {
    std::memcpy(m_data, src.m_data, nbytes());
    return;
  }

  switch (m_dtype) {
  case DType::FLOAT32:
    cast_copy_from<float>(m_data, src.m_data, src.m_dtype, m_size);
    break;
  case DType::FLOAT64:
    cast_copy_from<double>(m_data, src.m_data, src.m_dtype, m_size);
    break;
  case DType::INT32:
    cast_copy_from<std::int32_t>(m_data, src.m_data, src.m_dtype, m_size);
    break;
  case DType::INT64:
    cast_copy_from<std::int64_t>(m_data, src.m_data, src.m_dtype, m_size);
    break;
  }
}

std::string Tensor::to_string() const {
  std::ostringstream oss;
  oss << m_name << dims_to_string(m_dims) << ":" << dtype_name(m_dtype);
  if (m_data == nullptr) {
    oss << " (empty)";
  } else if (m_owned) {
    oss << " (owned)";
  } else {
    oss << (m_writable ? " (view)" : " (const view)");
  }
  return oss.str();
}

// ===========================================================================
// TensorSpec
// ===========================================================================

TensorSpec TensorSpec::parse(const std::string &text) {
  auto trim = [](std::string s) {
    const char *ws = " \t\r\n";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) {
      return std::string();
    }
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
  };

  std::string body = trim(text);
  EMULATOR_INFER_REQUIRE(!body.empty(), "Empty tensor spec.");

  TensorSpec spec;

  // Optional trailing ":dtype" (after any bracketed shape).
  const auto bracket_end = body.rfind(']');
  const auto colon = body.rfind(':');
  if (colon != std::string::npos &&
      (bracket_end == std::string::npos || colon > bracket_end)) {
    spec.dtype = dtype_from_string(body.substr(colon + 1));
    body = trim(body.substr(0, colon));
  }

  // Optional "[d0,d1,...]" shape.
  const auto lb = body.find('[');
  if (lb != std::string::npos) {
    const auto rb = body.find(']', lb);
    EMULATOR_INFER_REQUIRE(rb != std::string::npos,
                           "Malformed tensor spec '" << text
                                                     << "': missing ']'.");
    std::string shape = body.substr(lb + 1, rb - lb - 1);
    body = trim(body.substr(0, lb));

    std::istringstream iss(shape);
    std::string token;
    while (std::getline(iss, token, ',')) {
      token = trim(token);
      if (token.empty()) {
        continue;
      }
      try {
        spec.dims.push_back(static_cast<std::int64_t>(std::stoll(token)));
      } catch (const std::exception &) {
        EMULATOR_INFER_REQUIRE(false, "Malformed tensor spec '"
                                          << text << "': bad extent '" << token
                                          << "'.");
      }
      EMULATOR_INFER_REQUIRE(spec.dims.back() >= -1,
                             "Malformed tensor spec '"
                                 << text << "': extent " << spec.dims.back()
                                 << " (use -1 for a dynamic extent).");
    }
  }

  spec.name = trim(body);
  EMULATOR_INFER_REQUIRE(!spec.name.empty(),
                         "Malformed tensor spec '" << text
                                                   << "': missing name.");
  return spec;
}

std::string TensorSpec::to_string() const {
  std::ostringstream oss;
  oss << name;
  if (!dims.empty()) {
    oss << dims_to_string(dims);
  }
  oss << ":" << dtype_name(dtype);
  return oss.str();
}

std::vector<std::int64_t>
TensorSpec::dims_with_batch(std::int64_t batch) const {
  std::vector<std::int64_t> out = dims;
  for (auto &d : out) {
    if (d < 0) {
      d = batch;
    }
  }
  return out;
}

std::int64_t TensorSpec::size_with_batch(std::int64_t batch) const {
  std::int64_t n = 1;
  for (std::int64_t d : dims_with_batch(batch)) {
    n *= d;
  }
  return n;
}

bool TensorSpec::is_static() const {
  return std::none_of(dims.begin(), dims.end(),
                      [](std::int64_t d) { return d < 0; });
}

Tensor TensorSpec::make_tensor(std::int64_t batch) const {
  return Tensor(name, dims_with_batch(batch), dtype);
}

bool spec_matches(const TensorSpec &spec, const Tensor &tensor,
                  std::string *why) {
  if (spec.dtype != tensor.dtype()) {
    if (why) {
      *why = "expected " + std::string(dtype_name(spec.dtype)) + ", got " +
             dtype_name(tensor.dtype());
    }
    return false;
  }
  if (spec.dims.empty()) {
    return true; // shape unconstrained
  }
  if (spec.dims.size() != tensor.dims().size()) {
    if (why) {
      *why = "expected rank " + std::to_string(spec.dims.size()) + ", got " +
             std::to_string(tensor.dims().size());
    }
    return false;
  }
  for (std::size_t i = 0; i < spec.dims.size(); ++i) {
    if (spec.dims[i] < 0) {
      continue; // dynamic
    }
    if (spec.dims[i] != tensor.dims()[i]) {
      if (why) {
        *why = "expected " + dims_to_string(spec.dims) + ", got " +
               dims_to_string(tensor.dims());
      }
      return false;
    }
  }
  return true;
}

// ===========================================================================
// TensorMap
// ===========================================================================

Tensor &TensorMap::add(Tensor tensor) {
  EMULATOR_INFER_REQUIRE(!tensor.name().empty(),
                         "Cannot add an unnamed tensor to a TensorMap.");
  EMULATOR_INFER_REQUIRE(!has(tensor.name()),
                         "TensorMap already contains a tensor named '"
                             << tensor.name() << "'.");
  m_tensors.push_back(std::move(tensor));
  return m_tensors.back();
}

Tensor &TensorMap::emplace(const std::string &name,
                           std::vector<std::int64_t> dims, DType dtype) {
  return add(Tensor(name, std::move(dims), dtype));
}

Tensor &TensorMap::wrap(const std::string &name, void *data,
                        std::vector<std::int64_t> dims, DType dtype) {
  return add(Tensor::wrap(name, data, std::move(dims), dtype));
}

Tensor &TensorMap::wrap(const std::string &name, const void *data,
                        std::vector<std::int64_t> dims, DType dtype) {
  return add(Tensor::wrap(name, data, std::move(dims), dtype));
}

bool TensorMap::has(const std::string &name) const {
  return find(name) != nullptr;
}

Tensor *TensorMap::find(const std::string &name) {
  for (auto &t : m_tensors) {
    if (t.name() == name) {
      return &t;
    }
  }
  return nullptr;
}

const Tensor *TensorMap::find(const std::string &name) const {
  for (const auto &t : m_tensors) {
    if (t.name() == name) {
      return &t;
    }
  }
  return nullptr;
}

Tensor &TensorMap::at(const std::string &name) {
  Tensor *t = find(name);
  EMULATOR_INFER_REQUIRE(t != nullptr, "No tensor named '"
                                           << name
                                           << "' in TensorMap. Available: "
                                           << names_string() << ".");
  return *t;
}

const Tensor &TensorMap::at(const std::string &name) const {
  const Tensor *t = find(name);
  EMULATOR_INFER_REQUIRE(t != nullptr, "No tensor named '"
                                           << name
                                           << "' in TensorMap. Available: "
                                           << names_string() << ".");
  return *t;
}

Tensor &TensorMap::operator[](std::size_t i) {
  EMULATOR_INFER_REQUIRE(i < m_tensors.size(),
                         "TensorMap index " << i << " out of range (size "
                                            << m_tensors.size() << ").");
  return m_tensors[i];
}

const Tensor &TensorMap::operator[](std::size_t i) const {
  EMULATOR_INFER_REQUIRE(i < m_tensors.size(),
                         "TensorMap index " << i << " out of range (size "
                                            << m_tensors.size() << ").");
  return m_tensors[i];
}

bool TensorMap::erase(const std::string &name) {
  for (auto it = m_tensors.begin(); it != m_tensors.end(); ++it) {
    if (it->name() == name) {
      m_tensors.erase(it);
      return true;
    }
  }
  return false;
}

std::vector<std::string> TensorMap::names() const {
  std::vector<std::string> out;
  out.reserve(m_tensors.size());
  for (const auto &t : m_tensors) {
    out.push_back(t.name());
  }
  return out;
}

std::string TensorMap::names_string() const {
  std::ostringstream oss;
  for (std::size_t i = 0; i < m_tensors.size(); ++i) {
    oss << (i ? ", " : "") << m_tensors[i].name();
  }
  const std::string s = oss.str();
  return s.empty() ? "<none>" : s;
}

TensorMap make_tensors(const std::vector<TensorSpec> &specs,
                       std::int64_t batch) {
  TensorMap map;
  for (const auto &spec : specs) {
    map.add(spec.make_tensor(batch));
  }
  return map;
}

} // namespace inference
} // namespace emulator
