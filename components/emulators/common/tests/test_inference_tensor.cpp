// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "inference_error.hpp"
#include "tensor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace emulator {
namespace inference {
namespace test {

TEST_CASE("Owning tensor allocates and zero-fills", "[tensor]") {
  Tensor t("dT", {4, 3});

  REQUIRE(t.name() == "dT");
  REQUIRE(t.dims() == std::vector<std::int64_t>{4, 3});
  REQUIRE(t.size() == 12);
  REQUIRE(t.writable());
  REQUIRE(t.cdata()[11] == 0.0);
  REQUIRE(t.to_string() == "dT[4,3]");
}

TEST_CASE("Owning tensor survives a move", "[tensor]") {
  Tensor a("dT", {2});
  a.data()[1] = 7.0;

  Tensor b = std::move(a);
  REQUIRE(b.cdata()[1] == 7.0);
}

TEST_CASE("View shares the caller's memory", "[tensor]") {
  std::vector<double> field(6, 1.5);
  Tensor t = Tensor::view("T", field.data(), {2, 3});

  REQUIRE(t.cdata() == field.data());

  t.data()[0] = 42.0;
  REQUIRE(field[0] == 42.0);
}

TEST_CASE("Const view cannot be written through", "[tensor]") {
  const std::vector<double> field(6, 1.5);
  Tensor t = Tensor::const_view("T", field.data(), {2, 3});

  REQUIRE_FALSE(t.writable());
  REQUIRE(t.cdata()[3] == 1.5);
  REQUIRE_THROWS_AS(t.data(), InferenceError);
}

TEST_CASE("Views are contiguous and on the host by default", "[tensor]") {
  std::vector<double> field(24, 0.0);
  Tensor t = Tensor::view("T", field.data(), {2, 3, 4});

  REQUIRE(t.strides() == std::vector<std::int64_t>{12, 4, 1});
  REQUIRE(t.contiguous());
  REQUIRE(t.span() == 24);
  REQUIRE_FALSE(t.on_device());

  Tensor owned("dT", {2, 3});
  REQUIRE(owned.strides() == std::vector<std::int64_t>{3, 1});
  REQUIRE(owned.contiguous());
}

TEST_CASE("Views can be strided", "[tensor]") {
  // (2, 3) values in a (2, 4) buffer: rows padded by one element
  std::vector<double> padded(8, 0.0);
  Tensor t = Tensor::view("T", padded.data(), {2, 3}, {4, 1});

  REQUIRE(t.size() == 6);
  REQUIRE_FALSE(t.contiguous());
  REQUIRE(t.span() == 7);
  REQUIRE(t.strides() == std::vector<std::int64_t>{4, 1});
}

TEST_CASE("Views can be in device memory", "[tensor]") {
  std::vector<double> field(6, 0.0);
  TensorMap m;
  m.wrap("T", static_cast<const double *>(field.data()), {2, 3}, {},
         TensorMemory{MemorySpace::DEVICE, 1});

  const Tensor &t = *m.begin();
  REQUIRE(t.on_device());
  REQUIRE(t.memory().device == 1);
  REQUIRE_FALSE(t.writable());
}

TEST_CASE("Invalid strides are refused", "[tensor]") {
  std::vector<double> field(6, 0.0);

  REQUIRE_THROWS_AS(Tensor::view("T", field.data(), {2, 3}, {3}), InferenceError);
  REQUIRE_THROWS_AS(Tensor::view("T", field.data(), {2, 3}, {-3, 1}), InferenceError);
}

TEST_CASE("Invalid views are refused", "[tensor]") {
  std::vector<double> field(6, 0.0);

  REQUIRE_THROWS_AS(Tensor::view("T", nullptr, {2, 3}), InferenceError);
  REQUIRE_THROWS_AS(Tensor::view("T", field.data(), {-2, 3}), InferenceError);
}

TEST_CASE("Empty tensors are legal", "[tensor]") {
  // e.g. an MPI rank that owns no columns
  Tensor owned("dT", {0, 3});
  REQUIRE(owned.size() == 0);
  REQUIRE(owned.writable());

  Tensor v = Tensor::view("dT", nullptr, {0, 3});
  REQUIRE(v.writable());
  REQUIRE(v.data() == nullptr);

  REQUIRE_FALSE(Tensor::const_view("T", nullptr, {0, 3}).writable());
}

TEST_CASE("TensorMap preserves order and rejects duplicates", "[tensor]") {
  const std::vector<double> a(4, 1.0);
  std::vector<double> b(4, 2.0);

  TensorMap map;
  map.wrap("T", a.data(), {4});
  map.wrap("q", b.data(), {4});
  map.add(Tensor("p", {4}));

  std::vector<std::string> names;
  for (const auto &t : map) {
    names.push_back(t.name());
  }
  REQUIRE(names == std::vector<std::string>{"T", "q", "p"});

  // const and non-const pointers select the matching view
  REQUIRE_FALSE(map.begin()->writable());
  REQUIRE((map.begin() + 1)->writable());

  REQUIRE_THROWS_AS(map.wrap("T", b.data(), {4}), InferenceError);
  REQUIRE(map.size() == 3);
}

} // namespace test
} // namespace inference
} // namespace emulator
