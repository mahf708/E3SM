// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>

#include "tensor.hpp"

namespace emulator {
namespace inference {
namespace test {

TEST_CASE("dtype helpers", "[tensor]") {
  REQUIRE(dtype_size(DType::FLOAT32) == 4);
  REQUIRE(dtype_size(DType::FLOAT64) == 8);
  REQUIRE(dtype_size(DType::INT32) == 4);
  REQUIRE(dtype_size(DType::INT64) == 8);

  REQUIRE(std::string(dtype_name(DType::FLOAT32)) == "float32");
  REQUIRE(std::string(dtype_name(DType::FLOAT64)) == "float64");

  // Canonical names and the aliases E3SM developers are likely to type.
  REQUIRE(dtype_from_string("float32") == DType::FLOAT32);
  REQUIRE(dtype_from_string("float") == DType::FLOAT32);
  REQUIRE(dtype_from_string("real4") == DType::FLOAT32);
  REQUIRE(dtype_from_string("r8") == DType::FLOAT64);
  REQUIRE(dtype_from_string("DOUBLE") == DType::FLOAT64);
  REQUIRE(dtype_from_string("int64") == DType::INT64);
  REQUIRE_THROWS_AS(dtype_from_string("complex128"), InferenceError);
}

TEST_CASE("owning tensor allocates zeroed storage", "[tensor]") {
  Tensor t("state", {3, 4}, DType::FLOAT32);

  REQUIRE(t.name() == "state");
  REQUIRE(t.rank() == 2);
  REQUIRE(t.dim(0) == 3);
  REQUIRE(t.dim(1) == 4);
  REQUIRE(t.size() == 12);
  REQUIRE(t.nbytes() == 48);
  REQUIRE(t.owns_data());
  REQUIRE_FALSE(t.is_view());
  REQUIRE(t.writable());
  REQUIRE_FALSE(t.empty());

  for (std::int64_t i = 0; i < t.size(); ++i) {
    REQUIRE(t.flat<float>(i) == 0.0f);
  }

  REQUIRE_THROWS_AS(t.dim(2), InferenceError);
  REQUIRE_THROWS_AS(t.flat<float>(12), InferenceError);
  // Wrong-type access is caught, not reinterpreted.
  REQUIRE_THROWS_AS(t.data<double>(), InferenceError);
}

TEST_CASE("default-constructed tensor is empty", "[tensor]") {
  Tensor t;
  REQUIRE(t.empty());
  REQUIRE(t.size() == 0);
  REQUIRE(t.rank() == 0);
  REQUIRE_FALSE(t.owns_data());
  REQUIRE_THROWS_AS(t.data(), InferenceError);
}

TEST_CASE("tensor views external memory without copying", "[tensor]") {
  std::vector<double> field(6, 2.5);

  Tensor view = Tensor::wrap("T", field.data(), {2, 3});
  REQUIRE(view.is_view());
  REQUIRE_FALSE(view.owns_data());
  REQUIRE(view.writable());
  REQUIRE(view.dtype() == DType::FLOAT64);
  REQUIRE(view.data<double>() == field.data());

  // Writes through the view land in the caller's memory.
  view.flat<double>(0) = 42.0;
  REQUIRE(field[0] == 42.0);

  // Views over const memory are read-only.
  const double *const_ptr = field.data();
  Tensor ro = Tensor::wrap("T_ro", const_ptr, {6});
  REQUIRE_FALSE(ro.writable());
  REQUIRE_THROWS_AS(ro.zero(), InferenceError);

  // Asking for write access fails; cdata()/cflat() read without asking.
  REQUIRE_THROWS_AS(ro.data(), InferenceError);
  REQUIRE_THROWS_AS(ro.data<double>(), InferenceError);
  REQUIRE(ro.cdata<double>()[0] == 42.0);
  REQUIRE(ro.cflat<double>(0) == 42.0);

  const Tensor &ro_const = ro;
  REQUIRE(ro_const.data<double>()[0] == 42.0);
}

TEST_CASE("tensors move and clone explicitly", "[tensor]") {
  Tensor a("a", {4}, DType::FLOAT64);
  a.flat<double>(2) = 7.0;

  Tensor b = std::move(a);
  REQUIRE(b.size() == 4);
  REQUIRE(b.flat<double>(2) == 7.0);
  REQUIRE(a.empty()); // moved-from tensor is left in the empty state

  Tensor c = b.clone();
  REQUIRE(c.owns_data());
  REQUIRE(c.flat<double>(2) == 7.0);
  c.flat<double>(2) = 9.0;
  REQUIRE(b.flat<double>(2) == 7.0); // deep copy, not aliased

  // Cloning a view produces owned storage.
  std::vector<float> raw{1.f, 2.f, 3.f};
  Tensor v = Tensor::wrap("v", raw.data(), {3});
  Tensor v_clone = v.clone();
  REQUIRE(v_clone.owns_data());
  raw[0] = 100.f;
  REQUIRE(v_clone.flat<float>(0) == 1.0f);
}

TEST_CASE("reshape keeps memory, resize can grow", "[tensor]") {
  Tensor t("t", {2, 6}, DType::FLOAT32);
  const void *before = t.data();

  t.reshape({3, 4});
  REQUIRE(t.dims() == std::vector<std::int64_t>{3, 4});
  REQUIRE(t.data() == before); // no reallocation
  REQUIRE_THROWS_AS(t.reshape({5, 5}), InferenceError);

  // Shrinking reuses the allocation (no per-step malloc traffic).
  t.resize({2, 4});
  REQUIRE(t.size() == 8);
  REQUIRE(t.data() == before);

  t.resize({100, 4});
  REQUIRE(t.size() == 400);

  // Resizing can also change the element type.
  t.resize({4}, DType::INT64);
  REQUIRE(t.dtype() == DType::INT64);
  REQUIRE(t.size() == 4);

  // Views cannot be resized, but their logical shape can shrink.
  std::vector<double> raw(10, 1.0);
  Tensor v = Tensor::wrap("v", raw.data(), {10, 1});
  REQUIRE_THROWS_AS(v.resize({20, 1}), InferenceError);
  v.set_batch_size(4);
  REQUIRE(v.size() == 4);
  REQUIRE(v.data<double>() == raw.data());
}

TEST_CASE("set_batch_size resizes owning tensors", "[tensor]") {
  Tensor t("t", {1, 3}, DType::FLOAT64);
  t.set_batch_size(5);
  REQUIRE(t.dims() == std::vector<std::int64_t>{5, 3});
  REQUIRE(t.size() == 15);

  Tensor scalarish;
  REQUIRE_THROWS_AS(scalarish.set_batch_size(2), InferenceError);
}

TEST_CASE("copy_from converts between element types", "[tensor]") {
  Tensor src("src", {4}, DType::FLOAT64);
  for (std::int64_t i = 0; i < 4; ++i) {
    src.flat<double>(i) = 1.5 * static_cast<double>(i + 1);
  }

  Tensor dst("dst", {4}, DType::FLOAT32);
  dst.copy_from(src);
  REQUIRE(dst.flat<float>(0) == 1.5f);
  REQUIRE(dst.flat<float>(3) == 6.0f);

  // Same shape in a different arrangement is fine: only the count matters.
  Tensor dst2("dst2", {2, 2}, DType::FLOAT32);
  dst2.copy_from(src);
  REQUIRE(dst2.flat<float>(3) == 6.0f);

  Tensor wrong("wrong", {3}, DType::FLOAT32);
  REQUIRE_THROWS_AS(wrong.copy_from(src), InferenceError);

  // Truncation toward zero when converting to an integer type.
  Tensor ints("ints", {4}, DType::INT32);
  ints.copy_from(src);
  REQUIRE(ints.flat<std::int32_t>(0) == 1);
  REQUIRE(ints.flat<std::int32_t>(1) == 3);

  // Host-array shorthands go through the same conversion path.
  const float host_in[4] = {1.f, 2.f, 3.f, 4.f};
  Tensor from_host("fh", {4}, DType::FLOAT64);
  from_host.copy_from_host(host_in, 4);
  REQUIRE(from_host.flat<double>(3) == 4.0);

  double host_out[4] = {0, 0, 0, 0};
  from_host.copy_to_host(host_out, 4);
  REQUIRE(host_out[2] == 3.0);
}

TEST_CASE("negative dims are rejected", "[tensor]") {
  REQUIRE_THROWS_AS(Tensor("bad", {-1, 4}), InferenceError);
}

TEST_CASE("TensorSpec parses the compact form", "[tensor][spec]") {
  const auto full = TensorSpec::parse("T[-1,72]:float32");
  REQUIRE(full.name == "T");
  REQUIRE(full.dims == std::vector<std::int64_t>{-1, 72});
  REQUIRE(full.dtype == DType::FLOAT32);
  REQUIRE_FALSE(full.is_static());
  REQUIRE(full.to_string() == "T[-1,72]:float32");

  const auto no_dtype = TensorSpec::parse("q[4, 8]");
  REQUIRE(no_dtype.dims == std::vector<std::int64_t>{4, 8});
  REQUIRE(no_dtype.dtype == DType::FLOAT64); // E3SM default precision
  REQUIRE(no_dtype.is_static());

  const auto bare = TensorSpec::parse("  ps  ");
  REQUIRE(bare.name == "ps");
  REQUIRE(bare.dims.empty());

  REQUIRE(full.dims_with_batch(10) == std::vector<std::int64_t>{10, 72});
  REQUIRE(full.size_with_batch(10) == 720);

  REQUIRE_THROWS_AS(TensorSpec::parse("bad[1,2"), InferenceError);
  REQUIRE_THROWS_AS(TensorSpec::parse("bad[a]"), InferenceError);
  REQUIRE_THROWS_AS(TensorSpec::parse("[4]"), InferenceError);
  REQUIRE_THROWS_AS(TensorSpec::parse(""), InferenceError);
}

TEST_CASE("spec_matches reports why a tensor does not fit", "[tensor][spec]") {
  const TensorSpec spec("T", {-1, 4}, DType::FLOAT32);

  Tensor good("T", {8, 4}, DType::FLOAT32);
  REQUIRE(spec_matches(spec, good));

  std::string why;
  Tensor bad_type("T", {8, 4}, DType::FLOAT64);
  REQUIRE_FALSE(spec_matches(spec, bad_type, &why));
  REQUIRE(why.find("float32") != std::string::npos);

  Tensor bad_rank("T", {8}, DType::FLOAT32);
  REQUIRE_FALSE(spec_matches(spec, bad_rank, &why));
  REQUIRE(why.find("rank") != std::string::npos);

  Tensor bad_extent("T", {8, 5}, DType::FLOAT32);
  REQUIRE_FALSE(spec_matches(spec, bad_extent, &why));

  // An empty spec shape constrains only the element type.
  const TensorSpec loose("T", {}, DType::FLOAT32);
  REQUIRE(spec_matches(loose, bad_extent));
}

TEST_CASE("TensorMap is ordered and name-addressable", "[tensor][map]") {
  TensorMap map;
  map.emplace("a", {2, 2}, DType::FLOAT32);
  std::vector<double> raw(4, 3.0);
  map.wrap("b", raw.data(), {4});

  REQUIRE(map.size() == 2);
  REQUIRE(map.has("a"));
  REQUIRE(map.has("b"));
  REQUIRE_FALSE(map.has("c"));

  // Insertion order is preserved for positional backends.
  REQUIRE(map[0].name() == "a");
  REQUIRE(map[1].name() == "b");
  REQUIRE(map.names() == std::vector<std::string>{"a", "b"});

  REQUIRE(map.at("b").flat<double>(0) == 3.0);
  REQUIRE(map.find("nope") == nullptr);
  REQUIRE_THROWS_AS(map.at("nope"), InferenceError);
  REQUIRE_THROWS_AS(map[2], InferenceError);

  // Duplicate and unnamed tensors are rejected.
  REQUIRE_THROWS_AS(map.emplace("a", {1}), InferenceError);
  REQUIRE_THROWS_AS(map.add(Tensor("", {1})), InferenceError);

  REQUIRE(map.erase("a"));
  REQUIRE_FALSE(map.erase("a"));
  REQUIRE(map.size() == 1);

  map.clear();
  REQUIRE(map.empty());
}

TEST_CASE("TensorMap references survive later insertions", "[tensor][map]") {
  // A component builds up a map field by field and may keep hold of earlier
  // tensors while doing so.
  TensorMap map;
  Tensor &first = map.emplace("first", {4});
  first.flat<double>(0) = 1.0;

  for (int i = 0; i < 64; ++i) {
    map.emplace("filler_" + std::to_string(i), {4});
  }

  REQUIRE(first.flat<double>(0) == 1.0);  // reference is still good
  REQUIRE(&first == &map.at("first"));
  first.flat<double>(0) = 2.0;
  REQUIRE(map.at("first").flat<double>(0) == 2.0);
}

TEST_CASE("make_tensors builds a map from specs", "[tensor][map]") {
  const std::vector<TensorSpec> specs{TensorSpec("T", {-1, 3}, DType::FLOAT32),
                                      TensorSpec("ps", {-1}, DType::FLOAT64)};

  TensorMap map = make_tensors(specs, 5);
  REQUIRE(map.size() == 2);
  REQUIRE(map.at("T").dims() == std::vector<std::int64_t>{5, 3});
  REQUIRE(map.at("T").dtype() == DType::FLOAT32);
  REQUIRE(map.at("ps").dims() == std::vector<std::int64_t>{5});
  REQUIRE(map.at("ps").owns_data());
}

} // namespace test
} // namespace inference
} // namespace emulator
