// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>
#include <vector>

#include "inference_executor.hpp"

namespace emulator {
namespace inference {
namespace test {

TEST_CASE("execution policy names round-trip", "[executor]") {
  REQUIRE(execution_policy_name(ExecutionPolicy::LOCAL_REPLICA) ==
          "local_replica");
  REQUIRE(execution_policy_from_string("local_replica") ==
          ExecutionPolicy::LOCAL_REPLICA);
  REQUIRE(execution_policy_from_string("") == ExecutionPolicy::LOCAL_REPLICA);
  REQUIRE(execution_policy_from_string("Local-Replica") ==
          ExecutionPolicy::LOCAL_REPLICA);
  REQUIRE(execution_policy_from_string("gpu_group") ==
          ExecutionPolicy::GPU_GROUP);
  REQUIRE(execution_policy_from_string("spatial_distributed") ==
          ExecutionPolicy::SPATIAL_DISTRIBUTED);
  REQUIRE_THROWS_AS(execution_policy_from_string("magic"), InferenceError);
}

TEST_CASE("the default policy is one replica per rank", "[executor]") {
  InferenceConfig config;
  config.set("mode", std::string("affine")).set("scale", 3.0);
  config.outputs.push_back(TensorSpec("y", {-1, 2}, DType::FLOAT64));

  auto executor = create_executor(config, InferenceContext::serial());

  REQUIRE(executor->policy() == ExecutionPolicy::LOCAL_REPLICA);
  REQUIRE(executor->owns_model());
  REQUIRE_FALSE(executor->is_initialized());
  REQUIRE(executor->backend().name() == "Stub");

  TensorMap inputs;
  Tensor &x = inputs.emplace("x", {2, 2});
  x.flat<double>(0) = 1.0;
  TensorMap outputs;

  // infer() initializes lazily, like the backend does.
  REQUIRE(executor->infer(inputs, outputs));
  REQUIRE(executor->is_initialized());
  REQUIRE(outputs.at("y").flat<double>(0) == Approx(3.0));

  executor->finalize();
  REQUIRE_FALSE(executor->is_initialized());
}

TEST_CASE("shapes an executor sees are local", "[executor]") {
  // Under local_replica a "batch" is this rank's columns; the executor makes
  // no attempt to relate them to a global grid.
  InferenceConfig config;
  config.set("mode", std::string("copy"));
  config.outputs.push_back(TensorSpec("y", {-1, 3}, DType::FLOAT64));

  InferenceContext context = InferenceContext::serial();
  context.rank = 7; // pretend we are one rank of many
  context.size = 64;

  auto executor = create_and_init_executor(config, context);
  REQUIRE(executor->context().rank == 7);
  REQUIRE(executor->context().is_parallel());
  REQUIRE_FALSE(executor->context().is_root());

  TensorMap inputs;
  Tensor &x = inputs.emplace("x", {5, 3}); // 5 local columns
  for (std::int64_t i = 0; i < x.size(); ++i) {
    x.flat<double>(i) = static_cast<double>(i);
  }
  TensorMap outputs;
  REQUIRE(executor->infer(inputs, outputs));
  REQUIRE(outputs.at("y").dims() == std::vector<std::int64_t>{5, 3});
  REQUIRE(outputs.at("y").flat<double>(14) == Approx(14.0));
}

TEST_CASE("unimplemented policies say what they would need", "[executor]") {
  InferenceConfig config;

  config.set("execution_policy", std::string("gpu_group"));
  REQUIRE_THROWS_WITH(create_executor(config, InferenceContext::serial()),
                      Catch::Contains("not implemented") &&
                          Catch::Contains("one rank per device"));

  config.set("execution_policy", std::string("spatial_distributed"));
  REQUIRE_THROWS_WITH(create_executor(config, InferenceContext::serial()),
                      Catch::Contains("not implemented") &&
                          Catch::Contains("torch.distributed"));

  config.set("execution_policy", std::string("nonsense"));
  REQUIRE_THROWS_AS(create_executor(config, InferenceContext::serial()),
                    InferenceError);
}

TEST_CASE("only the root rank narrates", "[executor]") {
  InferenceConfig config;
  config.verbose = true;

  InferenceContext root = InferenceContext::serial();
  auto on_root = create_executor(config, root);
  REQUIRE(on_root->backend().config().verbose);

  InferenceContext worker = InferenceContext::serial();
  worker.rank = 3;
  worker.size = 8;
  auto on_worker = create_executor(config, worker);
  REQUIRE_FALSE(on_worker->backend().config().verbose);

  // ... unless a debugging session asks for every rank.
  config.set("verbose_all_ranks", true);
  auto chatty = create_executor(config, worker);
  REQUIRE(chatty->backend().config().verbose);
}

TEST_CASE("device assignment comes from the context", "[executor]") {
  InferenceConfig config;
  config.set("device", std::string("cuda"));

  SECTION("a rank alone on its node may assume device 0") {
    InferenceContext context = InferenceContext::serial();
    auto executor = create_executor(config, context);
    REQUIRE(executor->backend().config().get_int("device_id") == 0);
  }

  SECTION("the context's ordinal is used when it has one") {
    InferenceContext context = InferenceContext::serial();
    context.rank = 5;
    context.size = 8;
    context.node_rank = 1;
    context.node_size = 4;
    context.device_id = 1;
    auto executor = create_executor(config, context);
    REQUIRE(executor->backend().config().get_int("device_id") == 1);
  }

  SECTION("ranks sharing a node without an assignment are refused") {
    // Silently putting four ranks on device 0 is the failure this prevents.
    InferenceContext context = InferenceContext::serial();
    context.rank = 2;
    context.size = 8;
    context.node_rank = 2;
    context.node_size = 4;
    REQUIRE_THROWS_WITH(create_executor(config, context),
                        Catch::Contains("device 0") &&
                            Catch::Contains("device_id"));
  }

  SECTION("an explicit option always wins") {
    InferenceContext context = InferenceContext::serial();
    context.node_size = 4;
    context.node_rank = 3;
    config.set("device_id", 3);
    auto executor = create_executor(config, context);
    REQUIRE(executor->backend().config().get_int("device_id") == 3);
  }

  SECTION("cpu execution is unaffected") {
    InferenceConfig cpu_config;
    cpu_config.set("device", std::string("cpu"));
    InferenceContext context = InferenceContext::serial();
    context.node_size = 4;
    REQUIRE_NOTHROW(create_executor(cpu_config, context));
  }
}

TEST_CASE("the context reports itself", "[executor]") {
  InferenceContext context = InferenceContext::serial();
  REQUIRE(context.is_root());
  REQUIRE(context.is_node_root());
  REQUIRE_FALSE(context.is_parallel());
  REQUIRE(context.comm == k_no_comm);
  REQUIRE(context.device_id == k_no_device);
  REQUIRE(context.to_string().find("host") != std::string::npos);

  context.rank = 3;
  context.size = 16;
  context.device_id = 2;
  const std::string text = context.to_string();
  REQUIRE(text.find("3/16") != std::string::npos);
  REQUIRE(text.find("device 2") != std::string::npos);
}

} // namespace test
} // namespace inference
} // namespace emulator
