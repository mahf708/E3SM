/**
 * @file inference_context.cpp
 * @brief InferenceContext reporting (MPI-free).
 */

#include "inference_context.hpp"

#include <sstream>

namespace emulator {
namespace inference {

std::string InferenceContext::to_string() const {
  std::ostringstream oss;
  oss << "rank " << rank << "/" << size << " (node " << node_rank << "/"
      << node_size << ")";
  if (device_id != k_no_device) {
    oss << " device " << device_id;
  } else {
    oss << " host";
  }
  if (comm == k_no_comm) {
    oss << " [no communicator]";
  }
  return oss.str();
}

} // namespace inference
} // namespace emulator
