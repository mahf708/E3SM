/**
 * @file inference_context.cpp
 * @brief InferenceContext implementation.
 *
 * The only translation unit in the inference layer that includes <mpi.h>.
 */

#include "inference_context.hpp"

#include "inference_error.hpp"

#include <sstream>

#ifdef EMULATOR_HAVE_MPI
#include <mpi.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace emulator {
namespace inference {

void InferenceContext::set_grid(int nx_, int ny_, int num_global_cols_,
                                const int *gids, const double *lat_,
                                const double *lon_, int num_local_cols_) {
  EMULATOR_INFER_REQUIRE(num_local_cols_ >= 0,
                         "Negative local column count " << num_local_cols_
                                                        << ".");
  nx = nx_;
  ny = ny_;
  num_global_cols = num_global_cols_;
  const auto n = static_cast<std::size_t>(num_local_cols_);
  col_gids.assign(gids, gids + n);
  lat.assign(lat_, lat_ + n);
  lon.assign(lon_, lon_ + n);
}

std::string InferenceContext::to_string() const {
  std::ostringstream oss;
  oss << "rank " << rank << "/" << size << " (local " << local_rank << "/"
      << local_size << ") on " << (node_name.empty() ? "?" : node_name)
      << ", rendezvous " << master_addr << ":" << master_port << ", grid "
      << nx << "x" << ny << ", " << num_local_cols() << " of "
      << num_global_cols << " columns";
  return oss.str();
}

#ifdef EMULATOR_HAVE_MPI

namespace {

/**
 * @brief Ask the kernel for a port nobody is using, then let it go.
 *
 * Binding to port 0 and reading back the assignment is the standard trick
 * (torch.distributed's own `find_free_port` works exactly so).  It races
 * against anything else on the node that binds between the close and the
 * model's own bind; a fixed port would collide with the neighbouring
 * component instead, which in a coupled job is worse.  Set
 * `inference.master_port` to pin it if a site needs a reserved range.
 */
int find_free_port() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return 0;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = 0;
  int port = 0;
  if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
    socklen_t len = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) == 0) {
      port = ntohs(addr.sin_port);
    }
  }
  ::close(fd);
  return port;
}

std::string hostname() {
  char buf[256];
  if (::gethostname(buf, sizeof(buf)) != 0) {
    return "127.0.0.1";
  }
  buf[sizeof(buf) - 1] = '\0';
  return std::string(buf);
}

void broadcast_string(std::string &s, MPI_Comm comm) {
  int len = static_cast<int>(s.size());
  MPI_Bcast(&len, 1, MPI_INT, 0, comm);
  s.resize(static_cast<std::size_t>(len));
  if (len > 0) {
    MPI_Bcast(&s[0], len, MPI_CHAR, 0, comm);
  }
}

/// The live communicator behind a Fortran handle, or MPI_COMM_NULL.
MPI_Comm live_comm(int fcomm) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    return MPI_COMM_NULL; // MPI linked but never started (tools, unit tests)
  }
  return MPI_Comm_f2c(static_cast<MPI_Fint>(fcomm));
}

} // namespace

InferenceContext make_context(int fcomm) {
  InferenceContext context;
  MPI_Comm comm = live_comm(fcomm);
  if (comm == MPI_COMM_NULL) {
    return context;
  }

  MPI_Comm_rank(comm, &context.rank);
  MPI_Comm_size(comm, &context.size);
  context.node_name = hostname();

  // Ranks of *this component* that share memory with us.  This is what
  // decides device affinity: a local rank must count only our own ranks on
  // the node, never every rank of the coupled job.
  MPI_Comm node_comm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, context.rank,
                          MPI_INFO_NULL, &node_comm) == MPI_SUCCESS &&
      node_comm != MPI_COMM_NULL) {
    MPI_Comm_rank(node_comm, &context.local_rank);
    MPI_Comm_size(node_comm, &context.local_size);
    MPI_Comm_free(&node_comm);
  }

  // Rendezvous for any process group the model builds: rank 0 of the
  // *component* communicator, on a port it has just confirmed is free.
  if (context.is_root()) {
    context.master_addr = context.node_name;
    context.master_port = find_free_port();
  }
  broadcast_string(context.master_addr, comm);
  MPI_Bcast(&context.master_port, 1, MPI_INT, 0, comm);

  return context;
}

void agree_or_throw(int fcomm, const std::string &problem) {
  int any_bad = problem.empty() ? 0 : 1;
  int first_bad = 0;

  MPI_Comm comm = live_comm(fcomm);
  if (comm != MPI_COMM_NULL) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const int local_bad = problem.empty() ? 0 : 1;
    MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_MAX, comm);
    // The lowest unhappy rank, so every rank names the same one.
    const int candidate = local_bad ? rank : size;
    MPI_Allreduce(&candidate, &first_bad, 1, MPI_INT, MPI_MIN, comm);
  }

  if (any_bad == 0) {
    return;
  }
  EMULATOR_INFER_REQUIRE(problem.empty(), problem);
  EMULATOR_INFER_REQUIRE(
      false, "This rank's configuration is fine, but rank "
                 << first_bad
                 << " rejected its own. Stopping together so the run fails "
                    "instead of hanging; see that rank's message for the "
                    "cause.");
}

#else // !EMULATOR_HAVE_MPI

InferenceContext make_context(int fcomm) {
  (void)fcomm;
  return InferenceContext();
}

void agree_or_throw(int fcomm, const std::string &problem) {
  (void)fcomm;
  EMULATOR_INFER_REQUIRE(problem.empty(), problem);
}

#endif // EMULATOR_HAVE_MPI

} // namespace inference
} // namespace emulator
