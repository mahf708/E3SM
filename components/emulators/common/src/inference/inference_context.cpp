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
#include <cstring>
#endif

namespace emulator {
namespace inference {

namespace {

#ifdef EMULATOR_HAVE_MPI
/**
 * @brief Ask the kernel for a port nobody is using, then let it go.
 *
 * Binding to port 0 and reading back the assignment is the standard way to
 * do this (torch.distributed's own `find_free_port` works exactly so).  It
 * races against anything else on the node that binds between the close and
 * the model's own bind; the alternative — a fixed port — collides with the
 * neighbouring component instead, which is worse in a coupled job.  Set
 * `option.master_port` to pin it if a site needs a reserved range.
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

/// Broadcast a std::string from rank 0 over `comm`.
void broadcast_string(std::string &s, MPI_Comm comm) {
  int len = static_cast<int>(s.size());
  MPI_Bcast(&len, 1, MPI_INT, 0, comm);
  s.resize(static_cast<std::size_t>(len));
  if (len > 0) {
    MPI_Bcast(&s[0], len, MPI_CHAR, 0, comm);
  }
}
#endif // EMULATOR_HAVE_MPI

} // namespace

bool have_mpi() {
#ifdef EMULATOR_HAVE_MPI
  return true;
#else
  return false;
#endif
}

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

InferenceContext make_context(int fcomm) {
  InferenceContext context;

  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    return context; // serial: MPI linked but never started (tools, tests)
  }

  MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(fcomm));
  if (comm == MPI_COMM_NULL) {
    return context;
  }

  MPI_Comm_rank(comm, &context.rank);
  MPI_Comm_size(comm, &context.size);
  context.node_name = hostname();

  // Ranks of *this component* that share memory with us.  This is what
  // decides device affinity: LOCAL_RANK must count only our own ranks on the
  // node, never every rank of the coupled job.
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

#else // !EMULATOR_HAVE_MPI

InferenceContext make_context(int fcomm) {
  (void)fcomm;
  return InferenceContext();
}

#endif // EMULATOR_HAVE_MPI

} // namespace inference
} // namespace emulator
