/**
 * @file emulator_test_support.hpp
 * @brief What the component tests share: stand-in coupler attribute
 *        vectors, input files written on the fly, the real-data paths, and
 *        an MPI main.
 */

#ifndef E3SM_EMULATOR_TEST_SUPPORT_HPP
#define E3SM_EMULATOR_TEST_SUPPORT_HPP

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "emulator_c_api.hpp"

namespace emulator {
namespace test {

inline const std::string kGaussianGrid =
    "/pscratch/sd/m/mahf708/eocn-inputdata/share/meshes/"
    "gaussian_180x360_latlon_grnwst.scrip.20260913.nc";
inline const std::string kSamudrace = "/pscratch/sd/m/mahf708/SamudrACE-E3SMv3/";
inline const std::string kSamudraceAtmModel =
    kSamudrace + "eatm/samudrace_atm_traced_cuda.pt";
inline const std::string kSamudraceAtmIc = kSamudrace + "eatm/samudrace_atm_ic_0.nc";
inline const std::string kSamudraOcnModel =
    kSamudrace + "eocn/samudra_ocn_traced_masked_cuda_v2.pt";
inline const std::string kSamudraOcnIc =
    kSamudrace + "eocn/samudra_ocn_ic_0_icemask.nc";
inline const std::string kAce2Model =
    "/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/ace_traced_cuda.pt";
inline const std::string kAce2Ic = "/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/"
                                   "initial_conditions/1971010100_time_1.nc";

inline std::string spec_path(const std::string &file) {
  return std::string(EMULATOR_SPEC_DIR) + "/" + file;
}

/// An MCT attribute vector's storage: rAttr(nflds, lsize), point-major in C.
struct AttrVect {
  std::vector<std::string> names;
  std::vector<double> data;
  AttrVect(const std::string &list, std::size_t np, double fill = 0.0) {
    std::size_t start = 0;
    while (true) {
      const auto colon = list.find(':', start);
      names.push_back(list.substr(start, colon - start));
      if (colon == std::string::npos) {
        break;
      }
      start = colon + 1;
    }
    data.assign(names.size() * np, fill);
  }
  double &at(const std::string &n, std::size_t p) {
    const auto row = static_cast<std::size_t>(
        std::find(names.begin(), names.end(), n) - names.begin());
    return data[p * names.size() + row];
  }
};

/// The coupling descriptor for an import and an export vector.
inline EmulatorCouplingDesc coupling(AttrVect &in, AttrVect &out,
                                     std::size_t np) {
  return {in.data.data(), out.data.data(), static_cast<int>(in.names.size()),
          static_cast<int>(out.names.size()), static_cast<int>(np)};
}

/// A file with these contents, unique to this process and rank, removed when
/// this goes out of scope.
struct TempFile {
  std::string path;
  TempFile(const std::string &stem, const std::string &contents) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    path = (std::filesystem::temp_directory_path() /
            (stem + "_" + std::to_string(::getpid()) + "_" +
             std::to_string(rank)))
               .string();
    std::ofstream(path) << contents;
  }
  ~TempFile() { std::remove(path.c_str()); }
};

inline double global_sum(double local) {
  double total = 0.0;
  MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return total;
}

} // namespace test
} // namespace emulator

/// A Catch2 main that brackets the run with MPI, prints on rank 0 only, and
/// fails every rank if any fails.
#define EMULATOR_TEST_MPI_MAIN                                                 \
  int main(int argc, char *argv[]) {                                           \
    MPI_Init(&argc, &argv);                                                    \
    int rank = 0;                                                              \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                      \
    Catch::Session session;                                                    \
    if (rank != 0) {                                                           \
      session.configData().outputFilename = "%debug";                          \
    }                                                                          \
    int status = session.run(argc, argv);                                      \
    int worst = 0;                                                             \
    MPI_Allreduce(&status, &worst, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);       \
    MPI_Finalize();                                                            \
    return worst;                                                              \
  }

#endif // E3SM_EMULATOR_TEST_SUPPORT_HPP
