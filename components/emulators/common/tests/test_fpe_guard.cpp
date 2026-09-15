// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "fpe_guard.hpp"

#include <cfenv>

namespace emulator {
namespace inference {
namespace test {

#if defined(__GLIBC__)

TEST_CASE("FpeGuard suspends traps and restores exactly the prior set",
          "[fpe]") {
  // The set an E3SM debug build enables.
  const int debug_traps = FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW;

  fedisableexcept(FE_ALL_EXCEPT);
  feenableexcept(debug_traps);
  REQUIRE(fegetexcept() == debug_traps);

  {
    FpeGuard guard;
    REQUIRE(fegetexcept() == 0);

    // What a model kernel is entitled to do while guarded.  volatile keeps
    // the compiler from folding the division away.
    volatile double zero = 0.0;
    volatile double inf = 1.0 / zero;
    (void)inf;
  }

  REQUIRE(fegetexcept() == debug_traps);

  // The flag raised inside the guard must not survive it: re-enabling a
  // trap with its flag already set is how the fault would fire later, at
  // an innocent line of Fortran.
  REQUIRE(fetestexcept(FE_DIVBYZERO) == 0);

  fedisableexcept(FE_ALL_EXCEPT);
}

TEST_CASE("FpeGuard with no traps enabled changes nothing", "[fpe]") {
  fedisableexcept(FE_ALL_EXCEPT);
  {
    FpeGuard guard;
    REQUIRE(fegetexcept() == 0);
  }
  REQUIRE(fegetexcept() == 0);
}

TEST_CASE("FpeGuards nest", "[fpe]") {
  fedisableexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID);
  {
    FpeGuard outer;
    {
      FpeGuard inner;
      REQUIRE(fegetexcept() == 0);
    }
    REQUIRE(fegetexcept() == 0);
  }
  REQUIRE(fegetexcept() == FE_INVALID);
  fedisableexcept(FE_ALL_EXCEPT);
}

#else

TEST_CASE("FpeGuard is a no-op without glibc", "[fpe]") {
  FpeGuard guard;
  SUCCEED();
}

#endif

} // namespace test
} // namespace inference
} // namespace emulator
