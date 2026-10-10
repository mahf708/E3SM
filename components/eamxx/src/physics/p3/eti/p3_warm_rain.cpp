#include "p3_warm_rain_impl.hpp"

namespace scream {
namespace p3 {

/*
 * Explicit instantiation for the warm-rain stage on Reals using the
 * default device.
 */

template struct Functions<Real,DefaultDevice>;

} // namespace p3
} // namespace scream
