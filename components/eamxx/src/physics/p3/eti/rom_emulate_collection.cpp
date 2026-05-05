#include "rom_emulate_collection_impl.hpp"

namespace scream {
namespace p3 {

/*
 * Explicit instantiation for doing autoconversion on Reals using the
 * default device.
 * ++ LL
 * Python must run on Host ONLY. 
 * not using template struct Functions<Real,DefaultDevice>;
 * now using template struct Functions<Real,HostDevice>;
 */

template struct Functions<Real,HostDevice>;

} // namespace p3
} // namespace scream
