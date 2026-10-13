#ifndef EAMXX_CONTRACTION_UTILS_HPP
#define EAMXX_CONTRACTION_UTILS_HPP

#include "share/field/field.hpp"

namespace scream {

// Replace the entries of f (where valid, if f has a valid_mask) with their
// square root. Used to get a standard deviation from a variance.
inline void sqrt_in_place (const Field& f)
{
  const auto& fl = f.get_header().get_identifier().get_layout();
  EKAT_REQUIRE_MSG (f.get_header().get_alloc_properties().get_alloc_size()==fl.size()*sizeof(Real),
      "Error! sqrt_in_place expects a contiguous, unpadded field.\n"
      " - field name: " + f.name() + "\n");
  auto data = f.get_internal_view_data<Real>();
  const bool masked = f.has_valid_mask();
  auto mask = masked ? f.get_valid_mask().get_internal_view_data<const int>() : nullptr;
  Kokkos::parallel_for("sqrt_in_place", fl.size(), KOKKOS_LAMBDA(const int i) {
    if (not masked or mask[i]) {
      data[i] = Kokkos::sqrt(data[i]);
    }
  });
}

} // namespace scream

#endif // EAMXX_CONTRACTION_UTILS_HPP
