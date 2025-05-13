#include "share/diag_process/diags_redux.hpp"

namespace scream {

DiagsRedux::DiagsRedux(const ekat::Comm &comm, const ekat::ParameterList &params)
    : m_comm(comm), m_params(params) {
  // Constructor implementation
}

template <RequestType RT>
void DiagsRedux::add_field(const std::string &fn, const FieldLayout &l, const ekat::units::Units &u,
                           const std::string &gn) {
  static_assert(RT == Required || RT == Computed,
                "Error! Invalid request type in call to add_field.\n"
                "Only Required and Computed are supported.");
  FieldIdentifier fid(fn, l, u, gn);
  FieldRequest req(fid);
  if constexpr (RT == Required) {
    m_required_field_requests.push_back(req);
  } else if constexpr (RT == Computed) {
    m_computed_field_requests.push_back(req);
  }
}

template void DiagsRedux::add_field<Required>(const std::string &, const FieldLayout &,
                                              const ekat::units::Units &, const std::string &);
template void DiagsRedux::add_field<Computed>(const std::string &, const FieldLayout &,
                                              const ekat::units::Units &, const std::string &);

} // namespace scream
