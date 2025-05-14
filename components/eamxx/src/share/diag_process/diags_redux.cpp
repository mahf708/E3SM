#include "share/diag_process/diags_redux.hpp"

namespace scream {

DiagsRedux::DiagsRedux(const ekat::Comm &comm, const ekat::ParameterList &params)
    : m_comm(comm), m_params(params) {
  // Constructor implementation
}

} // namespace scream
