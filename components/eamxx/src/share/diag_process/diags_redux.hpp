#ifndef EAMXX_DIAGS_REDUX_HPP
#define EAMXX_DIAGS_REDUX_HPP

#include "ekat/ekat_parameter_list.hpp"
#include "ekat/mpi/ekat_comm.hpp"

#include "share/field/field.hpp"
#include "share/field/field_identifier.hpp"
#include "share/field/field_request.hpp"

#include <list>

namespace scream {

/*
  DiagsRedux is the (new) diagnostics process impl

  The name "DiagsRedux" is temporary and it will eventually
  be replaced by AtmosphereDiagnostics (plural).
*/

class DiagsRedux : public std::enable_shared_from_this<DiagsRedux> {
public:
  // Constructor
  DiagsRedux(const ekat::Comm &comm, const ekat::ParameterList &params);

  // Destructor
  virtual ~DiagsRedux() = default;

  // getters
  const ekat::Comm &get_comm() const { return m_comm; }
  const ekat::ParameterList &get_params() const { return m_params; }
  const std::list<FieldRequest> &get_required_field_requests() const {
    return m_required_field_requests;
  }
  const std::list<FieldRequest> &get_computed_field_requests() const {
    return m_computed_field_requests;
  }

  // Add fields (RT can be Required or Computed)
  template <RequestType RT>
  void add_field(const std::string &fn, const FieldLayout &l, const ekat::units::Units &u,
                 const std::string &gn);

protected:
  // Fields in/out
  std::list<FieldRequest> m_required_field_requests;
  std::list<FieldRequest> m_computed_field_requests;

  // MPI communicator
  ekat::Comm m_comm;

  // Parameter list
  ekat::ParameterList m_params;
};

} // namespace scream

#endif // EAMXX_DIAGS_REDUX_HPP
