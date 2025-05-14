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
  be replaced by AtmosphereDiagnostic.
*/

class DiagsRedux : public std::enable_shared_from_this<DiagsRedux> {
public:
  // Constructor
  DiagsRedux(const ekat::Comm &comm, const ekat::ParameterList &params);

  // Destructor
  virtual ~DiagsRedux() = default;

  // Get the name of the diagnostics
  virtual std::string name() const = 0;

  // Initialize the diagnostics
  void initialize() {
    // Call the implementation-specific initialization
    initialize_impl();
  }
  // Run the diagnostics with a given timestamp
  void run(const util::TimeStamp &timestamp) {
    // Call the implementation-specific run
    run_impl(timestamp);
  }
  // Finalize the diagnostics
  void finalize() {
    // Call the implementation-specific finalization
    finalize_impl();
  }

  // Getters
  const ekat::Comm &get_comm() const { return m_comm; }
  const ekat::ParameterList &get_params() const { return m_params; }
  const std::list<FieldRequest> &get_required_field_requests() const {
    return m_required_field_requests;
  }
  const std::list<FieldRequest> &get_computed_field_requests() const {
    return m_computed_field_requests;
  }
  const std::list<Field> &get_fields_in() const { return m_fields_in; }
  const std::list<Field> &get_fields_out() const { return m_fields_out; }

  // Add fields (RT can be Required or Computed)
  template <RequestType RT>
  void add_field(const std::string &field_name, const FieldLayout &layout,
                 const ekat::units::Units &units, const std::string &grid_name) {
    static_assert(RT == Required || RT == Computed,
                  "Error! Invalid request type in call to add_field.\n"
                  "Only Required and Computed are supported.");
    FieldIdentifier fid(field_name, layout, units, grid_name);
    FieldRequest req(fid);
    if constexpr (RT == Required) {
      m_required_field_requests.push_back(std::move(req));
    } else if constexpr (RT == Computed) {
      m_computed_field_requests.push_back(std::move(req));
    }
  }

  // Somewhere else, we will populate the fields
  // First, sanity checks
  bool has_required_field(const FieldIdentifier &fid) const {
    return std::any_of(m_required_field_requests.begin(), m_required_field_requests.end(),
                       [&fid](const FieldRequest &req) { return req.fid == fid; });
  }
  bool has_computed_field(const FieldIdentifier &fid) const {
    return std::any_of(m_computed_field_requests.begin(), m_computed_field_requests.end(),
                       [&fid](const FieldRequest &req) { return req.fid == fid; });
  }
  // Second, set fields if not set
  void set_required_field(const Field &f) {
    // Check if the field is in the list of required field requests
    EKAT_REQUIRE_MSG(
        has_required_field(f.get_header().get_identifier()),
        "Error! Input field is not required by this diagnostics process.\n"
        "    field id: " + f.get_header().get_identifier().get_id_string() + "\n"
        "    diag process: " + this->name() + "\n"
        "Something is wrong up the call stack. Please, contact developers.\n");
    // If the field is not already in the list of fields_in, add it
    if (std::find(m_fields_in.begin(), m_fields_in.end(), f) == m_fields_in.end()) {
      m_fields_in.push_back(f);
    }
  }
  void set_computed_field(const Field &f) {
    // Check if the field is in the list of computed field requests
    EKAT_REQUIRE_MSG(
        has_computed_field(f.get_header().get_identifier()),
        "Error! Input field is not computed by this diagnostics process.\n"
        "    field id: " + f.get_header().get_identifier().get_id_string() + "\n"
        "    diag process: " + this->name() + "\n"
        "Something is wrong up the call stack. Please, contact developers.\n");
    // If the field is not already in the list of fields_out, add it
    if (std::find(m_fields_out.begin(), m_fields_out.end(), f) == m_fields_out.end()) {
      m_fields_out.push_back(f);
    }
  }

protected:
  // Initialize (including requesting fields)
  virtual void initialize_impl() = 0;
  // Run with a timestamp to return early
  virtual void run_impl(const util::TimeStamp &timestamp) = 0;
  // Any finalization needed
  virtual void finalize_impl() = 0;

  // Fields in/out
  // FieldRequest
  std::list<FieldRequest> m_required_field_requests;
  std::list<FieldRequest> m_computed_field_requests;
  // Field
  std::list<Field> m_fields_in;
  std::list<Field> m_fields_out;

  // MPI communicator
  ekat::Comm m_comm;

  // Parameter list
  ekat::ParameterList m_params;
};

} // namespace scream

#endif // EAMXX_DIAGS_REDUX_HPP
