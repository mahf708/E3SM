#ifndef EAMXX_TEST_DIAGS_REDUX_HPP
#define EAMXX_TEST_DIAGS_REDUX_HPP

#include "share/diag_process/diags_redux.hpp"

namespace scream {

// A concrete implementation of DiagsRedux for testing purposes
class TestDiagsRedux : public DiagsRedux {
public:
  // Use the base class constructor
  TestDiagsRedux(const ekat::Comm &comm, const ekat::ParameterList &params)
      : DiagsRedux(comm, params) {}

  // Override the name method
  std::string name() const override {
    return "TestDiagsRedux";
  }

protected:
  // Implementation of the pure virtual methods
  void initialize_impl() override {
    // No-op for testing
  }

  void run_impl(const util::TimeStamp &timestamp) override {
    // No-op for testing
    (void)timestamp; // Avoid unused parameter warning
  }

  void finalize_impl() override {
    // No-op for testing
  }
};

} // namespace scream

#endif // EAMXX_TEST_DIAGS_REDUX_HPP
