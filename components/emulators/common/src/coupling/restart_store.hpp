/**
 * @file restart_store.hpp
 * @brief The seam a component's restart state goes through.
 */

#ifndef E3SM_EMULATOR_COUPLING_RESTART_STORE_HPP
#define E3SM_EMULATOR_COUPLING_RESTART_STORE_HPP

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace emulator {
namespace coupling {

/**
 * @brief Somewhere named arrays and integers can be put and got back.
 *
 * Deliberately tiny, so an implementation over netCDF, SCORPIO or a map for
 * a test is a few lines each, and so the round trip -- which is the whole
 * point of a restart -- can be tested in memory.
 */
class RestartStore {
public:
  virtual ~RestartStore() = default;

  virtual void write_array(std::string_view name,
                           std::span<const double> data) = 0;
  virtual void write_int(std::string_view name, std::int64_t value) = 0;

  /**
   * @return false if the store has no such array, leaving `data` untouched
   * @throws std::runtime_error if it has one of a different length: a
   *         partial read is worse than a missing field
   */
  virtual bool read_array(std::string_view name, std::span<double> data) = 0;
  /// @return false if absent
  virtual bool read_int(std::string_view name, std::int64_t &value) = 0;
};

/// A RestartStore in memory, for tests.
class MemoryRestartStore : public RestartStore {
public:
  void write_array(std::string_view name,
                   std::span<const double> data) override;
  void write_int(std::string_view name, std::int64_t value) override;
  bool read_array(std::string_view name, std::span<double> data) override;
  bool read_int(std::string_view name, std::int64_t &value) override;

  bool has(std::string_view name) const;
  /// Drop an entry, to test a restart written before it existed.
  void erase(std::string_view name);
  std::vector<std::string> names() const;

private:
  std::map<std::string, std::vector<double>, std::less<>> m_arrays;
  std::map<std::string, std::int64_t, std::less<>> m_ints;
};

/// What to do when a restart lacks something.
enum class Missing {
  /// Refuse to start.  For anything whose loss changes the answer.
  Fatal,
  /**
   * Start from the empty state.  Only for state a newer version added, whose
   * empty value is self-consistent (an accumulator with zero samples).
   */
  StartEmpty
};

/// Join a prefix and a name as restart files spell them: "prefix.name".
std::string restart_name(std::string_view prefix, std::string_view name);

} // namespace coupling
} // namespace emulator

#endif // E3SM_EMULATOR_COUPLING_RESTART_STORE_HPP
