/**
 * @file calendar.hpp
 * @brief The model calendar, as the driver's NO_LEAP clock counts it.
 */

#ifndef E3SM_EMULATOR_COUPLING_CALENDAR_HPP
#define E3SM_EMULATOR_COUPLING_CALENDAR_HPP

namespace emulator {
namespace coupling {

/**
 * @brief Day of year with fraction, 1.0 at 00:00 on 1 January: shr_cal's
 *        julian day on the NO_LEAP calendar.
 * @throws std::invalid_argument on a date that calendar does not have
 */
double julian_day_noleap(int ymd, int tod);

} // namespace coupling
} // namespace emulator

#endif // E3SM_EMULATOR_COUPLING_CALENDAR_HPP
