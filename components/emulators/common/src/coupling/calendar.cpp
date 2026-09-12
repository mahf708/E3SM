/**
 * @file calendar.cpp
 * @brief shr_cal_ymd2julian on the NO_LEAP calendar (share/util).
 */

#include "calendar.hpp"

#include <stdexcept>
#include <string>

namespace emulator {
namespace coupling {

namespace {

constexpr int days_before_month[12] = {0,   31,  59,  90,  120, 151,
                                       181, 212, 243, 273, 304, 334};
constexpr int days_in_month[12] = {31, 28, 31, 30, 31, 30,
                                   31, 31, 30, 31, 30, 31};

} // namespace

double julian_day_noleap(int ymd, int tod) {
  const int month = (ymd / 100) % 100;
  const int day = ymd % 100;
  if (month < 1 || month > 12 || day < 1 ||
      day > days_in_month[month - 1] || tod < 0 || tod > 86400) {
    throw std::invalid_argument("Date " + std::to_string(ymd) + " " +
                                std::to_string(tod) +
                                "s is not on the NO_LEAP calendar.");
  }
  return days_before_month[month - 1] + day + tod / 86400.0;
}

} // namespace coupling
} // namespace emulator
