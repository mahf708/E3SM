#ifndef SCREAM_IO_CONTROL_HPP
#define SCREAM_IO_CONTROL_HPP

#include "share/util/eamxx_time_stamp.hpp"

#include <ekat_assert.hpp>
#include <ekat_string_utils.hpp>

#include <string>

namespace scream
{

// How to combine multiple snapshots in the output: instant, Max, Min, Average
// NOTE: this lives here (rather than in eamxx_io_utils.hpp) b/c IOControl needs
//       it to tell which timestamp identifies the snapshot of the current window
enum class OutputAvgType {
  Instant,
  Max,
  Min,
  Average,
  Invalid
};

inline std::string e2str(const OutputAvgType avg) {
  using OAT = OutputAvgType;
  switch (avg) {
    case OAT::Instant:  return "INSTANT";
    case OAT::Max:      return "MAX";
    case OAT::Min:      return "MIN";
    case OAT::Average:  return "AVERAGE";
    default:            return "INVALID";
  }
}

inline OutputAvgType str2avg (const std::string& s) {
  auto s_ci = ekat::upper_case(s);
  using OAT = OutputAvgType;
  for (auto e : {OAT::Instant, OAT::Max, OAT::Min, OAT::Average}) {
    if (s_ci==e2str(e)) {
      return e;
    }
  }

  return OAT::Invalid;
}

// Mini struct to hold IO frequency info
struct IOControl {

  // If frequency_units is not "none" or "never", frequency *must* be set to a positive number
  int frequency = -1;
  std::string frequency_units = "none";

  int nsamples_since_last_write = 0;  // Needed when updating output data, such as with the OAT::Average flag

  // The window of the snapshot that is currently being accumulated:
  //  - window_beg: when the current window started (i.e., the last write, or the run t0)
  //  - window_end: when the current window ends (i.e., the next scheduled write)
  // NOTE: these rotate *inside* OutputManager::run, when advance_window is called
  //       (right after the snapshot is written). Before that call they describe the
  //       snapshot being written, after it they describe the next one.
  util::TimeStamp window_end;
  util::TimeStamp window_beg;

  // At run time, set dt in the struct, so we can compute window_end correctly,
  // even if freq_units is "nsteps"
  // NOTE: this ASSUMES dt is constant throughout the run (i.e., no time adaptivity).
  //       An error will be thrown if dt changes, so developers can fix this if we ever support variable dt
  double dt = 0;

  bool output_enabled () const {
    return frequency_units!="none" && frequency_units!="never";
  }

  bool is_write_step (const util::TimeStamp& ts) const {
    if (not output_enabled()) return false;
    return frequency_units=="nsteps" ? ts.get_num_steps()==window_end.get_num_steps()
                                     : (ts.get_date()==window_end.get_date() and
                                        ts.get_time()==window_end.get_time());
  }

  // The timestamp identifying the snapshot of the current window, that is, the
  // *start* of its averaging window. This is what tells which day/month/year a
  // snapshot belongs to, for file storage types other than NumSnaps.
  // NOTE: if you consider INSTANT output as an "average" over the degenerate
  //       window [window_end,window_end], then its start is window_end. Using
  //       window_beg for INSTANT would point at the *previous* snapshot instead.
  const util::TimeStamp& snapshot_ts (const OutputAvgType avg_type) const {
    return avg_type==OutputAvgType::Instant ? window_end : window_beg;
  }

  void set_frequency_units (const std::string& freq_unit) {
    if (freq_unit=="none" or freq_unit=="never") {
      frequency_units = freq_unit;
    } else if (freq_unit=="nstep" or freq_unit=="nsteps") {
      frequency_units = "nsteps";
    } else if (freq_unit=="nsecond" or freq_unit=="nseconds" or freq_unit=="nsecs") {
      frequency_units = "nsecs";
    } else if (freq_unit=="nminute" or freq_unit=="nminutes" or freq_unit=="nmins") {
      frequency_units = "nmins";
    } else if (freq_unit=="nhour" or freq_unit=="nhours") {
      frequency_units = "nhours";
    } else if (freq_unit=="nday" or freq_unit=="ndays") {
      frequency_units = "ndays";
    } else if (freq_unit=="nmonth" or freq_unit=="nmonths") {
      frequency_units = "nmonths";
    } else if (freq_unit=="nyear" or freq_unit=="nyears") {
      frequency_units = "nyears";
    } else {
      // TODO - add support for "end" as an option
      EKAT_ERROR_MSG("Error! Unsupported frequency units of " + freq_unit + " provided.");
    }
  }

  void set_dt (const double dt_in) {
    EKAT_REQUIRE_MSG (dt==0,
        "[IOControl::set_dt] Error! Cannot reset dt once it is set.\n");

    dt = dt_in;
  }

  // Closes the current window at ts, and opens the next one.
  // NOTE: this is the ONLY way the window is meant to rotate. Doing it in a
  //       single place is what keeps window_beg/window_end consistent with
  //       each other (and with nsamples_since_last_write).
  void advance_window (const util::TimeStamp& ts) {
    window_beg = ts;
    compute_window_end();
    nsamples_since_last_write = 0;
  }

  // Computes window_end from frequency and window_beg
  void compute_window_end () {
    EKAT_REQUIRE_MSG (window_beg.is_valid(),
        "Error! Cannot compute window_end, since window_beg was never set.\n");
    window_end = window_beg;
    if (frequency_units=="nsteps") {
      // This avoids having an invalid/wrong date/time in StorageSpecs::snapshot_fits
      // if storage type is NumSnaps
      window_end += dt*frequency;
      window_end.set_num_steps(window_beg.get_num_steps()+frequency);
    } else if (frequency_units=="nsecs") {
      window_end += frequency;
    } else if (frequency_units=="nmins") {
      window_end += frequency*60;
    } else if (frequency_units=="nhours") {
      window_end += frequency*3600;
    } else if (frequency_units=="ndays") {
      window_end += frequency*86400;
    } else if (frequency_units=="nmonths") {
      auto date = window_beg.get_date();
      int temp = date[1] + frequency - 1;
      date[1]  = temp % 12 + 1;
      date[0] += temp / 12;

      // NOTE: we MAY have moved to an invalid date. E.g., if last_write
      // was on Mar 31st, and units='nmonths', date now points to Apr 31st.
      // We fix this by adjusting the date to the last day of the month.
      // HOWEVER, this means we will *always* write on the 30th of each month after then,
      // since we have no memory of the fact that we were writing on the 31st before.
      auto month_beg = util::TimeStamp({date[0],date[1],1},{0,0,0});
      auto last_day = month_beg.days_in_curr_month();
      date[2] = std::min(date[2],last_day);

      window_end = util::TimeStamp(date,window_beg.get_time());
    } else if (frequency_units=="nyears") {
      auto date = window_beg.get_date();
      date[0] += frequency;
      window_end = util::TimeStamp(date,window_beg.get_time());
    } else {
      EKAT_ERROR_MSG ("Error! Unrecognized/unsupported frequency unit '" + frequency_units + "'\n");
    }
  }
};

} // namespace scream
#endif // SCREAM_IO_CONTROL_HPP
