#ifndef P3_PROCESS_RATES_HPP
#define P3_PROCESS_RATES_HPP

#include <string>
#include <vector>

namespace scream {
namespace p3 {

/*
 * Named registry of the quantities p3_main_part2 computes before applying
 * them: every process rate, mapped back to cell averages, plus the two
 * auxiliary quantities the update needs (rime density of new rime, and the
 * wet-growth flag, stored as 0/1).
 *
 * part2 can run in two steps (see P3Part2Mode): compute these quantities and
 * store them, then read them back and apply them (conservation checks, update
 * of the prognostic state, diagnostics). Between the two steps, anything can
 * overwrite any subset of them by name, e.g. an emulator of some processes.
 *
 * Rates are CELL AVERAGES, in P3's units and sign conventions, i.e., as seen
 * by the conservation routines and update_prognostic_liquid/ice.
 *
 * To add a quantity: add it to the list below (the name must match the local
 * variable in p3_main_part2), nothing else is needed.
 */
#define P3_PROCESS_RATE_PACKS(X)                                              \
  /* warm-phase process rates */                                              \
  X(qc2qr_accret_tend)        /* cloud droplet accretion by rain       */     \
  X(qr2qv_evap_tend)          /* rain evaporation                      */     \
  X(qc2qr_autoconv_tend)      /* cloud droplet autoconversion to rain  */     \
  X(nc_accret_tend)           /* nc change from accretion by rain      */     \
  X(nc_selfcollect_tend)      /* nc change from self-collection        */     \
  X(nc2nr_autoconv_tend)      /* nc change from autoconversion         */     \
  X(nr_selfcollect_tend)      /* nr change from self-collection        */     \
  X(nr_evap_tend)             /* nr change from evaporation            */     \
  X(ncautr)                   /* nr change from autoconversion         */     \
  /* ice-phase process rates */                                               \
  X(qi2qv_sublim_tend)        /* sublimation of ice                    */     \
  X(nr_ice_shed_tend)         /* nr source from rain/ice shedding      */     \
  X(qc2qi_hetero_freeze_tend) /* immersion freezing of droplets        */     \
  X(qr2qi_collect_tend)       /* collection of rain mass by ice        */     \
  X(qc2qr_ice_shed_tend)      /* qr source from cloud/ice shedding     */     \
  X(qi2qr_melt_tend)          /* melting of ice                        */     \
  X(qc2qi_collect_tend)       /* collection of cloud water by ice      */     \
  X(qr2qi_immers_freeze_tend) /* immersion freezing of rain            */     \
  X(qv2qi_nucleat_tend)       /* deposition/condensation-freezing nuc. */     \
  X(ni2nr_melt_tend)          /* ni change from melting                */     \
  X(nc_collect_tend)          /* nc change from collection by ice      */     \
  X(ncshdc)                   /* nr source from cloud/ice shedding     */     \
  X(nc2ni_immers_freeze_tend) /* nc change from immersion freezing     */     \
  X(nr_collect_tend)          /* nr change from collection by ice      */     \
  X(ni_selfcollect_tend)      /* ni change from ice self-collection    */     \
  X(ni_nucleat_tend)          /* ni change from nucleation             */     \
  X(qv2qi_vapdep_tend)        /* vapor deposition                      */     \
  X(qc2qi_berg_tend)          /* Bergeron process                      */     \
  X(nr2ni_immers_freeze_tend) /* nr change from immersion freezing     */     \
  X(ni_sublim_tend)           /* ni change from sublimation            */     \
  /* heterogeneous freezing (classical nucleation theory) */                  \
  X(ncheti_cnt) X(qcheti_cnt) X(nicnt) X(qicnt) X(ninuc_cnt) X(qinuc_cnt)     \
  /* auxiliary */                                                             \
  X(rho_qm_cloud)             /* density of new rime                   */

// All the quantities: the packs above, and the wet-growth flag (a mask in part2)
#define P3_PROCESS_RATES(X) P3_PROCESS_RATE_PACKS(X) X(wetgrowth)

struct P3ProcessRates {
#define P3_PR_ENUM(name) name,
  enum Index : int { P3_PROCESS_RATES(P3_PR_ENUM) num_rates };
#undef P3_PR_ENUM

  static constexpr int num_packs = num_rates - 1; // all but wetgrowth

  static const char* name (const int i) {
#define P3_PR_NAME(name) #name,
    static const char* names[] = { P3_PROCESS_RATES(P3_PR_NAME) };
#undef P3_PR_NAME
    return i>=0 && i<num_rates ? names[i] : "";
  }

  // Index of a quantity from its name, or -1 if there is none
  static int index (const std::string& n) {
    for (int i=0; i<num_rates; ++i) {
      if (n==name(i)) return i;
    }
    return -1;
  }
};

/*
 * Named registry of what P3's sedimentation (cloud, rain, ice) does to the
 * state: one tendency per sedimenting quantity [unit/s], and the surface
 * precipitation rates (precip_liq_surf, precip_ice_surf [m/s], per column).
 *
 * With a sedimentation hook, p3_main saves the state before sedimentation,
 * runs it, stores these tendencies, runs the hook (outside of any kernel),
 * then re-applies the tendencies the hook may have changed:
 *   x = x_before + x_sed_tend*dt
 * so that tendencies the hook does not change stay BFB.
 */
#define P3_SEDIMENTATION_RATES(X) \
  X(qc_sed_tend) X(nc_sed_tend) X(qr_sed_tend) X(nr_sed_tend) \
  X(qi_sed_tend) X(ni_sed_tend) X(qm_sed_tend) X(bm_sed_tend)

struct P3SedimentationRates {
#define P3_SR_ENUM(name) name,
  enum Index : int { P3_SEDIMENTATION_RATES(P3_SR_ENUM) num_rates };
#undef P3_SR_ENUM

  static const char* name (const int i) {
#define P3_SR_NAME(name) #name,
    static const char* names[] = { P3_SEDIMENTATION_RATES(P3_SR_NAME) };
#undef P3_SR_NAME
    return i>=0 && i<num_rates ? names[i] : "";
  }

  // Index of a tendency from its name, or -1 if there is none
  static int index (const std::string& n) {
    for (int i=0; i<num_rates; ++i) {
      if (n==name(i)) return i;
    }
    return -1;
  }
};

/*
 * The computations of p3_main_part2 that produce the process rates, with what
 * each one writes and reads among the rates, and which other computations use
 * its local results. Emulators that replace (physics: skip) rates let part2
 * skip a computation when everything it writes is replaced, and nothing that
 * still runs reads it (see skippable).
 *
 * To add a computation: add it to the list, and to writes/reads/feeds; wrap
 * its call in part2 with P3_RUN_PRODUCER.
 */
#define P3_RATE_PRODUCERS(X) \
  X(ice_cldliq_collection) X(ice_rain_collection) X(ice_self_collection) X(ice_melting) \
  X(ice_cldliq_wet_growth) X(ice_relaxation_timescale) X(rime_density) \
  X(ice_classical_nucleation) X(cldliq_immersion_freezing) X(rain_immersion_freezing) \
  X(rain_evaporation) X(ice_deposition_sublimation) X(ice_nucleation) \
  X(cloud_water_autoconversion) X(droplet_self_collection) X(cloud_rain_accretion) \
  X(rain_self_collection)

struct P3RateProducers {
#define P3_RP_ENUM(name) name,
  enum Index : int { P3_RATE_PRODUCERS(P3_RP_ENUM) num_producers };
#undef P3_RP_ENUM
  static_assert(num_producers <= 32, "The producer mask is 32 bits.");

  static const char* name (const int i) {
#define P3_RP_NAME(name) #name,
    static const char* names[] = { P3_RATE_PRODUCERS(P3_RP_NAME) };
#undef P3_RP_NAME
    return i>=0 && i<num_producers ? names[i] : "";
  }

  // The rates (P3ProcessRates::Index) a computation sets or changes
  static std::vector<int> writes (const int p) {
    using R = P3ProcessRates;
    switch (p) {
      case ice_cldliq_collection:      return {R::qc2qi_collect_tend, R::nc_collect_tend, R::qc2qr_ice_shed_tend, R::ncshdc};
      case ice_rain_collection:        return {R::qr2qi_collect_tend, R::nr_collect_tend};
      case ice_self_collection:        return {R::ni_selfcollect_tend};
      case ice_melting:                return {R::qi2qr_melt_tend, R::ni2nr_melt_tend};
      case ice_cldliq_wet_growth:      return {R::wetgrowth, R::qc2qi_collect_tend, R::qr2qi_collect_tend,
                                               R::qc2qr_ice_shed_tend, R::nr_ice_shed_tend};
      case ice_relaxation_timescale:   return {};
      case rime_density:               return {R::rho_qm_cloud};
      case ice_classical_nucleation:   return {R::ncheti_cnt, R::qcheti_cnt, R::nicnt, R::qicnt, R::ninuc_cnt, R::qinuc_cnt};
      case cldliq_immersion_freezing:  return {R::qc2qi_hetero_freeze_tend, R::nc2ni_immers_freeze_tend};
      case rain_immersion_freezing:    return {R::qr2qi_immers_freeze_tend, R::nr2ni_immers_freeze_tend};
      case rain_evaporation:           return {R::qr2qv_evap_tend, R::nr_evap_tend};
      case ice_deposition_sublimation: return {R::qv2qi_vapdep_tend, R::qi2qv_sublim_tend, R::ni_sublim_tend, R::qc2qi_berg_tend};
      case ice_nucleation:             return {R::qv2qi_nucleat_tend, R::ni_nucleat_tend};
      case cloud_water_autoconversion: return {R::qc2qr_autoconv_tend, R::nc2nr_autoconv_tend, R::ncautr};
      case droplet_self_collection:    return {R::nc_selfcollect_tend};
      case cloud_rain_accretion:       return {R::qc2qr_accret_tend, R::nc_accret_tend};
      case rain_self_collection:       return {R::nr_selfcollect_tend};
      default:                         return {};
    }
  }

  // The rates a computation reads (or changes, which reads them too)
  static std::vector<int> reads (const int p) {
    using R = P3ProcessRates;
    switch (p) {
      case ice_cldliq_wet_growth:   return {R::qc2qi_collect_tend, R::qr2qi_collect_tend,
                                            R::qc2qr_ice_shed_tend, R::nr_ice_shed_tend};
      case rime_density:            return {R::qc2qi_collect_tend};
      case droplet_self_collection: return {R::nc2nr_autoconv_tend};
      default:                      return {};
    }
  }

  // The computations that use a computation's local results (not rates)
  static std::vector<int> feeds (const int p) {
    switch (p) {
      case ice_relaxation_timescale: return {rain_evaporation, ice_deposition_sublimation};
      default:                       return {};
    }
  }

  // Mask (bit p for producer p) of the computations part2 can skip when the
  // rates in `replaced` (by P3ProcessRates::Index) are replaced
  static unsigned skippable (const std::vector<bool>& replaced) {
    std::vector<bool> skip(num_producers);
    for (int p=0; p<num_producers; ++p) {
      bool all = true;
      for (int r : writes(p)) all = all && replaced[r];
      skip[p] = all;
    }
    // A computation runs if something that runs needs what it writes
    for (bool changed=true; changed; ) {
      changed = false;
      for (int p=0; p<num_producers; ++p) {
        if (not skip[p]) continue;
        bool needed = false;
        for (int q=0; q<num_producers and not needed; ++q) {
          if (skip[q]) continue;
          for (int r : reads(q))
            for (int w : writes(p)) needed = needed || r==w;
        }
        const auto f = feeds(p);
        if (not f.empty()) {
          for (int q : f) needed = needed || not skip[q];
        }
        if (needed) { skip[p] = false; changed = true; }
      }
    }
    unsigned mask = 0;
    for (int p=0; p<num_producers; ++p) if (skip[p]) mask |= 1u << p;
    return mask;
  }
};

// How p3_main_part2 runs: in one go (default), or in two steps, compute
// (Rates: fill the P3ProcessRates storage) then apply (Apply: read it).
enum class P3Part2Mode { Fused, Rates, Apply };

} // namespace p3
} // namespace scream

#endif // P3_PROCESS_RATES_HPP
