#ifndef P3_PROCESS_RATES_HPP
#define P3_PROCESS_RATES_HPP

#include <string>

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

// How p3_main_part2 runs: in one go (default), or in two steps, compute
// (Rates: fill the P3ProcessRates storage) then apply (Apply: read it).
enum class P3Part2Mode { Fused, Rates, Apply };

} // namespace p3
} // namespace scream

#endif // P3_PROCESS_RATES_HPP
