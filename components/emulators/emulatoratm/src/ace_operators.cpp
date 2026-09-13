/**
 * @file ace_operators.cpp
 * @brief Implementation of the ACE operators.
 */

#include "ace_operators.hpp"

#include <stdexcept>

namespace emulator {
namespace atm {

namespace {

model::FieldRef ref(const config::Section &s, const std::string &key) {
  return model::FieldRef::parse(s.string(key), s.where() + "." + key);
}

std::map<std::string, model::FieldRef>
refs(const config::Section &s, std::initializer_list<const char *> required,
     std::initializer_list<const char *> optional) {
  std::vector<const char *> allowed(required);
  allowed.insert(allowed.end(), optional.begin(), optional.end());
  for (const auto &k : s.keys()) {
    bool known = false;
    for (const char *a : allowed) {
      known = known || k == a;
    }
    if (!known) {
      throw std::invalid_argument(s.where() + ": unknown key '" + k + "'.");
    }
  }
  std::map<std::string, model::FieldRef> out;
  for (const char *k : required) {
    out[k] = ref(s, k);
  }
  for (const char *k : optional) {
    if (s.has(k)) {
      out[k] = ref(s, k);
    }
  }
  return out;
}

} // namespace

SurfaceInputsOperator::SurfaceInputsOperator(const config::Section &o,
                                             const model::ModelInfo &)
    : m_tolerance(o.number_or("tolerance", 0.05)) {
  o.only({"operator", "coupler", "emulator_ts", "ocean", "to", "tolerance"});
  const auto c = o.section("coupler");
  c.only({"lfrac", "ofrac", "ifrac", "merged_ts"});
  m_lfrac = ref(c, "lfrac");
  m_ofrac = ref(c, "ofrac");
  m_ifrac = ref(c, "ifrac");
  m_sx_t = ref(c, "merged_ts");
  m_ts_emulator = ref(o, "emulator_ts");
  if (o.has("ocean")) {
    const auto oc = o.section("ocean");
    oc.only({"ice_fraction", "sst"});
    m_from_ocean = true;
    m_ocean_ice = ref(oc, "ice_fraction");
    m_ocean_sst = ref(oc, "sst");
  }
  const auto to = o.section("to");
  to.only({"landfrac", "ocnfrac", "icefrac", "ts"});
  m_landfrac = ref(to, "landfrac");
  m_ocnfrac = ref(to, "ocnfrac");
  m_icefrac = ref(to, "icefrac");
  m_ts = ref(to, "ts");
  for (const auto *r : {&m_landfrac, &m_ocnfrac, &m_icefrac, &m_ts}) {
    if (r->set() != model::FieldRef::Set::Inputs) {
      throw std::invalid_argument(to.where() + ": '" + r->to_string() +
                                  "' must be a network input.");
    }
  }
}

model::Declarations SurfaceInputsOperator::declarations() const {
  model::Declarations d;
  d.writes_inputs = {m_landfrac.name(), m_ocnfrac.name(), m_icefrac.name(),
                     m_ts.name()};
  return d;
}

void SurfaceInputsOperator::before_step(const model::StepInfo &,
                                        model::Fields &f) {
  SurfaceChannels surface{m_landfrac.write(f), m_ocnfrac.write(f),
                          m_icefrac.write(f), m_ts.write(f)};
  SurfaceCouplerInputs from_coupler{m_lfrac.read(f), m_ofrac.read(f),
                                    m_ifrac.read(f), m_sx_t.read(f),
                                    m_ts_emulator.read(f)};
  if (m_from_ocean) {
    from_coupler.ocean_ice_fraction = m_ocean_ice.read(f);
    from_coupler.ocean_sst = m_ocean_sst.read(f);
  }
  compute_surface_inputs(from_coupler, surface, m_tolerance);
}

SurfaceExportsOperator::SurfaceExportsOperator(const config::Section &o,
                                               const model::ModelInfo &) {
  o.only({"operator", "layer", "reference_height", "cap_humidity",
          "frozen_precip_in_m_per_s", "diurnal_shortwave", "from", "to"});
  const auto layer = o.string("layer");
  if (layer == "near_surface") {
    m_options.layer = SurfaceLayer::NearSurface;
  } else if (layer == "lowest_level") {
    m_options.layer = SurfaceLayer::LowestLevel;
  } else {
    throw std::invalid_argument(o.where() + ".layer: '" + layer +
                                "' is neither near_surface nor lowest_level.");
  }
  m_options.reference_height =
      o.number_or("reference_height", m_options.reference_height);
  m_options.cap_humidity = o.boolean_or("cap_humidity", m_options.cap_humidity);
  m_options.frozen_precip_in_m_per_s =
      o.boolean_or("frozen_precip_in_m_per_s", m_options.frozen_precip_in_m_per_s);
  m_options.diurnal_shortwave =
      o.boolean_or("diurnal_shortwave", m_options.diurnal_shortwave);
  m_from = refs(o.section("from"),
                {"ps", "phis", "t_lowest", "q_lowest", "u_lowest", "v_lowest",
                 "flds", "fsds", "precip", "solin_now", "solin_window"},
                {"t_2m", "q_2m", "u_10m", "v_10m", "fsus", "frozen_precip"});
  m_to = refs(o.section("to"),
              {"z", "u", "v", "tbot", "ptem", "shum", "pbot", "pslv", "dens",
               "topo", "lwdn", "rainc", "rainl", "snowc", "snowl", "swndr",
               "swvdr", "swndf", "swvdf", "swnet"},
              {});
}

void SurfaceExportsOperator::exports(const model::StepInfo &,
                                     model::Fields &f) {
  auto in_ = [&](const char *k) {
    const auto it = m_from.find(k);
    return it == m_from.end() ? std::span<const double>{} : it->second.read(f);
  };
  auto out_ = [&](const char *k) { return m_to.at(k).write(f); };
  SurfaceInputs in;
  in.ps = in_("ps");
  in.phis = in_("phis");
  in.t_lowest = in_("t_lowest");
  in.q_lowest = in_("q_lowest");
  in.u_lowest = in_("u_lowest");
  in.v_lowest = in_("v_lowest");
  in.t_2m = in_("t_2m");
  in.q_2m = in_("q_2m");
  in.u_10m = in_("u_10m");
  in.v_10m = in_("v_10m");
  in.flds = in_("flds");
  in.fsds = in_("fsds");
  in.fsus = in_("fsus");
  in.precip = in_("precip");
  in.frozen_precip = in_("frozen_precip");
  in.solin_now = in_("solin_now");
  in.solin_window = in_("solin_window");
  SurfaceExports out{out_("z"),     out_("u"),     out_("v"),     out_("tbot"),
                     out_("ptem"),  out_("shum"),  out_("pbot"),  out_("pslv"),
                     out_("dens"),  out_("topo"),  out_("lwdn"),  out_("rainc"),
                     out_("rainl"), out_("snowc"), out_("snowl"), out_("swndr"),
                     out_("swvdr"), out_("swndf"), out_("swvdf"), out_("swnet")};
  compute_surface_exports(in, m_options, out);
}

void register_atm_operators() {
  auto &r = model::OperatorRegistry::instance();
  r.add("ace.surface_inputs",
        [](const config::Section &o, const model::ModelInfo &i) {
          return std::make_unique<SurfaceInputsOperator>(o, i);
        });
  r.add("ace.surface_exports",
        [](const config::Section &o, const model::ModelInfo &i) {
          return std::make_unique<SurfaceExportsOperator>(o, i);
        });
}

} // namespace atm
} // namespace emulator
