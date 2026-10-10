#ifndef P3_WARM_RAIN_MLP_HPP
#define P3_WARM_RAIN_MLP_HPP

#include "share/core/eamxx_types.hpp"

#include <ekat_assert.hpp>
#include <ekat_pack_kokkos.hpp>
#include <ekat_pack_math.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace scream {
namespace p3 {

/*
 * Device-native evaluation of the warm-rain MLP emulator, for use inside
 * Kokkos kernels (one Pack of levels at a time).
 *
 * It implements the same contract as p3_warm_rain_emulator.py, and all model
 * numbers (weights, normalization, gates, training envelope, number-rate
 * constants) come from a text file written by export_kokkos_mlp.py from the
 * same .pt model file. So, as with the python module, a retrained model is a
 * file swap, as long as the architecture is an MLP with at most max_layers
 * layers of at most max_width neurons.
 *
 * Unlike the python module, the network is evaluated in the precision of Pack
 * (not float32), so results match the python ones to float32 roundoff.
 */
template <typename PackT, typename DeviceT>
struct WarmRainMLP {
  using Pack   = PackT;
  using Scalar = typename Pack::scalar;
  using Mask   = ekat::Mask<Pack::n>;
  using KT     = ekat::KokkosTypes<DeviceT>;

  static constexpr int n_in       = 4;  // qc, nc, qr, nr per volume
  static constexpr int n_out      = 4;  // AU, AC, SC_c, SC_r per volume
  static constexpr int max_layers = 4;
  static constexpr int max_width  = 64;

  enum Activation { Tanh = 0, Relu, Silu, Softplus };

  // Everything but the weights is small, and is passed by value to kernels
  struct Config {
    int num_layers = 0;
    int width[max_layers+1] = {0};       // width[0] = n_in, width[num_layers] = n_out
    int weight_offset[max_layers] = {0}; // offsets in the params view, row-major (out,in)
    int bias_offset[max_layers] = {0};
    int activation = Tanh, output_activation = Softplus;
    Scalar x_mean[n_in], x_std[n_in], floors[n_in], y_scale[n_out], y_log_std[n_out];
    Scalar qc_gt, qr_gt;                  // gates [kg/m3]
    Scalar cloud_qc_max, cloud_nc_min, cloud_nc_max, cloud_qr_min, cloud_qr_max, cloud_nr_max;
    Scalar rain_qr_max, rain_nr_max;      // training envelope (per volume)
    Scalar embryo_mass, drops_per_embryo, ac_n_factor;
  };

  Config cfg;
  typename KT::template view_1d<Scalar> params; // all weights and biases

  // Read a file written by export_kokkos_mlp.py
  static WarmRainMLP load (const std::string& filename)
  {
    std::ifstream f(filename);
    EKAT_REQUIRE_MSG (f.good(), "[WarmRainMLP] Error! Could not open " + filename + "\n");

    auto expect = [&](const std::string& key) {
      std::string s; f >> s;
      EKAT_REQUIRE_MSG (s==key, "[WarmRainMLP] Error! Expected '" + key + "' in " + filename +
                                ", got '" + s + "'.\n");
    };
    auto act_id = [&](const std::string& a) {
      if (a=="tanh") return int(Tanh);
      if (a=="relu") return int(Relu);
      if (a=="silu") return int(Silu);
      EKAT_REQUIRE_MSG (a=="softplus", "[WarmRainMLP] Error! Unsupported activation '" + a + "'.\n");
      return int(Softplus);
    };

    WarmRainMLP mlp;
    auto& c = mlp.cfg;
    int version;
    std::string act, out_act;
    expect("p3_warm_rain_mlp"); f >> version;
    EKAT_REQUIRE_MSG (version==1, "[WarmRainMLP] Error! Unsupported file version.\n");
    expect("activation");        f >> act >> out_act;
    c.activation = act_id(act);
    c.output_activation = act_id(out_act);
    expect("widths"); f >> c.num_layers;
    EKAT_REQUIRE_MSG (c.num_layers>=1 && c.num_layers<=max_layers,
        "[WarmRainMLP] Error! Unsupported number of layers.\n");
    for (int l=0; l<=c.num_layers; ++l) {
      f >> c.width[l];
      EKAT_REQUIRE_MSG (c.width[l]>0 && c.width[l]<=max_width, "[WarmRainMLP] Error! Layer too wide.\n");
    }
    EKAT_REQUIRE_MSG (c.width[0]==n_in && c.width[c.num_layers]==n_out,
        "[WarmRainMLP] Error! The network must map 4 inputs to 4 outputs.\n");

    auto read_n = [&](const std::string& key, Scalar* v, int n) {
      expect(key);
      for (int i=0; i<n; ++i) f >> v[i];
    };
    read_n("x_mean", c.x_mean, n_in);
    read_n("x_std", c.x_std, n_in);
    read_n("floors", c.floors, n_in);
    read_n("y_scale", c.y_scale, n_out);
    read_n("y_log_std", c.y_log_std, n_out);
    Scalar gates[2], cloud[6], rain[2], nrates[3];
    read_n("gates", gates, 2);
    read_n("envelope_cloud", cloud, 6);
    read_n("envelope_rain", rain, 2);
    read_n("number_rates", nrates, 3);
    c.qc_gt = gates[0]; c.qr_gt = gates[1];
    c.cloud_qc_max = cloud[0]; c.cloud_nc_min = cloud[1]; c.cloud_nc_max = cloud[2];
    c.cloud_qr_min = cloud[3]; c.cloud_qr_max = cloud[4]; c.cloud_nr_max = cloud[5];
    c.rain_qr_max = rain[0]; c.rain_nr_max = rain[1];
    c.embryo_mass = 4.0/3.0*M_PI*1000.0*nrates[0]*nrates[0]*nrates[0];
    c.drops_per_embryo = nrates[1];
    c.ac_n_factor = nrates[2];

    std::vector<Scalar> p;
    for (int l=0; l<c.num_layers; ++l) {
      const int nw = c.width[l+1]*c.width[l];
      const int nb = c.width[l+1];
      c.weight_offset[l] = p.size();
      p.resize(p.size()+nw);
      read_n("weight", p.data()+c.weight_offset[l], nw);
      c.bias_offset[l] = p.size();
      p.resize(p.size()+nb);
      read_n("bias", p.data()+c.bias_offset[l], nb);
    }
    expect("end");
    EKAT_REQUIRE_MSG (not f.fail(), "[WarmRainMLP] Error! Could not parse " + filename + "\n");

    mlp.params = decltype(mlp.params)("warm_rain_mlp_params", p.size());
    auto params_h = Kokkos::create_mirror_view(mlp.params);
    for (size_t i=0; i<p.size(); ++i) {
      params_h(i) = p[i];
    }
    Kokkos::deep_copy(mlp.params, params_h);
    return mlp;
  }

  KOKKOS_INLINE_FUNCTION
  static Pack activate (const int act, const Pack& x)
  {
    switch (act) {
      case Tanh: return tanh(x);
      case Relu: return max(x, 0);
      case Silu: return x / (1 + exp(-x));
      default:   return max(x, 0) + log(1 + exp(-abs(x))); // softplus, without overflow
    }
  }

  // Per-volume rates (AU, AC, SC_c, SC_r) from per-volume qc, nc, qr, nr, gated
  KOKKOS_INLINE_FUNCTION
  void raw_rates (const Pack (&x)[n_in], Pack (&y)[n_out]) const
  {
    Pack a[max_width], b[max_width];
    for (int i=0; i<n_in; ++i) {
      a[i] = (log10(max(x[i], cfg.floors[i])) - cfg.x_mean[i]) / cfg.x_std[i];
    }
    for (int l=0; l<cfg.num_layers; ++l) {
      const int nin = cfg.width[l], nout = cfg.width[l+1];
      const Scalar* W = params.data() + cfg.weight_offset[l];
      const Scalar* B = params.data() + cfg.bias_offset[l];
      const int act = l==cfg.num_layers-1 ? cfg.output_activation : cfg.activation;
      for (int o=0; o<nout; ++o) {
        Pack s = B[o];
        for (int i=0; i<nin; ++i) {
          s += W[o*nin+i]*a[i];
        }
        b[o] = activate(act, s);
      }
      for (int o=0; o<nout; ++o) {
        a[o] = b[o];
      }
    }
    const auto cloud = x[0] > cfg.qc_gt;
    for (int o=0; o<n_out; ++o) {
      y[o] = expm1(a[o]*cfg.y_log_std[o])*cfg.y_scale[o];
    }
    y[0].set(!cloud, 0);
    y[1].set(!(cloud && x[2] > 0), 0);
    y[2].set(!cloud, 0);
    y[3].set(!(x[2] > cfg.qr_gt), 0);
  }

  // P3-ready grid-mean rates per kg, and masks, from P3's dry mixing ratios and density.
  // Outputs in the order of P3WarmRainRates, then use_cloud, use_rain.
  KOKKOS_INLINE_FUNCTION
  void p3_rates (const Pack& qc, const Pack& nc, const Pack& qr, const Pack& nr, const Pack& rho,
                 Pack& qc2qr_autoconv_tend, Pack& nc2nr_autoconv_tend, Pack& ncautr,
                 Pack& nc_selfcollect_tend, Pack& qc2qr_accret_tend, Pack& nc_accret_tend,
                 Pack& nr_selfcollect_tend, Pack& use_cloud, Pack& use_rain) const
  {
    const Pack x[n_in] = {qc*rho, nc*rho, qr*rho, nr*rho};
    Pack y[n_out];
    raw_rates(x, y);
    const Pack inv_rho = 1/rho;
    const Pack au = y[0]*inv_rho, ac = y[1]*inv_rho;

    qc2qr_autoconv_tend = au;
    ncautr              = au / cfg.embryo_mass;
    nc2nr_autoconv_tend = cfg.drops_per_embryo * ncautr;
    qc2qr_accret_tend   = ac;
    nc_accret_tend      = 0;
    nc_accret_tend.set(qc > 0, cfg.ac_n_factor * ac * nc / max(qc, std::numeric_limits<Scalar>::min()));
    nc_selfcollect_tend = -y[2]*inv_rho;
    nr_selfcollect_tend = y[3]*inv_rho;

    const auto in_cloud = x[0] > cfg.qc_gt && x[0] <= cfg.cloud_qc_max &&
                          x[1] >= cfg.cloud_nc_min && x[1] <= cfg.cloud_nc_max &&
                          x[2] > cfg.cloud_qr_min && x[2] <= cfg.cloud_qr_max &&
                          x[3] <= cfg.cloud_nr_max;
    const auto in_rain  = x[2] <= cfg.rain_qr_max && x[3] <= cfg.rain_nr_max;
    use_cloud = 0; use_cloud.set(in_cloud, 1);
    use_rain  = 0; use_rain.set(in_rain, 1);
  }
};

} // namespace p3
} // namespace scream

#endif // P3_WARM_RAIN_MLP_HPP
