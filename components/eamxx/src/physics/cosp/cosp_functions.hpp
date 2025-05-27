#ifndef SCREAM_COSP_FUNCTIONS_HPP
#define SCREAM_COSP_FUNCTIONS_HPP
#include "share/eamxx_types.hpp"
using scream::Real;
extern "C" void cosp_c2f_init(int ncol, int nsubcol, int nlay);
extern "C" void cosp_c2f_final();
extern "C" void cosp_c2f_run(const int ncol, const int nsubcol, const int nlay, const int ntau, const int nctp, const int ncth,
    const int nLWP, const int nIWP, const int nReffLiq, const int nReffIce, 
    const Real emsfc_lw, const Real* sunlit, const Real* skt,
    const Real* T_mid, const Real* p_mid, const Real* p_int, const Real* z_mid, const Real* qv, const Real* qc, const Real* qi,
    const Real* cldfrac, const Real* reff_qc, const Real* reff_qi, const Real* dtau067, const Real* dtau105,
    Real* isccp_cldtot, Real* isccp_ctptau, Real* modis_ctptau, Real* misr_cthtau, 
    Real* modis_ctptau_liq, Real* modis_ctptau_ice, Real* modis_lwpre, Real* modis_iwpre,
    Real* modis_cldtot, Real* modis_clwtot, Real* modis_clitot, 
    Real* modis_taut, Real* modis_tauw, Real* modis_taui, 
    Real* modis_reffw, Real* modis_reffi, Real* modis_lwp, Real* modis_iwp,
    Real* modis_cld_Q06, Real* modis_nd_Q06, Real* modis_lwp_Q06, Real* modis_tau_Q06, Real* modis_reff_Q06, 
    Real* modis_cld_ALL, Real* modis_nd_ALL, Real* modis_lwp_ALL, Real* modis_tau_ALL, Real* modis_reff_ALL
    );

namespace scream {

    namespace CospFunc {
        using lview_host_1d = typename ekat::KokkosTypes<HostDevice>::template lview<Real*  >;
        using lview_host_2d = typename ekat::KokkosTypes<HostDevice>::template lview<Real** >;
        using lview_host_3d = typename ekat::KokkosTypes<HostDevice>::template lview<Real***>;
        template <typename S>
        using view_1d = typename ekat::KokkosTypes<HostDevice>::template view_1d<S>;
        template <typename S>
        using view_2d = typename ekat::KokkosTypes<HostDevice>::template view_2d<S>;
        template <typename S>
        using view_3d = typename ekat::KokkosTypes<HostDevice>::template view_3d<S>;

        inline void initialize(int ncol, int nsubcol, int nlay) {
            cosp_c2f_init(ncol, nsubcol, nlay);
        };
        inline void finalize() {
            cosp_c2f_final();
        };
        inline void main(
                const Int ncol, const Int nsubcol, const Int nlay, const Int ntau, const Int nctp, const Int ncth, 
                const Int nLWP, const Int nIWP, const Int nReffLiq, const Int nReffIce, 
                const Real emsfc_lw,
                const view_1d<const Real>& sunlit , const view_1d<const Real>& skt,
                const view_2d<const Real>& T_mid  , const view_2d<const Real>& p_mid  ,
                const view_2d<const Real>& p_int,  const view_2d<const Real>& z_mid,
                const view_2d<const Real>& qv     , const view_2d<const Real>& qc,
                const view_2d<const Real>& qi, const view_2d<const Real>& cldfrac,
                const view_2d<const Real>& reff_qc, const view_2d<const Real>& reff_qi,
                const view_2d<const Real>& dtau067, const view_2d<const Real>& dtau105,
                const view_1d<Real>& isccp_cldtot , const view_3d<Real>& isccp_ctptau,
                const view_3d<Real>& modis_ctptau, const view_3d<Real>& misr_cthtau,
                const view_3d<Real>& modis_ctptau_liq, const view_3d<Real>& modis_ctptau_ice, const view_3d<Real>& modis_lwpre, const view_3d<Real>& modis_iwpre,
                const view_1d<Real>& modis_cldtot, const view_1d<Real>& modis_clwtot, const view_1d<Real>& modis_clitot, 
                const view_1d<Real>& modis_taut, const view_1d<Real>& modis_tauw, const view_1d<Real>& modis_taui, 
                const view_1d<Real>& modis_reffw, const view_1d<Real>& modis_reffi, const view_1d<Real>& modis_lwp, const view_1d<Real>&modis_iwp,
                const view_1d<Real>& modis_cld_Q06, const view_1d<Real>& modis_nd_Q06, const view_1d<Real>& modis_lwp_Q06, const view_1d<Real>& modis_tau_Q06, const view_1d<Real>& modis_reff_Q06,
                const view_1d<Real>& modis_cld_ALL, const view_1d<Real>& modis_nd_ALL, const view_1d<Real>& modis_lwp_ALL, const view_1d<Real>& modis_tau_ALL, const view_1d<Real>& modis_reff_ALL
                ) {

            // Make host copies and permute data as needed
            lview_host_2d
                  T_mid_h("T_mid_h", ncol, nlay), p_mid_h("p_mid_h", ncol, nlay), p_int_h("p_int_h", ncol, nlay+1),
                  z_mid_h("z_mid_h", ncol, nlay), qv_h("qv_h", ncol, nlay), qc_h("qc_h", ncol, nlay), qi_h("qi_h", ncol, nlay),
                  cldfrac_h("cldfrac_h", ncol, nlay),
                  reff_qc_h("reff_qc_h", ncol, nlay), reff_qi_h("reff_qi_h", ncol, nlay),
                  dtau067_h("dtau_067_h", ncol, nlay), dtau105_h("dtau105_h", ncol, nlay);
            lview_host_3d isccp_ctptau_h("isccp_ctptau_h", ncol, ntau, nctp);
            lview_host_3d modis_ctptau_h("modis_ctptau_h", ncol, ntau, nctp);
            lview_host_3d misr_cthtau_h("misr_cthtau_h", ncol, ntau, ncth);
            lview_host_3d modis_ctptau_liq_h("modis_ctptau_liq_h", ncol, ntau, nctp);
            lview_host_3d modis_ctptau_ice_h("modis_ctptau_ice_h", ncol, ntau, nctp);
            lview_host_3d modis_lwpre_h("modis_lwpre_h", ncol, nLWP, nReffLiq);
            lview_host_3d modis_iwpre_h("modis_iwpre_h", ncol, nIWP, nReffIce);

            // Copy to layoutLeft host views
            for (int i = 0; i < ncol; i++) {
                for (int j = 0; j < nlay; j++) {
                    T_mid_h(i,j) = T_mid(i,j);
                    p_mid_h(i,j) = p_mid(i,j);
                    z_mid_h(i,j) = z_mid(i,j);
                    qv_h(i,j) = qv(i,j);
                    qc_h(i,j) = qc(i,j);
                    qi_h(i,j) = qi(i,j);
                    cldfrac_h(i,j) = cldfrac(i,j);
                    reff_qc_h(i,j) = reff_qc(i,j);
                    reff_qi_h(i,j) = reff_qi(i,j);
                    dtau067_h(i,j) = dtau067(i,j);
                    dtau105_h(i,j) = dtau105(i,j);
                }
            }
            for (int i = 0; i < ncol; i++) {
                for (int j = 0; j < nlay+1; j++) {
                    p_int_h(i,j) = p_int(i,j);
                }
            }

            // Subsample here?

            // Call COSP wrapper
            cosp_c2f_run(ncol, nsubcol, nlay, ntau, nctp, ncth,
                    nLWP, nIWP, nReffLiq, nReffIce, 
                    emsfc_lw, sunlit.data(), skt.data(), T_mid_h.data(), p_mid_h.data(), p_int_h.data(),
                    z_mid_h.data(), qv_h.data(), qc_h.data(), qi_h.data(),
                    cldfrac_h.data(), reff_qc_h.data(), reff_qi_h.data(), dtau067_h.data(), dtau105_h.data(),
                    isccp_cldtot.data(), isccp_ctptau_h.data(), modis_ctptau_h.data(), misr_cthtau_h.data(),
                    modis_ctptau_liq_h.data(), modis_ctptau_ice_h.data(), modis_lwpre_h.data(), modis_iwpre_h.data(),
                    modis_cldtot.data(), modis_clwtot.data(), modis_clitot.data(), 
                    modis_taut.data(), modis_tauw.data(), modis_taui.data(), 
                    modis_reffw.data(), modis_reffi.data(), modis_lwp.data(), modis_iwp.data(),
                    modis_cld_Q06.data(), modis_nd_Q06.data(), modis_lwp_Q06.data(), modis_tau_Q06.data(), modis_reff_Q06.data(),
                    modis_cld_ALL.data(), modis_nd_ALL.data(), modis_lwp_ALL.data(), modis_tau_ALL.data(), modis_reff_ALL.data()
                    );

            // Copy outputs back to layoutRight views
            for (int i = 0; i < ncol; i++) {
                for (int j = 0; j < ntau; j++) {
                    for (int k = 0; k < nctp; k++) {
                        isccp_ctptau(i,j,k) = isccp_ctptau_h(i,j,k);
                        modis_ctptau(i,j,k) = modis_ctptau_h(i,j,k);
                        modis_ctptau_liq(i,j,k) = modis_ctptau_liq_h(i,j,k);
                        modis_ctptau_ice(i,j,k) = modis_ctptau_ice_h(i,j,k);
                    }
                    for (int k = 0; k < ncth; k++) {
                        misr_cthtau(i,j,k) = misr_cthtau_h(i,j,k);
                    }
                }
            }
            for (int i = 0; i < ncol; i++) {
                for (int j = 0; j < nLWP; j++) {
                    for (int k = 0; k < nReffLiq; k++) {
                        modis_lwpre(i,j,k) = modis_lwpre_h(i,j,k);
                    }
                }
            }
            for (int i = 0; i < ncol; i++) {
                for (int j = 0; j < nIWP; j++) {
                    for (int k = 0; k < nReffIce; k++) {
                        modis_iwpre(i,j,k) = modis_iwpre_h(i,j,k);
                    }
                }
            }
        }
    }
}
#endif  /* SCREAM_COSP_FUNCTIONS_HPP */
