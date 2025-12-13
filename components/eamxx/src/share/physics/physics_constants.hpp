#ifndef EAMXX_PHYSICS_CONSTANTS_HPP
#define EAMXX_PHYSICS_CONSTANTS_HPP

#include "share/core/eamxx_types.hpp"

#include <ekat_string_utils.hpp>
#include <ekat_math_utils.hpp>
#include <ekat_units.hpp>

#include <Kokkos_NumericTraits.hpp>
#include <vector>

namespace scream {
namespace physics {

/*
 * Wrapper class to store a physical constant with its value and units.
 * Provides backward compatibility through implicit conversion to Scalar.
 */
template <typename Scalar>
struct PhysicalConstant {
  Scalar value;
  ekat::units::Units unit;

  // Implicit conversion to Scalar for backward compatibility
  operator Scalar() const { return value; }

  // Default constructor
  PhysicalConstant() : value(0), unit(ekat::units::Units::invalid()) {}

  // Constructor with value and units
  PhysicalConstant(Scalar v, const ekat::units::Units& u) 
    : value(v), unit(u) {}
};

/*
 * Mathematical constants used by atmosphere processes.
 *
 * Constants are stored as PhysicalConstant objects that contain both
 * a value and units. They support three access patterns:
 *   - PC::RHO_H2O         : Implicit conversion to scalar (backward compatible)
 *   - PC::RHO_H2O.value   : Explicit access to the numeric value
 *   - PC::RHO_H2O.unit    : Access to the ekat::units::Units object
 *
 * This enables constants to be used in diagnostics, binary operations,
 * conditional sampling, and other contexts where units are needed.
 *
 * Note that a potential optimization could be to change the type of
 * Scalar constants that have integer values to int.
 */

template <typename Scalar>
struct Constants
{
  using ci_string      = ekat::CaseInsensitiveString;

  // Initialize PhysicalConstant objects with values and units
  static inline const PhysicalConstant<Scalar> Cpair = 
    PhysicalConstant<Scalar>(1004.64, ekat::units::pow(ekat::units::m, 2) / (ekat::units::pow(ekat::units::s, 2) * ekat::units::K));
  static inline const PhysicalConstant<Scalar> Rair = 
    PhysicalConstant<Scalar>(287.042, ekat::units::pow(ekat::units::m, 2) / (ekat::units::pow(ekat::units::s, 2) * ekat::units::K));
  static inline const PhysicalConstant<Scalar> RH2O = 
    PhysicalConstant<Scalar>(461.505, ekat::units::pow(ekat::units::m, 2) / (ekat::units::pow(ekat::units::s, 2) * ekat::units::K));
  static inline const PhysicalConstant<Scalar> RV = RH2O;  // Water vapor gas constant ~ J/K/kg     !461.51
  static inline const PhysicalConstant<Scalar> RHO_H2O = 
    PhysicalConstant<Scalar>(1000.0, ekat::units::kg / ekat::units::pow(ekat::units::m, 3));
  static inline const PhysicalConstant<Scalar> INV_RHO_H2O = 
    PhysicalConstant<Scalar>(1.0/1000.0, ekat::units::pow(ekat::units::m, 3) / ekat::units::kg);
  static inline const PhysicalConstant<Scalar> RhoIce = 
    PhysicalConstant<Scalar>(917.0, ekat::units::kg / ekat::units::pow(ekat::units::m, 3));  // Ice density at 0 C from Wallace+Hobbes 1977
  static inline const PhysicalConstant<Scalar> MWH2O = 
    PhysicalConstant<Scalar>(18.016, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> MWdry = 
    PhysicalConstant<Scalar>(28.966, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> o2mmr = 
    PhysicalConstant<Scalar>(0.23143, ekat::units::Units::nondimensional());  // o2 mass mixing ratio
  static inline const PhysicalConstant<Scalar> ep_2 = 
    PhysicalConstant<Scalar>(18.016/28.966, ekat::units::Units::nondimensional());  // ratio of molecular mass of water to the molecular mass of dry air !0.622
  static inline const PhysicalConstant<Scalar> gravit = 
    PhysicalConstant<Scalar>(9.80616, ekat::units::m / ekat::units::pow(ekat::units::s, 2));
  static inline const PhysicalConstant<Scalar> LatVap = 
    PhysicalConstant<Scalar>(2501000.0, ekat::units::pow(ekat::units::m, 2) / ekat::units::pow(ekat::units::s, 2));
  static inline const PhysicalConstant<Scalar> LatIce = 
    PhysicalConstant<Scalar>(333700.0, ekat::units::pow(ekat::units::m, 2) / ekat::units::pow(ekat::units::s, 2));
  static inline const PhysicalConstant<Scalar> CpLiq = 
    PhysicalConstant<Scalar>(4188.0, ekat::units::pow(ekat::units::m, 2) / (ekat::units::pow(ekat::units::s, 2) * ekat::units::K));
  static inline const PhysicalConstant<Scalar> Tmelt = 
    PhysicalConstant<Scalar>(273.15, ekat::units::K);
  static inline const PhysicalConstant<Scalar> T_zerodegc = Tmelt;
  static inline const PhysicalConstant<Scalar> T_homogfrz = 
    PhysicalConstant<Scalar>(Tmelt.value - 40, ekat::units::K);
  static inline const PhysicalConstant<Scalar> T_rainfrz = 
    PhysicalConstant<Scalar>(Tmelt.value - 4, ekat::units::K);
  static inline const PhysicalConstant<Scalar> Pi = 
    PhysicalConstant<Scalar>(3.14159265358979323, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> RHOW = RHO_H2O;
  static inline const PhysicalConstant<Scalar> INV_RHOW = INV_RHO_H2O;
  static inline const PhysicalConstant<Scalar> RHO_RIMEMIN = 
    PhysicalConstant<Scalar>(50.0, ekat::units::kg / ekat::units::pow(ekat::units::m, 3));  //Min limit for rime density [kg m-3]
  static inline const PhysicalConstant<Scalar> RHO_RIMEMAX = 
    PhysicalConstant<Scalar>(900.0, ekat::units::kg / ekat::units::pow(ekat::units::m, 3));  //Max limit for rime density [kg m-3]
  static inline const PhysicalConstant<Scalar> INV_RHO_RIMEMAX = 
    PhysicalConstant<Scalar>(1.0/RHO_RIMEMAX.value, ekat::units::pow(ekat::units::m, 3) / ekat::units::kg); // Inverse for limits for rime density [kg m-3]
  static inline const PhysicalConstant<Scalar> THIRD = 
    PhysicalConstant<Scalar>(1.0/3.0, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> SXTH = 
    PhysicalConstant<Scalar>(1.0/6.0, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> PIOV3 = 
    PhysicalConstant<Scalar>(Pi.value*THIRD.value, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> PIOV6 = 
    PhysicalConstant<Scalar>(Pi.value*SXTH.value, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> BIMM = 
    PhysicalConstant<Scalar>(2.0, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> CONS1 = 
    PhysicalConstant<Scalar>(PIOV6.value*RHOW.value, ekat::units::kg / ekat::units::pow(ekat::units::m, 3));
  static inline const PhysicalConstant<Scalar> CONS2 = 
    PhysicalConstant<Scalar>(4.*PIOV3.value*RHOW.value, ekat::units::kg / ekat::units::pow(ekat::units::m, 3));
  static inline const PhysicalConstant<Scalar> CONS3 = 
    PhysicalConstant<Scalar>(1.0/(CONS2.value*1.562500000000000e-14), ekat::units::pow(ekat::units::m, 3) / ekat::units::kg); // 1./(CONS2*pow(25.e-6,3.0));
  static inline const PhysicalConstant<Scalar> CONS5 = 
    PhysicalConstant<Scalar>(PIOV6.value*BIMM.value, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> CONS6 = 
    PhysicalConstant<Scalar>(PIOV6.value*PIOV6.value*RHOW.value*BIMM.value, ekat::units::kg / ekat::units::pow(ekat::units::m, 3));
  static inline const PhysicalConstant<Scalar> CONS7 = 
    PhysicalConstant<Scalar>(4.*PIOV3.value*RHOW.value*1.e-18, ekat::units::kg / ekat::units::pow(ekat::units::m, 3));
  static inline const PhysicalConstant<Scalar> QSMALL = 
    PhysicalConstant<Scalar>(1.e-14, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> QTENDSMALL = 
    PhysicalConstant<Scalar>(1e-20, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> BSMALL = 
    PhysicalConstant<Scalar>(1.e-15, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> NSMALL = 
    PhysicalConstant<Scalar>(1.e-16, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> ZERO = 
    PhysicalConstant<Scalar>(0.0, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> ONE = 
    PhysicalConstant<Scalar>(1.0, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> P0 = 
    PhysicalConstant<Scalar>(100000.0, ekat::units::Pa);  // reference pressure, Pa
  static inline const PhysicalConstant<Scalar> RD = Rair;  // gas constant for dry air, J/kg/K
  static inline const PhysicalConstant<Scalar> RHOSUR = 
    PhysicalConstant<Scalar>(P0.value/(RD.value*Tmelt.value), ekat::units::kg / ekat::units::pow(ekat::units::m, 3));
  static inline const PhysicalConstant<Scalar> rhosui = 
    PhysicalConstant<Scalar>(60000/(RD.value*253.15), ekat::units::kg / ekat::units::pow(ekat::units::m, 3));
  static inline const PhysicalConstant<Scalar> RHO_1000MB = 
    PhysicalConstant<Scalar>(P0.value/(RD.value*Tmelt.value), ekat::units::kg / ekat::units::pow(ekat::units::m, 3));
  static inline const PhysicalConstant<Scalar> RHO_600MB = 
    PhysicalConstant<Scalar>(60000/(RD.value*253.15), ekat::units::kg / ekat::units::pow(ekat::units::m, 3));
  static inline const PhysicalConstant<Scalar> CP = Cpair;  // heat constant of air at constant pressure, J/kg
  static inline const PhysicalConstant<Scalar> INV_CP = 
    PhysicalConstant<Scalar>(1.0/CP.value, ekat::units::pow(ekat::units::s, 2) * ekat::units::K / ekat::units::pow(ekat::units::m, 2));
  //  static constexpr Scalar Tol           = ekat::is_single_precision<Real>::value ? 2e-5 : 1e-14;
  static inline const PhysicalConstant<Scalar> macheps = 
    PhysicalConstant<Scalar>(std::numeric_limits<Real>::epsilon(), ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> dt_left_tol = 
    PhysicalConstant<Scalar>(1.e-4, ekat::units::s);
  static inline const PhysicalConstant<Scalar> bcn = 
    PhysicalConstant<Scalar>(2., ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> dropmass = 
    PhysicalConstant<Scalar>(5.2e-7, ekat::units::kg);
  static inline const PhysicalConstant<Scalar> NCCNST = 
    PhysicalConstant<Scalar>(200.0e+6, ekat::units::Units::nondimensional() / ekat::units::pow(ekat::units::m, 3));
  static inline const PhysicalConstant<Scalar> incloud_limit = 
    PhysicalConstant<Scalar>(5.1e-3, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> precip_limit = 
    PhysicalConstant<Scalar>(1.0e-2, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> Karman = 
    PhysicalConstant<Scalar>(0.4, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> Avogad = 
    PhysicalConstant<Scalar>(6.02214e26, ekat::units::Units::nondimensional() / ekat::units::mol);
  static inline const PhysicalConstant<Scalar> Boltz = 
    PhysicalConstant<Scalar>(1.38065e-23, ekat::units::kg * ekat::units::pow(ekat::units::m, 2) / (ekat::units::pow(ekat::units::s, 2) * ekat::units::K));
  static inline const PhysicalConstant<Scalar> Rgas = 
    PhysicalConstant<Scalar>(Avogad.value * Boltz.value, ekat::units::kg * ekat::units::pow(ekat::units::m, 2) / (ekat::units::pow(ekat::units::s, 2) * ekat::units::mol * ekat::units::K));
  static inline const PhysicalConstant<Scalar> MWWV = MWH2O;
  static inline const PhysicalConstant<Scalar> RWV = 
    PhysicalConstant<Scalar>(Rgas.value / MWWV.value, ekat::units::pow(ekat::units::m, 2) / (ekat::units::pow(ekat::units::s, 2) * ekat::units::K));
  static inline const PhysicalConstant<Scalar> ZVIR = 
    PhysicalConstant<Scalar>(RWV.value / Rair.value - 1.0, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> f1r = 
    PhysicalConstant<Scalar>(0.78, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> f2r = 
    PhysicalConstant<Scalar>(0.32, ekat::units::Units::nondimensional());
  static inline const PhysicalConstant<Scalar> nmltratio = 
    PhysicalConstant<Scalar>(1.0, ekat::units::Units::nondimensional()); // ratio of rain number produced to ice number loss from melting
  static inline const PhysicalConstant<Scalar> basetemp = 
    PhysicalConstant<Scalar>(300.0, ekat::units::K);
  static inline const PhysicalConstant<Scalar> r_earth = 
    PhysicalConstant<Scalar>(6.376e6, ekat::units::m); // Radius of the earth in m
  static inline const PhysicalConstant<Scalar> stebol = 
    PhysicalConstant<Scalar>(5.670374419e-8, ekat::units::kg / (ekat::units::pow(ekat::units::s, 3) * ekat::units::pow(ekat::units::K, 4))); // Stefan-Boltzmann's constant (W/m^2/K^4)
  static inline const PhysicalConstant<Scalar> omega = 
    PhysicalConstant<Scalar>(7.292e-5, ekat::units::Units::nondimensional() / ekat::units::s); // Earth's rotation (rad/sec)

  // Table dimension constants
  static constexpr int VTABLE_DIM0    = 300;
  static constexpr int VTABLE_DIM1    = 10;
  static constexpr int MU_R_TABLE_DIM = 150;

  // Turbulent Mountain Stress constants
  static inline const PhysicalConstant<Scalar> orocnst = 
    PhysicalConstant<Scalar>(1, ekat::units::Units::nondimensional());     // Converts from standard deviation to height [ no unit ]
  static inline const PhysicalConstant<Scalar> z0fac = 
    PhysicalConstant<Scalar>(0.075, ekat::units::Units::nondimensional()); // Factor determining z_0 from orographic standard deviation [ no unit ]

  // switch for warm-rain parameterization
  // = 1 Seifert and Beheng 2001
  // = 2 Beheng 1994
  // = 3 Khairoutdinov and Kogan 2000
  static constexpr int IPARAM         = 3;

  // Gases
  static Scalar get_gas_mol_weight(ci_string gas_name);

  // For use in converting area to length for a column cell
  // World Geodetic System 1984 (WGS84)
  static inline const PhysicalConstant<Scalar> earth_ellipsoid1 = 
    PhysicalConstant<Scalar>(111132.92, ekat::units::m); // first coefficient, meters per degree longitude at equator
  static inline const PhysicalConstant<Scalar> earth_ellipsoid2 = 
    PhysicalConstant<Scalar>(559.82, ekat::units::m);    // second expansion coefficient for WGS84 ellipsoid
  static inline const PhysicalConstant<Scalar> earth_ellipsoid3 = 
    PhysicalConstant<Scalar>(1.175, ekat::units::m);     // third expansion coefficient for WGS84 ellipsoid
};

// Gases
// Define the molecular weight for each gas, which can then be
// used to determine the volume mixing ratio for each gas.
template <typename Scalar>
Scalar Constants<Scalar>::get_gas_mol_weight(ci_string gas_name) {
  //TODO: Possible improvement would be to design a device friendly function
  if        (gas_name == "h2o") {
    return Scalar(Constants<Scalar>::MWH2O.value);
  } else if (gas_name == "co2") {
    return 44.0095;
  } else if (gas_name == "o3" ) {
    return 47.9982;
  } else if (gas_name == "n2o") {
    return 44.0128;
  } else if (gas_name == "co" ) {
    return 28.0101;
  } else if (gas_name == "ch4") {
    return 16.04246;
  } else if (gas_name == "o2" ) {
    return 31.998;
  } else if (gas_name == "n2" ) {
    return 28.0134;
  } else if (gas_name == "cfc11" ) {
    return 136.;
  } else if (gas_name == "cfc12" ) {
    return 120.;
  }
  return ekat::invalid<Scalar>();
}

} // namespace physics
} // namespace scream

#endif // EAMXX_PHYSICS_CONSTANTS_HPP
