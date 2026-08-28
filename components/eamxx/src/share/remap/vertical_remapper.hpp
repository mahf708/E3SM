#ifndef EAMXX_VERTICAL_REMAPPER_HPP
#define EAMXX_VERTICAL_REMAPPER_HPP

#include "share/remap/abstract_remapper.hpp"

#include <ekat_lin_interp.hpp>

namespace scream
{

/*
 * A remapper to interpolate fields on a separate vertical grid
 */

class VerticalRemapper : public AbstractRemapper
{
public:
  enum ExtrapType {
    Mask,  // Use fixed value
    P0     // Constant extrapolation
  };

  enum TopBot {
    Top = 1,
    Bot = 2,
    TopAndBot = Top | Bot
  };

  // How a field's values relate to the vertical grid it is defined on:
  //  - Intensive: the value is a point-wise sample at the layer midpoint
  //    (e.g. a mixing ratio, a single scattering albedo). Remapping it means
  //    interpolating it at the target midpoints.
  //  - Extensive: the value is the integral of some density over the layer
  //    (e.g. a layer aerosol optical depth, which is the extinction integrated
  //    through the layer). Such a quantity is proportional to the layer
  //    thickness, so interpolating it point-wise would rescale the column sum
  //    by the ratio of the src/tgt layer thicknesses. These fields are instead
  //    remapped conservatively, so that the column sum is preserved.
  enum RemapKind {
    Intensive,
    Extensive
  };

  VerticalRemapper (const grid_ptr_type& src_grid,
                    const std::string& map_file);

  VerticalRemapper (const grid_ptr_type& src_grid,
                    const grid_ptr_type& tgt_grid);

  ~VerticalRemapper () = default;

  void set_extrapolation_type (const ExtrapType etype, const TopBot where = TopAndBot);

  // Declare how a field must be remapped. Must be called BEFORE registration_ends().
  // Only fields defined at layer midpoints can be marked as Extensive.
  void set_remap_kind (const std::string& field_name, const RemapKind kind);

  void set_source_pressure (const Field& p);
  void set_target_pressure (const Field& p);

  // This method simply creates the tgt grid from a map file
  static std::shared_ptr<AbstractGrid>
  create_tgt_grid (const grid_ptr_type& src_grid, const std::string& map_file);

  bool compatible_layouts (const FieldLayout& src, const FieldLayout& tgt) const override;

  bool is_valid_tgt_layout (const FieldLayout& layout) const override;
  bool is_valid_src_layout (const FieldLayout& layout) const override;
protected:

  void set_pressure (const Field& p, const std::string& src_or_tgt);

  FieldLayout create_layout (const FieldLayout& from_layout,
                             const std::shared_ptr<const AbstractGrid>& to_grid) const override;

  void registration_ends_impl () override;

  void remap_fwd_impl () override;

#ifdef KOKKOS_ENABLE_CUDA
public:
#endif

  template<int N>
  void apply_vertical_interpolation (const ekat::LinInterp<Real,N>& lin_interp,
                                     const Field& f_src, const Field& f_tgt,
                                     const Field& p_src, const Field& p_tgt) const;

  void extrapolate (const Field& f_src, const Field& f_tgt,
                    const Field& p_src, const Field& p_tgt) const;

  void apply_conservative_remap (const Field& f_src, const Field& f_tgt,
                                 const Field& p_src, const Field& p_tgt) const;

  template<int N>
  void setup_lin_interp (const ekat::LinInterp<Real,N>& lin_interp,
                         const Field& p_src, const Field& p_tgt) const;
protected:

  void create_lin_interp ();

  using KT = KokkosTypes<DefaultDevice>;

  template<typename T>
  using view_1d = typename KT::template view_1d<T>;
  template<typename T>
  using view_2d = typename KT::template view_2d<T>;

  ekat::Comm            m_comm;

  // Tgt grid masks (in case extrap type at top or bot is Mask)
  std::map<std::string,Field>    m_masks;

  // Vertical profile fields, both for source and target.
  std::map<FieldTag,Field> m_src_pressure;
  std::map<FieldTag,Field> m_tgt_pressure;

  // If user provides pressure profiles that are NOT compatible with SCREAM_PACK_SIZE,
  // we will set these booleans to false, and use ONLY the "scalar" LinInterp structures
  std::map<FieldTag, bool> m_packs_supported;

  // Extrapolation settings at top/bottom. Default to P0 extrapolation
  ExtrapType            m_etype_top = P0;
  ExtrapType            m_etype_bot = P0;

  // Small struct holding metadata for a field's vertical remapping:
  //  - packs_supported: true if both field and pressure data allow SIMD packing.
  //  - src_vtag/tgt_vtag: the vertical FieldTag for source/target (Invalid for 2D fields).
  //  - li_vtag: used to select the correct LinInterp object. To ensure midpoints (LEV)
  //    and interfaces (ILEV) fields (if present) use separate LinInterp objects,
  //    this tag follows the "most specific" vertical identity available:
  //      1. If either src or tgt field has LEV or ILEV, li_vtag takes that tag.
  //      2. If neither field has it (i.e., both src and tgt grids have vkind=Pressure),
  //         li_vtag defaults to LEVP.
  //  - kind: whether the field is remapped point-wise or conservatively.
  struct FType {
    bool packs_supported = false;
    FieldTag li_vtag  = FieldTag::Invalid;
    FieldTag src_vtag = FieldTag::Invalid;
    FieldTag tgt_vtag = FieldTag::Invalid;
    RemapKind kind = Intensive;
  };
  std::map<std::string,FType> m_field2type;

  // Remap kind requested by the user, keyed by field name. Fields missing
  // from this map are remapped as Intensive (the default).
  std::map<std::string,RemapKind> m_remap_kind;

  // Maps to store the interpolation operators, keyed by the logical 'li_vtag'.
  // We maintain separate packed and scalar variants to maximize SIMD usage.
  // See FType::li_vtag for how the FieldTag key is determined in mixed-grid cases.
  std::map<FieldTag, ekat::LinInterp<Real,SCREAM_PACK_SIZE>> m_lin_interp_packed;
  std::map<FieldTag, ekat::LinInterp<Real,1>>                m_lin_interp_scalar;
};

} // namespace scream

#endif // EAMXX_VERTICAL_REMAPPER_HPP
