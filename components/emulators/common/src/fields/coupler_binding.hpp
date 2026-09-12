/**
 * @file coupler_binding.hpp
 * @brief The seam between a component's FieldSet and an MCT attribute vector.
 */

#ifndef E3SM_EMULATOR_FIELDS_COUPLER_BINDING_HPP
#define E3SM_EMULATOR_FIELDS_COUPLER_BINDING_HPP

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "field_list.hpp"
#include "field_set.hpp"
#include "mask_set.hpp"

namespace emulator {
namespace fields {

/**
 * @brief A view of an attribute vector's real data, as the coupler lays it out.
 *
 * MCT stores `rAttr(nfields, npoints)` in Fortran order, so what C receives
 * from `c_loc(av%rAttr(1,1))` is point-major: field `f` at point `p` is
 * `data[p * nfields + f]`.  Reading it as `data[f * npoints + p]` -- the
 * natural guess -- is silently wrong and produces no NaN.  This class is
 * the one place that arithmetic is written down.
 *
 * The view is checked against the coupler's own field count at construction:
 * a FieldList parsed from a truncated string (Fortran copying a long
 * `seq_flds_*_fields` into a short buffer) disagrees with `nRattr`, and that
 * is an error here rather than a quietly shifted field later.
 */
class AttrVectView {
public:
  /**
   * @param data     `c_loc(av%rAttr(1,1))`; may be null only if `npoints` is 0
   * @param layout   the attribute vector's field names, in order
   * @param nfields  `mct_aVect_nRattr(av)`, as the coupler reports it
   * @param npoints  `mct_aVect_lsize(av)`
   * @throws std::invalid_argument if `layout` does not have `nfields` names,
   *         or on a null buffer with points in it
   */
  AttrVectView(double *data, const FieldList &layout, std::size_t nfields,
               std::size_t npoints);

  const FieldList &layout() const { return *m_layout; }
  std::size_t nfields() const { return m_nfields; }
  std::size_t npoints() const { return m_npoints; }

  double &at(std::size_t field, std::size_t point) {
    return m_data[point * m_nfields + field];
  }
  double at(std::size_t field, std::size_t point) const {
    return m_data[point * m_nfields + field];
  }

  /// Copy one field out into a contiguous array of npoints() values.
  void read(std::size_t field, std::span<double> dst) const;
  /// Copy a contiguous array of npoints() values into one field.
  void write(std::size_t field, std::span<const double> src);
  /// Set one field to a single value at every point.
  void fill(std::size_t field, double value);

private:
  double *m_data = nullptr;
  const FieldList *m_layout = nullptr;
  std::size_t m_nfields = 0;
  std::size_t m_npoints = 0;
};

/// Whether a component can run without a coupler field.
enum class Need {
  Required, ///< Missing from the coupler's list is an error at bind time.
  Optional  ///< Missing is allowed: the component's field keeps its fill.
};

/// One coupler field a component reads or writes.
struct FieldSpec {
  std::string name; ///< The coupler's name for it, e.g. "Sa_tbot"
  Need need = Need::Required;
  /// Units, e.g. "K".  Unused by the MCT seam; the C++ coupler's registry
  /// refuses a field without them (see RegistryAdapter).
  std::string units = {};
  /// Exports only: the MaskSet entry bounding this field.  Outside it the
  /// coupler receives zero, whatever the component's array holds there.
  std::string mask = {};
};

/**
 * @brief Which of a component's fields go to or come from which coupler rows.
 *
 * Built once, when the coupler's field lists are known, and then used every
 * step.  All the name checking happens at construction, so the per-step
 * copies are index walks with nothing to fail except a changed point count.
 *
 * The rules, and the failures they exist to prevent:
 *
 *  - **Import.** Every Required field must be in the coupler's list.  A
 *    missing one fails at bind time and names its near misses; otherwise the
 *    component would run on its initial fill.  Coupler fields no component
 *    field asks for are ignored, and listed.
 *
 *  - **Export.** Every Required field must be in the coupler's list: a
 *    component producing a field the coupler does not carry has been paired
 *    with the wrong compset.  Coupler fields the component does not produce
 *    are written as exactly zero on every push -- the coupler would see zero
 *    anyway on the first step, and on later steps it would see whatever was
 *    left there -- and are listed in the report, so "the coupler receives
 *    zero for Faxa_swvdr" is something a log says rather than something an
 *    energy budget eventually reveals.
 *
 * An export spec may name a mask (FieldSpec::mask); the mask must be in the
 * MaskSet passed at construction, and push() writes zero outside it.  An
 * import spec naming a mask is refused: what the coupler sends is its own
 * business.
 *
 * Every field named in `specs` must already be in the FieldSet.
 */
class CouplerBinding {
public:
  enum class Direction { Import, Export };

  CouplerBinding(Direction direction, FieldSet &fields,
                 const std::vector<FieldSpec> &specs,
                 const FieldList &coupler_fields,
                 const MaskSet *masks = nullptr);

  /// Coupler to component.  Only valid for an Import binding.
  void pull(const AttrVectView &coupler);
  /// Component to coupler.  Only valid for an Export binding.
  void push(AttrVectView &coupler) const;

  Direction direction() const { return m_direction; }

  /// Fields exchanged with the coupler, in the order of `specs`.
  const std::vector<std::string> &bound() const { return m_bound; }
  /// Optional component fields the coupler does not carry.
  const std::vector<std::string> &absent() const { return m_absent; }
  /// Coupler fields no component field is bound to: ignored on import,
  /// zeroed on export.
  const std::vector<std::string> &unbound() const { return m_unbound; }

  /// A few lines for a component log: what is bound, absent and unbound.
  std::string summary() const;

private:
  struct Pair {
    std::span<double> field;
    std::size_t row;
    std::span<const double> mask; ///< empty: unmasked
  };

  void check_view(const AttrVectView &coupler) const;

  Direction m_direction;
  FieldSet *m_fields;
  const FieldList *m_coupler_fields;
  std::vector<Pair> m_pairs;
  std::vector<std::size_t> m_zero_rows;
  std::vector<std::string> m_bound;
  std::vector<std::string> m_absent;
  std::vector<std::string> m_unbound;
  std::vector<std::string> m_masked; ///< "field (mask)" for the summary
};

} // namespace fields
} // namespace emulator

#endif // E3SM_EMULATOR_FIELDS_COUPLER_BINDING_HPP
