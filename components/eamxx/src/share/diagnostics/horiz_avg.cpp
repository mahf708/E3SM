#include "horiz_avg.hpp"
#include "contraction_utils.hpp"

#include "share/field/field_utils.hpp"
#include "share/util/eamxx_universal_constants.hpp"

namespace scream {

HorizAvg::
HorizAvg(const ekat::Comm &comm,
             const ekat::ParameterList &params,
             const std::shared_ptr<const AbstractGrid>& grid)
 : AbstractDiagnostic(comm, params, grid)
{
  EKAT_REQUIRE_MSG (m_params.isParameter("field_name"),
      "[HorizAvg] Error! Missing required param 'field_name'\n");

  const auto &fn = m_params.get<std::string>("field_name");
  m_field_in_names.push_back(fn);

  m_contract_method = m_params.get<std::string>("contract_method","avg");
  EKAT_REQUIRE_MSG(
      m_contract_method == "avg" || m_contract_method == "sum" ||
      m_contract_method == "max" || m_contract_method == "min" ||
      m_contract_method == "var" || m_contract_method == "std",
      "Error! HorizAvg only supports 'avg', 'sum', 'max', 'min', 'var', or 'std' as contract_method.\n"
      " - contract_method: " + m_contract_method + "\n");

  // Averages (avg, var, std) are area weighted. The others get an all-ones
  // weight in initialize_impl, since we do not have field layouts yet.
  if (is_average()) {
    m_area = m_grid->get_geometry_data("area");
  }
}

void HorizAvg::initialize_impl()
{
  using namespace ShortFieldTagsNames;

  const auto &f      = m_fields_in.at(m_field_in_names.front());
  const auto &fid    = f.get_header().get_identifier();
  const auto &layout = fid.get_layout();

  EKAT_REQUIRE_MSG(layout.rank() >= 1 && layout.rank() <= 3,
      "Error! Field rank not supported by HorizAvg.\n"
      " - field name: " + fid.name() + "\n"
      " - field layout: " + layout.to_string() + "\n");
  EKAT_REQUIRE_MSG(layout.tags()[0] == COL,
      "Error! HorizAvg diagnostic expects a layout starting with the 'COL' tag.\n"
      " - field name  : " + fid.name() + "\n"
      " - field layout: " + layout.to_string() + "\n");

  const auto &fname = m_params.get<std::string>("field_name");
  const auto dname  = fname + "_horiz_" + m_contract_method;

  const auto units = m_contract_method == "var" ? fid.get_units()*fid.get_units() : fid.get_units();
  auto d_fid = fid.clone(dname).reset_layout(layout.clone().strip_dim(COL)).reset_units(units);
  m_diagnostic_output = Field(d_fid,true);

  if (not is_average()) {
    // Unweighted: an all-ones weight
    FieldIdentifier w_fid("ones", m_grid->get_2d_scalar_layout(), ekat::units::none,
                          m_grid->name(), f.data_type());
    m_area = Field(w_fid,true);
    m_area.deep_copy(1);
  }

  if (m_contract_method == "var" || m_contract_method == "std") {
    // Scratch for the squared deviations from the mean, masked like f
    m_dev = f.clone("deviation");
    if (f.has_valid_mask()) {
      m_dev.set_valid_mask(f.get_valid_mask());
    }
  }

  if (m_contract_method == "sum") {
    // A sum over no valid column is just 0, so there is nothing else to do
    return;
  }

  if (f.has_valid_mask()) {
    // Output valid_mask: 1 where den > 0 (i.e., at least one valid column)
    m_diagnostic_output.create_valid_mask(Field::MaskInit::None);
    m_diagnostic_output.get_header().set_may_be_filled(true);

    // To use at runtime to compute sum(1*w) (with f_in=1). For max/min, w=1,
    // so this counts valid columns, to detect where none is valid.
    m_ones = f.clone();
    m_ones.deep_copy(1);
    m_ones.set_valid_mask(f.get_valid_mask());

    m_denom = m_diagnostic_output.clone("denom");
  } else if (is_average()) {
    // Since area is constant and there is no masking, we can pre-compute
    // the area sum, and set weight = area/sum(area), so that we avoid
    // one loop at runtime.
    // NOTE: clone m_area, since the field from the grid is read-only
    m_area = m_area.clone("scaled_area", CloneFlags::CopyData);

    m_denom = m_area.subfield(COL,0).clone(); // This should create a 0d field
    m_ones = m_area.clone("ones");
    m_ones.deep_copy(1);

    horiz_contraction(m_denom, m_ones, m_area, m_comm);
    m_denom.sync_to_host();
    auto d = m_denom.get_internal_view_data<Real,Host>()[0];
    m_area.scale(1/d);

    // Free up the memory
    m_ones = Field();
    m_denom = Field();
  }
}

void HorizAvg::compute_impl()
{
  const auto &f = m_fields_in.at(m_field_in_names.front());
  auto& d = m_diagnostic_output;

  if (f.has_valid_mask() && m_contract_method != "sum") {
    // Denominator: sum(weight) masked. Its support is the output's valid mask.
    horiz_contraction(m_denom, m_ones, m_area, m_comm);
    compute_mask(m_denom,0,Comparison::NE,d.get_valid_mask());
  }

  if (not is_average()) {
    const auto op = m_contract_method == "max" ? ContractionOp::Max
                  : m_contract_method == "min" ? ContractionOp::Min
                                               : ContractionOp::Sum;
    horiz_contraction(d, f, m_area, m_comm, op);
    if (d.has_valid_mask()) {
      // Where no column is valid, d holds the op's identity
      d.deep_copy(constants::fill_value<Real>,d.get_valid_mask(),true);
    }
    return;
  }

  // The rest are weighted averages: of f itself for avg, and of its squared
  // deviation from that average for var/std (a two-pass algorithm, which
  // does not suffer the cancellation of computing avg(f^2)-avg(f)^2).
  auto weighted_avg = [&](const Field& src) {
    // sum(w * src) possibly masked. Without a mask, w is pre-normalized.
    horiz_contraction(d, src, m_area, m_comm);
    if (d.has_valid_mask()) {
      auto& valid_out = d.get_valid_mask();
      d.scale_inv(m_denom,valid_out);
      // IO relies on fill_value for masked-out entries
      d.deep_copy(constants::fill_value<Real>,valid_out,true);
    }
  };

  weighted_avg(f);
  if (m_contract_method == "avg") {
    return;
  }

  // dev = (f - avg)^2. Where no column is valid, avg is fill_value, but there
  // dev is masked out anyway.
  const auto& layout = f.get_header().get_identifier().get_layout();
  m_dev.deep_copy(f);
  m_dev.update(d.broadcast_to(layout), -1, 1);
  m_dev.scale(m_dev);
  weighted_avg(m_dev);

  if (m_contract_method == "std") {
    sqrt_in_place(d);
  }
}

}  // namespace scream
