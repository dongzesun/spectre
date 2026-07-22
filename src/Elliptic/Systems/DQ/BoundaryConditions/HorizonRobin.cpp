// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/DQ/BoundaryConditions/HorizonRobin.hpp"

#include <cmath>
#include <cstddef>
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace DQ::BoundaryConditions {
namespace {

template <size_t Dim>
double determinant4_at_point(
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& metric,
    const size_t point) {
  static_assert(Dim == 3);
  double m[4][4]{};
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < 4; ++b) {
      m[a][b] = metric.get(a, b)[point];
    }
  }
  double det = 0.0;
  for (size_t col = 0; col < 4; ++col) {
    double minor[3][3]{};
    for (size_t i = 1; i < 4; ++i) {
      size_t minor_col = 0;
      for (size_t j = 0; j < 4; ++j) {
        if (j == col) {
          continue;
        }
        minor[i - 1][minor_col] = m[i][j];
        ++minor_col;
      }
    }
    const double minor_det =
        minor[0][0] * (minor[1][1] * minor[2][2] - minor[1][2] * minor[2][1]) -
        minor[0][1] * (minor[1][0] * minor[2][2] - minor[1][2] * minor[2][0]) +
        minor[0][2] * (minor[1][0] * minor[2][1] - minor[1][1] * minor[2][0]);
    det += (col % 2 == 0 ? 1.0 : -1.0) * m[0][col] * minor_det;
  }
  return det;
}

template <size_t Dim>
void sanitize_gauge_h_and_metric(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> gauge_h,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*> metric) {
  static_assert(Dim == 3);
  const size_t size = get_size(gauge_h->get(0));
  for (size_t s = 0; s < size; ++s) {
    bool valid = true;
    for (size_t a = 0; a < 4; ++a) {
      valid = valid and std::isfinite(gauge_h->get(a)[s]);
      for (size_t b = 0; b < 4; ++b) {
        valid = valid and std::isfinite(metric->get(a, b)[s]);
      }
    }
    if (valid) {
      const double det = determinant4_at_point(*metric, s);
      valid = std::isfinite(det) and std::abs(det) > 1.0e-12;
    }
    if (not valid) {
      for (size_t a = 0; a < 4; ++a) {
        gauge_h->get(a)[s] = 0.0;
        for (size_t b = 0; b < 4; ++b) {
          metric->get(a, b)[s] = 0.0;
        }
      }
      metric->get(0, 0)[s] = -1.0;
      metric->get(1, 1)[s] = 1.0;
      metric->get(2, 2)[s] = 1.0;
      metric->get(3, 3)[s] = 1.0;
    }
  }
}

template <size_t Dim>
const Mesh<Dim>& extract_volume_mesh(const Mesh<Dim>& volume_mesh) {
  return volume_mesh;
}

template <size_t Dim>
const Mesh<Dim>& extract_volume_mesh(
    const DirectionalIdMap<Dim, Mesh<Dim>>& volume_meshes) {
  ASSERT(not volume_meshes.empty(),
         "No volume mesh is available for the horizon boundary condition.");
  return volume_meshes.begin()->second;
}

template <size_t Dim>
auto surface_laplacian(
    const DataVector& field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
    const Mesh<Dim - 1>& face_mesh) {
  static_assert(Dim == 3);
  const auto tangents =
      logical_partial_derivative(face_inertial_coords, face_mesh);

  auto surface_metric =
      make_with_value<tnsr::ii<DataVector, 2, Frame::ElementLogical>>(field,
                                                                      0.0);
  for (size_t a = 0; a < 2; ++a) {
    for (size_t b = a; b < 2; ++b) {
      for (size_t i = 0; i < Dim; ++i) {
        surface_metric.get(a, b) += tangents.get(a, i) * tangents.get(b, i);
      }
    }
  }

  Scalar<DataVector> det_surface_metric{};
  tnsr::II<DataVector, 2, Frame::ElementLogical> inverse_surface_metric{};
  determinant_and_inverse(make_not_null(&det_surface_metric),
                          make_not_null(&inverse_surface_metric),
                          surface_metric);
  const DataVector sqrt_det_surface_metric = sqrt(get(det_surface_metric));

  Scalar<DataVector> scalar_field{};
  get(scalar_field) = field;
  const auto deriv_field = logical_partial_derivative(scalar_field, face_mesh);

  auto weighted_gradient =
      make_with_value<tnsr::I<DataVector, 2, Frame::ElementLogical>>(field,
                                                                     0.0);
  for (size_t a = 0; a < 2; ++a) {
    for (size_t b = 0; b < 2; ++b) {
      weighted_gradient.get(a) += sqrt_det_surface_metric *
                                  inverse_surface_metric.get(a, b) *
                                  deriv_field.get(b);
    }
  }

  const auto deriv_weighted_gradient =
      logical_partial_derivative(weighted_gradient, face_mesh);
  DataVector result = make_with_value<DataVector>(field, 0.0);
  for (size_t a = 0; a < 2; ++a) {
    result += deriv_weighted_gradient.get(a, a);
  }
  result /= sqrt_det_surface_metric;
  return result;
}

template <size_t Dim>
DataVector radius(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords) {
  DataVector r2 = square(face_inertial_coords.get(0));
  for (size_t d = 1; d < Dim; ++d) {
    r2 += square(face_inertial_coords.get(d));
  }
  return sqrt(r2);
}

template <size_t Dim>
void apply_impl(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        n_dot_flux_xi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal,
    const Direction<Dim>& direction, const Mesh<Dim>& volume_mesh,
    const double mass, const tnsr::A<DataVector, Dim, Frame::Inertial>& gauge_h,
    const tnsr::a<DataVector, Dim, Frame::Inertial>& dtxi) {
  const auto face_mesh = volume_mesh.slice_away(direction.dimension());
  const DataVector r = radius(face_inertial_coords);
  DataVector normal_dot_radial =
      face_normal.get(0) * face_inertial_coords.get(0) / r;
  for (size_t d = 1; d < Dim; ++d) {
    normal_dot_radial += face_normal.get(d) * face_inertial_coords.get(d) / r;
  }
  const DataVector dq_flux_prefactor = 1.0 - 2.0 * mass / r;

  const DataVector angular_laplacian_xi_t =
      square(r) *
      surface_laplacian(xi->get(0), face_inertial_coords, face_mesh);
  // Restore the old physical DQ flux for the Xi equation. The horizon-regular
  // condition is still written for the regular radial derivative, so convert it
  // to the physical normal flux by multiplying by (1 - 2m/r).
  n_dot_flux_xi->get(0) = dq_flux_prefactor * normal_dot_radial *
                          (-1.0 + 2.0 * mass * gauge_h.get(0) - dtxi.get(0) -
                           0.5 * angular_laplacian_xi_t / mass);
  for (size_t d = 0; d < Dim; ++d) {
    const DataVector angular_laplacian_xi_i =
        square(r) *
        surface_laplacian(xi->get(d + 1), face_inertial_coords, face_mesh);
    n_dot_flux_xi->get(d + 1) =
        dq_flux_prefactor * normal_dot_radial *
        (face_inertial_coords.get(d) / r + 2.0 * mass * gauge_h.get(d + 1) -
         dtxi.get(d + 1) - 0.5 * angular_laplacian_xi_i / mass);
  }
}

template <size_t Dim>
void apply_linearized_impl(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        xi_correction,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        n_dot_flux_xi_correction,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal,
    const Direction<Dim>& direction, const Mesh<Dim>& volume_mesh,
    const double mass) {
  const auto face_mesh = volume_mesh.slice_away(direction.dimension());
  const DataVector r = radius(face_inertial_coords);
  DataVector normal_dot_radial =
      face_normal.get(0) * face_inertial_coords.get(0) / r;
  for (size_t d = 1; d < Dim; ++d) {
    normal_dot_radial += face_normal.get(d) * face_inertial_coords.get(d) / r;
  }
  const DataVector dq_flux_prefactor = 1.0 - 2.0 * mass / r;

  const DataVector angular_laplacian_xi_t =
      square(r) *
      surface_laplacian(xi_correction->get(0), face_inertial_coords, face_mesh);
  n_dot_flux_xi_correction->get(0) = -0.5 * dq_flux_prefactor *
                                     normal_dot_radial *
                                     angular_laplacian_xi_t / mass;
  for (size_t d = 0; d < Dim; ++d) {
    const DataVector angular_laplacian_xi_i =
        square(r) * surface_laplacian(xi_correction->get(d + 1),
                                      face_inertial_coords, face_mesh);
    n_dot_flux_xi_correction->get(d + 1) = -0.5 * dq_flux_prefactor *
                                           normal_dot_radial *
                                           angular_laplacian_xi_i / mass;
  }
}

}  // namespace

template <size_t Dim>
tnsr::A<DataVector, Dim, Frame::Inertial> HorizonRobin<Dim>::gauge_h(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords) const {
  tnsr::A<DataVector, Dim, Frame::Inertial> result{get_size(coords.get(0)),
                                                   0.0};
  if (not use_gauge_h_) {
    return result;
  }
  const auto source_coords = affine_map_.map_point(coords);
  const auto gauge_h_vars = numeric_data_.variables(
      source_coords,
      tmpl::list<
          gh::Tags::GaugeH<DataVector, Dim>,
          gr::Tags::SpacetimeMetric<DataVector, Dim, Frame::Inertial>>{});
  auto gauge_h_cov =
      tuples::get<gh::Tags::GaugeH<DataVector, Dim>>(gauge_h_vars);
  auto spacetime_metric =
      tuples::get<gr::Tags::SpacetimeMetric<DataVector, Dim, Frame::Inertial>>(
          gauge_h_vars);
  sanitize_gauge_h_and_metric(make_not_null(&gauge_h_cov),
                              make_not_null(&spacetime_metric));
  const auto det_and_inverse = determinant_and_inverse(spacetime_metric);
  const auto& inverse_spacetime_metric = det_and_inverse.second;
  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t b = 0; b < Dim + 1; ++b) {
      result.get(a) += inverse_spacetime_metric.get(a, b) * gauge_h_cov.get(b);
    }
  }
  auto mapped_result = affine_map_.map_vector(result, coords);
  const auto wave_map_correction =
      affine_map_.wave_map_correction(inverse_spacetime_metric, coords);
  for (size_t a = 0; a < Dim + 1; ++a) {
    mapped_result.get(a) += wave_map_correction.get(a);
  }
  return mapped_result;
}

template <size_t Dim>
tnsr::a<DataVector, Dim, Frame::Inertial> HorizonRobin<Dim>::dtxi(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords) const {
  tnsr::a<DataVector, Dim, Frame::Inertial> result{get_size(coords.get(0)),
                                                   0.0};
  if (not use_dtxi_) {
    return result;
  }
  return tuples::get<DQ::Tags::Xi<DataVector, Dim>>(dtxi_data_.variables(
      coords, tmpl::list<DQ::Tags::Xi<DataVector, Dim>>{}));
}

template <size_t Dim>
void HorizonRobin<Dim>::apply(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        n_dot_flux_xi,
    const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal,
    const Direction<Dim>& direction, const Mesh<Dim>& volume_mesh) const {
  if (zero_source_) {
    apply_linearized_impl(xi, n_dot_flux_xi, face_inertial_coords,
                          face_normal, direction, volume_mesh, mass_);
    return;
  }
  apply_impl(xi, n_dot_flux_xi, face_inertial_coords, face_normal, direction,
             volume_mesh, mass_, gauge_h(face_inertial_coords),
             dtxi(face_inertial_coords));
}

template <size_t Dim>
void HorizonRobin<Dim>::apply(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        n_dot_flux_xi,
    const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal,
    const Direction<Dim>& direction,
    const DirectionalIdMap<Dim, Mesh<Dim>>& volume_meshes) const {
  const auto& volume_mesh = extract_volume_mesh(volume_meshes);
  if (zero_source_) {
    apply_linearized_impl(xi, n_dot_flux_xi, face_inertial_coords,
                          face_normal, direction, volume_mesh, mass_);
    return;
  }
  apply_impl(xi, n_dot_flux_xi, face_inertial_coords, face_normal, direction,
             volume_mesh, mass_, gauge_h(face_inertial_coords),
             dtxi(face_inertial_coords));
}

template <size_t Dim>
void HorizonRobin<Dim>::apply_linearized(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        xi_correction,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        n_dot_flux_xi_correction,
    const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi_correction*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal,
    const Direction<Dim>& direction, const Mesh<Dim>& volume_mesh) const {
  apply_linearized_impl(xi_correction, n_dot_flux_xi_correction,
                        face_inertial_coords, face_normal, direction,
                        volume_mesh, mass_);
}

template <size_t Dim>
void HorizonRobin<Dim>::apply_linearized(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        xi_correction,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        n_dot_flux_xi_correction,
    const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi_correction*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal,
    const Direction<Dim>& direction,
    const DirectionalIdMap<Dim, Mesh<Dim>>& volume_meshes) const {
  apply_linearized_impl(xi_correction, n_dot_flux_xi_correction,
                        face_inertial_coords, face_normal, direction,
                        extract_volume_mesh(volume_meshes), mass_);
}

template <size_t Dim>
void HorizonRobin<Dim>::pup(PUP::er& p) {
  p | mass_;
  numeric_data_.pup(p);
  p | use_gauge_h_;
  p | affine_map_file_;
  p | zero_source_;
  dtxi_data_.pup(p);
  p | use_dtxi_;
  if (p.isUnpacking()) {
    affine_map_ = DQ::detail::BbhAffineMap::from_file(affine_map_file_);
  }
}

template <size_t Dim>
bool operator==(const HorizonRobin<Dim>& lhs, const HorizonRobin<Dim>& rhs) {
  return lhs.mass() == rhs.mass();
}

template <size_t Dim>
bool operator!=(const HorizonRobin<Dim>& lhs, const HorizonRobin<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
PUP::able::PUP_ID HorizonRobin<Dim>::my_PUP_ID = 0;  // NOLINT

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                \
  template class HorizonRobin<DIM(data)>;                   \
  template bool operator==(const HorizonRobin<DIM(data)>&,  \
                           const HorizonRobin<DIM(data)>&); \
  template bool operator!=(const HorizonRobin<DIM(data)>&,  \
                           const HorizonRobin<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (3))

#undef INSTANTIATE
#undef DIM

}  // namespace DQ::BoundaryConditions
