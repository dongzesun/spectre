// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/DQ/BoundaryConditions/SingleBhOuterBoundary.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/DQ/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace DQ::BoundaryConditions {

namespace {
template <size_t Dim>
std::array<DataVector, 2> theta_phi_of(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const std::array<double, 3>& center) {
  static_assert(Dim == 3);
  const DataVector dx = coords.get(0) - center[0];
  const DataVector dy = coords.get(1) - center[1];
  const DataVector dz = coords.get(2) - center[2];
  const DataVector radius = sqrt(dx * dx + dy * dy + dz * dz);
  std::array<DataVector, 2> theta_phi{DataVector{get_size(radius)},
                                      DataVector{get_size(radius)}};
  get<0>(theta_phi) = acos(dz / radius);
  get<1>(theta_phi) = atan2(dy, dx);
  return theta_phi;
}

std::array<double, 3> solve_3x3(
    std::array<std::array<double, 3>, 3> matrix,
    std::array<double, 3> rhs) {
  for (size_t pivot = 0; pivot < 3; ++pivot) {
    size_t pivot_row = pivot;
    double pivot_abs = std::abs(matrix[pivot][pivot]);
    for (size_t row = pivot + 1; row < 3; ++row) {
      const double candidate_abs = std::abs(matrix[row][pivot]);
      if (candidate_abs > pivot_abs) {
        pivot_abs = candidate_abs;
        pivot_row = row;
      }
    }
    ASSERT(pivot_abs > 1.0e-14,
           "Singular area-gauge angular relabeling matrix.");
    if (pivot_row != pivot) {
      std::swap(matrix[pivot], matrix[pivot_row]);
      std::swap(rhs[pivot], rhs[pivot_row]);
    }
    const double inverse_pivot = 1.0 / matrix[pivot][pivot];
    for (size_t column = pivot; column < 3; ++column) {
      matrix[pivot][column] *= inverse_pivot;
    }
    rhs[pivot] *= inverse_pivot;
    for (size_t row = 0; row < 3; ++row) {
      if (row == pivot) {
        continue;
      }
      const double factor = matrix[row][pivot];
      for (size_t column = pivot; column < 3; ++column) {
        matrix[row][column] -= factor * matrix[pivot][column];
      }
      rhs[row] -= factor * rhs[pivot];
    }
  }
  return rhs;
}

double norm(const std::array<double, 3>& vector) {
  return sqrt(vector[0] * vector[0] + vector[1] * vector[1] +
              vector[2] * vector[2]);
}

std::array<DataVector, 3> relabeled_directions(
    const tnsr::i<DataVector, 3, Frame::Inertial>& direction,
    const std::array<double, 3>& alpha, const double sign) {
  const size_t size = get_size(direction.get(0));
  std::array<DataVector, 3> result{
      DataVector{size}, DataVector{size}, DataVector{size}};
  const DataVector alpha_dot_n = alpha[0] * direction.get(0) +
                                 alpha[1] * direction.get(1) +
                                 alpha[2] * direction.get(2);
  DataVector norm_trial{size, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) =
        direction.get(i) + sign * (gsl::at(alpha, i) -
                                   alpha_dot_n * direction.get(i));
    norm_trial += gsl::at(result, i) * gsl::at(result, i);
  }
  norm_trial = sqrt(norm_trial);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) /= norm_trial;
  }
  return result;
}

template <size_t Dim>
tnsr::a<DataVector, Dim, Frame::Inertial> interpolate_xi(
    const NumericData& xi_data,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords) {
  const auto xi_vars =
      xi_data.variables(coords, tmpl::list<DQ::Tags::Xi<DataVector, Dim>>{});
  return tuples::get<DQ::Tags::Xi<DataVector, Dim>>(xi_vars);
}

template <size_t Dim>
tnsr::ia<DataVector, Dim, Frame::Inertial> outer_face_xi_gradient(
    const NumericData& xi_data,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_coords,
    const tnsr::a<DataVector, Dim, Frame::Inertial>& xi_on_face,
    const double step) {
  static_assert(Dim == 3);
  const size_t num_points = get_size(face_coords.get(0));
  tnsr::ia<DataVector, Dim, Frame::Inertial> deriv_xi{num_points};
  if (step <= 0.0) {
    ERROR("XiDerivativeStep must be positive, not " << step << ".");
  }

  DataVector radius{num_points, 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    radius += face_coords.get(i) * face_coords.get(i);
  }
  radius = sqrt(radius);

  std::array<DataVector, 3> normal{
      face_coords.get(0) / radius, face_coords.get(1) / radius,
      face_coords.get(2) / radius};
  std::array<DataVector, 3> tangent_one{
      DataVector{num_points}, DataVector{num_points}, DataVector{num_points}};
  std::array<DataVector, 3> tangent_two{
      DataVector{num_points}, DataVector{num_points}, DataVector{num_points}};
  for (size_t p = 0; p < num_points; ++p) {
    const bool use_z_axis = std::abs(normal[2][p]) < 0.9;
    const std::array<double, 3> axis{
        use_z_axis ? 0.0 : 1.0, 0.0, use_z_axis ? 1.0 : 0.0};
    std::array<double, 3> t1{
        axis[1] * normal[2][p] - axis[2] * normal[1][p],
        axis[2] * normal[0][p] - axis[0] * normal[2][p],
        axis[0] * normal[1][p] - axis[1] * normal[0][p]};
    const double t1_norm =
        sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
    for (size_t i = 0; i < Dim; ++i) {
      t1[i] /= t1_norm;
    }
    const std::array<double, 3> t2{
        normal[1][p] * t1[2] - normal[2][p] * t1[1],
        normal[2][p] * t1[0] - normal[0][p] * t1[2],
        normal[0][p] * t1[1] - normal[1][p] * t1[0]};
    for (size_t i = 0; i < Dim; ++i) {
      tangent_one[i][p] = t1[i];
      tangent_two[i][p] = t2[i];
    }
  }

  tnsr::I<DataVector, Dim, Frame::Inertial> radial_one{num_points};
  tnsr::I<DataVector, Dim, Frame::Inertial> radial_two{num_points};
  const DataVector radial_radius_one = radius - step;
  const DataVector radial_radius_two = radius - 2.0 * step;
  for (size_t i = 0; i < Dim; ++i) {
    radial_one.get(i) = radial_radius_one * normal[i];
    radial_two.get(i) = radial_radius_two * normal[i];
  }
  const auto xi_radial_one = interpolate_xi(xi_data, radial_one);
  const auto xi_radial_two = interpolate_xi(xi_data, radial_two);

  const auto tangent_derivative =
      [&xi_data, &face_coords, &radius, step, num_points](
          const std::array<DataVector, 3>& tangent) {
        tnsr::I<DataVector, Dim, Frame::Inertial> plus_coords{num_points};
        tnsr::I<DataVector, Dim, Frame::Inertial> minus_coords{num_points};
        const DataVector projected_radius =
            radius * (1.0 - 1.0e-12);
        for (size_t i = 0; i < Dim; ++i) {
          plus_coords.get(i) = face_coords.get(i) + step * tangent[i];
          minus_coords.get(i) = face_coords.get(i) - step * tangent[i];
        }
        for (auto* const coords : {&plus_coords, &minus_coords}) {
          DataVector trial_radius{num_points, 0.0};
          for (size_t i = 0; i < Dim; ++i) {
            trial_radius += coords->get(i) * coords->get(i);
          }
          trial_radius = sqrt(trial_radius);
          for (size_t i = 0; i < Dim; ++i) {
            coords->get(i) *= projected_radius / trial_radius;
          }
        }
        const auto xi_plus = interpolate_xi(xi_data, plus_coords);
        const auto xi_minus = interpolate_xi(xi_data, minus_coords);
        tnsr::a<DataVector, Dim, Frame::Inertial> result{num_points};
        for (size_t a = 0; a < Dim + 1; ++a) {
          result.get(a) = (xi_plus.get(a) - xi_minus.get(a)) / (2.0 * step);
        }
        return result;
      };
  const auto xi_tangent_one = tangent_derivative(tangent_one);
  const auto xi_tangent_two = tangent_derivative(tangent_two);

  for (size_t a = 0; a < Dim + 1; ++a) {
    const DataVector radial_derivative =
        (3.0 * xi_on_face.get(a) - 4.0 * xi_radial_one.get(a) +
         xi_radial_two.get(a)) /
        (2.0 * step);
    for (size_t j = 0; j < Dim; ++j) {
      deriv_xi.get(j, a) =
          normal[j] * radial_derivative +
          tangent_one[j] * xi_tangent_one.get(a) +
          tangent_two[j] * xi_tangent_two.get(a);
    }
  }
  return deriv_xi;
}

std::array<DataVector, 2> theta_phi_from_directions(
    const std::array<DataVector, 3>& direction) {
  const size_t size = gsl::at(direction, 0).size();
  std::array<DataVector, 2> result{DataVector{size}, DataVector{size}};
  for (size_t p = 0; p < size; ++p) {
    const double z = std::clamp(gsl::at(direction, 2)[p], -1.0, 1.0);
    gsl::at(result, 0)[p] = acos(z);
    gsl::at(result, 1)[p] =
        atan2(gsl::at(direction, 1)[p], gsl::at(direction, 0)[p]);
  }
  return result;
}

std::array<double, 3> relabeled_area_dipole(
    const ylm::Strahlkorper<Frame::Inertial>& surface,
    const tnsr::i<DataVector, 3, Frame::Inertial>& direction,
    const std::array<double, 3>& alpha, const double sign) {
  const auto relabeled_direction = relabeled_directions(direction, alpha, sign);
  const auto relabeled_theta_phi =
      theta_phi_from_directions(relabeled_direction);
  const auto interpolation_info =
      surface.ylm_spherepack().set_up_interpolation_info(relabeled_theta_phi);
  DataVector relabeled_radius{gsl::at(relabeled_direction, 0).size()};
  surface.ylm_spherepack().interpolate_from_coefs(
      make_not_null(&relabeled_radius), surface.coefficients(),
      interpolation_info);

  std::array<DataVector, 3> embedding{
      DataVector{relabeled_radius.size()}, DataVector{relabeled_radius.size()},
      DataVector{relabeled_radius.size()}};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(embedding, i) =
        gsl::at(surface.expansion_center(), i) +
        relabeled_radius * gsl::at(relabeled_direction, i);
  }

  std::array<tnsr::i<DataVector, 2, Frame::ElementLogical>, 3>
      embedding_gradient{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(embedding_gradient, i) =
        surface.ylm_spherepack().gradient(gsl::at(embedding, i));
  }
  DataVector h00{relabeled_radius.size(), 0.0};
  DataVector h01{relabeled_radius.size(), 0.0};
  DataVector h11{relabeled_radius.size(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    h00 += gsl::at(embedding_gradient, i).get(0) *
           gsl::at(embedding_gradient, i).get(0);
    h01 += gsl::at(embedding_gradient, i).get(0) *
           gsl::at(embedding_gradient, i).get(1);
    h11 += gsl::at(embedding_gradient, i).get(1) *
           gsl::at(embedding_gradient, i).get(1);
  }
  DataVector area_density = h00 * h11 - h01 * h01;
  for (double& value : area_density) {
    value = sqrt(std::max(value, 0.0));
  }

  std::array<double, 3> result{{0.0, 0.0, 0.0}};
  for (size_t i = 0; i < 3; ++i) {
    const DataVector integrand = direction.get(i) * area_density;
    result[i] = surface.ylm_spherepack().definite_integral(integrand.data());
  }
  return result;
}
}  // namespace

template <size_t Dim>
SingleBhOuterBoundary<Dim>::SingleBhOuterBoundary(
    std::string surface_h5_file, std::string surface_subfile,
    const double match_time, const double match_time_epsilon,
    const bool check_frame, std::string affine_map_file,
    std::string area_gauge_relabeling,
    std::optional<std::array<double, 3>> area_gauge_alpha,
    std::string boundary_data, const double mass,
    std::string metric_file_glob, std::string metric_subgroup,
    const int metric_observation_step,
    const bool metric_extrapolate_into_excisions,
    std::string metric_affine_map_file, std::string xi_file_glob,
    std::string xi_subgroup, const int xi_observation_step,
    const bool xi_extrapolate_into_excisions,
    const double xi_derivative_step,
    const Options::Context& context)
    : surface_h5_file_(std::move(surface_h5_file)),
      surface_subfile_(std::move(surface_subfile)),
      match_time_(match_time),
      match_time_epsilon_(match_time_epsilon),
      check_frame_(check_frame),
      affine_map_file_(std::move(affine_map_file)),
      area_gauge_relabeling_(std::move(area_gauge_relabeling)),
      area_gauge_alpha_override_(std::move(area_gauge_alpha)),
      boundary_data_(std::move(boundary_data)),
      mass_(mass),
      metric_data_(std::move(metric_file_glob), std::move(metric_subgroup),
                   metric_observation_step,
                   metric_extrapolate_into_excisions),
      metric_affine_map_file_(std::move(metric_affine_map_file)),
      metric_affine_map_(
          DQ::detail::BbhAffineMap::from_file(metric_affine_map_file_)),
      xi_data_(std::move(xi_file_glob), std::move(xi_subgroup),
               xi_observation_step, xi_extrapolate_into_excisions),
      xi_derivative_step_(xi_derivative_step),
      affine_map_(DQ::detail::BbhAffineMap::from_file(affine_map_file_)) {
  if constexpr (Dim != 3) {
    PARSE_ERROR(context, "SingleBhOuterBoundary is implemented only in 3D.");
  }
  if (surface_h5_file_.empty()) {
    PARSE_ERROR(context, "SurfaceH5File must not be empty.");
  }
  if (surface_subfile_.empty()) {
    PARSE_ERROR(context, "SurfaceSubfile must not be empty.");
  }
  if (match_time_epsilon_ <= 0.0) {
    PARSE_ERROR(context, "MatchTimeEpsilon must be positive.");
  }
  if (area_gauge_relabeling_ != "StoredPlus" and
      area_gauge_relabeling_ != "StoredMinus" and
      area_gauge_relabeling_ != "DqPlus" and
      area_gauge_relabeling_ != "None") {
    PARSE_ERROR(context,
                "AreaGaugeRelabeling must be one of 'StoredPlus', "
                "'StoredMinus', 'DqPlus', or 'None', not '"
                    << area_gauge_relabeling_ << "'.");
  }
  if (boundary_data_ != "XiSurface" and
      boundary_data_ != "DtXiTimeJacobian") {
    PARSE_ERROR(context,
                "BoundaryData must be 'XiSurface' or 'DtXiTimeJacobian', not '"
                    << boundary_data_ << "'.");
  }
  if (boundary_data_ == "DtXiTimeJacobian" and mass_ <= 0.0) {
    PARSE_ERROR(context, "Mass must be positive for DtXiTimeJacobian.");
  }
  if (boundary_data_ == "DtXiTimeJacobian" and xi_derivative_step_ <= 0.0) {
    PARSE_ERROR(context,
                "XiDerivativeStep must be positive for DtXiTimeJacobian.");
  }
  if (not std::filesystem::exists(surface_h5_file_)) {
    PARSE_ERROR(context, "SurfaceH5File '"
                             << surface_h5_file_
                             << "' does not exist. The SingleBH outer-surface "
                                "precompute step must complete successfully "
                                "before starting DQ3D.");
  }
  surface_ = ylm::read_surface_ylm_single_time<Frame::Inertial>(
      surface_h5_file_, surface_subfile_, match_time_, match_time_epsilon_,
      check_frame_);
  initialize_area_gauge_relabeling(context);
}

template <size_t Dim>
void SingleBhOuterBoundary<Dim>::initialize_area_gauge_relabeling(
    const Options::Context& /*context*/) {
  static_assert(Dim == 3);
  const auto radius = ylm::radius(surface_);
  const auto theta_phi = ylm::theta_phi(surface_);
  const auto direction = ylm::rhat(theta_phi);
  const auto radius_gradient =
      surface_.ylm_spherepack().gradient(get(radius));

  const DataVector area_density =
      get(radius) * sqrt(get(radius) * get(radius) +
                         radius_gradient.get(0) * radius_gradient.get(0) +
                         radius_gradient.get(1) * radius_gradient.get(1));

  std::array<std::array<double, 3>, 3> matrix{{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};
  for (size_t i = 0; i < 3; ++i) {
    const DataVector dipole_integrand = direction.get(i) * area_density;
    area_gauge_dipole_before_[i] =
        surface_.ylm_spherepack().definite_integral(dipole_integrand.data());
    for (size_t j = 0; j < 3; ++j) {
      const DataVector integrand =
          area_density *
          ((i == j ? 1.0 : 0.0) - direction.get(i) * direction.get(j));
      matrix[i][j] =
          surface_.ylm_spherepack().definite_integral(integrand.data());
    }
  }

  area_gauge_alpha_ = solve_3x3(matrix, area_gauge_dipole_before_);
  area_gauge_dipole_after_ =
      relabeled_area_dipole(surface_, direction, area_gauge_alpha_, 1.0);
  area_gauge_sign_flipped_ =
      norm(area_gauge_dipole_after_) > norm(area_gauge_dipole_before_);
  if (area_gauge_sign_flipped_) {
    for (double& component : area_gauge_alpha_) {
      component *= -1.0;
    }
    area_gauge_dipole_after_ =
        relabeled_area_dipole(surface_, direction, area_gauge_alpha_, 1.0);
  }
  if (area_gauge_alpha_override_.has_value()) {
    area_gauge_alpha_ = area_gauge_alpha_override_.value();
    area_gauge_dipole_after_ =
        relabeled_area_dipole(surface_, direction, area_gauge_alpha_, 1.0);
    area_gauge_sign_flipped_ = false;
  }

  std::cout << "DQ SingleBhOuterBoundary area-gauge relabeling: alpha = ("
            << area_gauge_alpha_[0] << ", " << area_gauge_alpha_[1] << ", "
            << area_gauge_alpha_[2] << "), dipole before = ("
            << area_gauge_dipole_before_[0] << ", "
            << area_gauge_dipole_before_[1] << ", "
            << area_gauge_dipole_before_[2] << "), dipole after = ("
            << area_gauge_dipole_after_[0] << ", "
            << area_gauge_dipole_after_[1] << ", "
            << area_gauge_dipole_after_[2] << "), sign flipped = "
            << (area_gauge_sign_flipped_ ? "true" : "false")
            << ", convention = " << area_gauge_relabeling_
            << ", alpha override = "
            << (area_gauge_alpha_override_.has_value() ? "true" : "false")
            << std::endl;
}

template <size_t Dim>
void SingleBhOuterBoundary<Dim>::apply(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
    /*n_dot_flux_xi*/,
    const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*face_normal*/) const {
  if (boundary_data_ == "DtXiTimeJacobian") {
    const size_t num_points = get_size(face_inertial_coords.get(0));
    for (size_t a = 0; a < Dim + 1; ++a) {
      xi->get(a) =
          make_with_value<DataVector>(face_inertial_coords.get(0), 0.0);
    }

    const auto xi_solution = interpolate_xi(xi_data_, face_inertial_coords);
    const auto deriv_xi_solution =
        outer_face_xi_gradient(xi_data_, face_inertial_coords, xi_solution,
                               xi_derivative_step_);

    tnsr::I<DataVector, Dim, Frame::Inertial> metric_coords{num_points};
    for (size_t i = 0; i < Dim; ++i) {
      metric_coords.get(i) =
          face_inertial_coords.get(i) + xi_solution.get(i + 1);
    }
    const auto metric_source_coords =
        metric_affine_map_.map_point(metric_coords);
    const auto metric_vars = metric_data_.variables(
        metric_source_coords,
        tmpl::list<gr::Tags::SpacetimeMetric<DataVector, Dim,
                                             Frame::Inertial>>{});
    const auto spacetime_metric =
        tuples::get<gr::Tags::SpacetimeMetric<DataVector, Dim,
                                             Frame::Inertial>>(metric_vars);

    const DataVector radius =
        sqrt(face_inertial_coords.get(0) * face_inertial_coords.get(0) +
             face_inertial_coords.get(1) * face_inertial_coords.get(1) +
             face_inertial_coords.get(2) * face_inertial_coords.get(2));

    for (size_t p = 0; p < num_points; ++p) {
      std::array<std::array<double, 4>, 3> dx_comoving{};
      for (size_t i = 0; i < Dim; ++i) {
        dx_comoving[i][0] = deriv_xi_solution.get(i, 0)[p];
        for (size_t j = 0; j < Dim; ++j) {
          dx_comoving[i][j + 1] =
              (i == j ? 1.0 : 0.0) + deriv_xi_solution.get(i, j + 1)[p];
        }
      }

      std::array<std::array<double, 3>, 3> spatial_metric{};
      for (size_t i = 0; i < Dim; ++i) {
        for (size_t j = 0; j < Dim; ++j) {
          spatial_metric[i][j] = 0.0;
          for (size_t a = 0; a < Dim + 1; ++a) {
            for (size_t b = 0; b < Dim + 1; ++b) {
              spatial_metric[i][j] += spacetime_metric.get(a, b)[p] *
                                      dx_comoving[i][a] *
                                      dx_comoving[j][b];
            }
          }
        }
      }
      std::array<std::array<double, 3>, 3> inverse_spatial_metric{};
      for (size_t column = 0; column < Dim; ++column) {
        std::array<double, 3> rhs{{0.0, 0.0, 0.0}};
        rhs[column] = 1.0;
        const auto solution = solve_3x3(spatial_metric, rhs);
        for (size_t row = 0; row < Dim; ++row) {
          inverse_spatial_metric[row][column] = solution[row];
        }
      }

      std::array<std::array<double, 3>, 3> normal_matrix{};
      std::array<double, 3> normal_rhs{};
      for (size_t i = 0; i < Dim; ++i) {
        double covector_time_component = 0.0;
        for (size_t b = 0; b < Dim + 1; ++b) {
          covector_time_component +=
              spacetime_metric.get(0, b)[p] * dx_comoving[i][b];
        }
        normal_rhs[i] = -covector_time_component;
        for (size_t j = 0; j < Dim; ++j) {
          normal_matrix[i][j] = 0.0;
          for (size_t b = 0; b < Dim + 1; ++b) {
            normal_matrix[i][j] += spacetime_metric.get(j + 1, b)[p] *
                                   dx_comoving[i][b];
          }
        }
      }
      const auto normal_spatial = solve_3x3(normal_matrix, normal_rhs);
      std::array<double, 4> normal{{1.0, normal_spatial[0],
                                    normal_spatial[1], normal_spatial[2]}};
      double normal_norm = 0.0;
      for (size_t a = 0; a < Dim + 1; ++a) {
        for (size_t b = 0; b < Dim + 1; ++b) {
          normal_norm += spacetime_metric.get(a, b)[p] * normal[a] * normal[b];
        }
      }
      if (not(normal_norm < 0.0)) {
        ERROR("DtXi outer boundary failed to construct a timelike normal. "
              "G_ab N^a N^b = "
              << normal_norm << " at boundary point " << p << ".");
      }
      const double normal_scale = 1.0 / sqrt(-normal_norm);
      for (double& component : normal) {
        component *= normal_scale;
      }
      if (normal[0] < 0.0) {
        for (double& component : normal) {
          component *= -1.0;
        }
      }

      std::array<double, 3> target_g_ti{};
      for (size_t i = 0; i < Dim; ++i) {
        target_g_ti[i] =
            2.0 * mass_ * face_inertial_coords.get(i)[p] /
            (radius[p] * radius[p]);
      }
      std::array<double, 3> target_shift{};
      double shift_norm = 0.0;
      for (size_t i = 0; i < Dim; ++i) {
        target_shift[i] = 0.0;
        for (size_t j = 0; j < Dim; ++j) {
          target_shift[i] += inverse_spatial_metric[i][j] * target_g_ti[j];
          shift_norm +=
              inverse_spatial_metric[i][j] * target_g_ti[i] * target_g_ti[j];
        }
      }
      const double target_g_tt = -1.0 + 2.0 * mass_ / radius[p];
      const double lapse_squared = shift_norm - target_g_tt;
      if (not(lapse_squared > 0.0)) {
        ERROR("DtXi outer boundary has non-positive target lapse squared: "
              << lapse_squared << " at boundary point " << p
              << ", shift_norm=" << shift_norm
              << ", target_g_tt=" << target_g_tt << ".");
      }
      const double normal_coefficient = sqrt(lapse_squared);
      std::array<double, 4> dt_x_target{};
      for (size_t a = 0; a < Dim + 1; ++a) {
        dt_x_target[a] = normal_coefficient * normal[a];
        for (size_t i = 0; i < Dim; ++i) {
          dt_x_target[a] += target_shift[i] * dx_comoving[i][a];
        }
      }
      xi->get(0)[p] = dt_x_target[0] - 1.0;
      for (size_t i = 0; i < Dim; ++i) {
        xi->get(i + 1)[p] = dt_x_target[i + 1];
      }
    }
    return;
  }

  const auto inertial_face_coords = affine_map_.map_point(face_inertial_coords);
  const auto& center = surface_.expansion_center();
  const DataVector dx = inertial_face_coords.get(0) - center[0];
  const DataVector dy = inertial_face_coords.get(1) - center[1];
  const DataVector dz = inertial_face_coords.get(2) - center[2];
  const DataVector inertial_radius = sqrt(dx * dx + dy * dy + dz * dz);
  tnsr::i<DataVector, Dim, Frame::Inertial> inertial_direction{
      get_size(face_inertial_coords.get(0))};
  inertial_direction.get(0) = dx / inertial_radius;
  inertial_direction.get(1) = dy / inertial_radius;
  inertial_direction.get(2) = dz / inertial_radius;

  std::array<DataVector, 3> relabeled_direction{
      inertial_direction.get(0), inertial_direction.get(1),
      inertial_direction.get(2)};
  if (area_gauge_relabeling_ == "StoredPlus" or
      area_gauge_relabeling_ == "StoredMinus") {
    relabeled_direction = relabeled_directions(
        inertial_direction, area_gauge_alpha_,
        area_gauge_relabeling_ == "StoredPlus" ? 1.0 : -1.0);
  } else if (area_gauge_relabeling_ == "DqPlus") {
    const DataVector local_radius =
        sqrt(face_inertial_coords.get(0) * face_inertial_coords.get(0) +
             face_inertial_coords.get(1) * face_inertial_coords.get(1) +
             face_inertial_coords.get(2) * face_inertial_coords.get(2));
    tnsr::i<DataVector, Dim, Frame::Inertial> local_direction{
        get_size(face_inertial_coords.get(0))};
    for (size_t i = 0; i < Dim; ++i) {
      local_direction.get(i) = face_inertial_coords.get(i) / local_radius;
    }
    const auto relabeled_local_direction =
        relabeled_directions(local_direction, area_gauge_alpha_, 1.0);
    tnsr::I<DataVector, Dim, Frame::Inertial> relabeled_local_coords{
        get_size(face_inertial_coords.get(0))};
    for (size_t i = 0; i < Dim; ++i) {
      relabeled_local_coords.get(i) =
          local_radius * gsl::at(relabeled_local_direction, i);
    }
    const auto relabeled_inertial_coords =
        affine_map_.map_point(relabeled_local_coords);
    const DataVector relabeled_dx =
        relabeled_inertial_coords.get(0) - center[0];
    const DataVector relabeled_dy =
        relabeled_inertial_coords.get(1) - center[1];
    const DataVector relabeled_dz =
        relabeled_inertial_coords.get(2) - center[2];
    const DataVector relabeled_radius = sqrt(relabeled_dx * relabeled_dx +
                                             relabeled_dy * relabeled_dy +
                                             relabeled_dz * relabeled_dz);
    relabeled_direction[0] = relabeled_dx / relabeled_radius;
    relabeled_direction[1] = relabeled_dy / relabeled_radius;
    relabeled_direction[2] = relabeled_dz / relabeled_radius;
  } else if (area_gauge_relabeling_ != "None") {
    ERROR("Unknown AreaGaugeRelabeling '" << area_gauge_relabeling_ << "'.");
  }
  const auto theta_phi = theta_phi_from_directions(relabeled_direction);
  const auto interpolation_info =
      surface_.ylm_spherepack().set_up_interpolation_info(theta_phi);
  DataVector surface_radius{get_size(face_inertial_coords.get(0))};
  surface_.ylm_spherepack().interpolate_from_coefs(
      make_not_null(&surface_radius), surface_.coefficients(),
      interpolation_info);

  xi->get(0) = make_with_value<DataVector>(surface_radius, 0.0);
  tnsr::I<DataVector, Dim, Frame::Inertial> inertial_surface_coords{
      get_size(face_inertial_coords.get(0))};
  inertial_surface_coords.get(0) =
      center[0] + surface_radius * gsl::at(relabeled_direction, 0);
  inertial_surface_coords.get(1) =
      center[1] + surface_radius * gsl::at(relabeled_direction, 1);
  inertial_surface_coords.get(2) =
      center[2] + surface_radius * gsl::at(relabeled_direction, 2);
  const auto local_surface_coords =
      affine_map_.inverse_map_point(inertial_surface_coords);
  xi->get(1) = local_surface_coords.get(0) - face_inertial_coords.get(0);
  xi->get(2) = local_surface_coords.get(1) - face_inertial_coords.get(1);
  xi->get(3) = local_surface_coords.get(2) - face_inertial_coords.get(2);
}

template <size_t Dim>
void SingleBhOuterBoundary<Dim>::apply_linearized(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        xi_correction,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
    /*n_dot_flux_xi_correction*/,
    const tnsr::ia<DataVector, Dim, Frame::Inertial>&
    /*deriv_xi_correction*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*face_inertial_coords*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*face_normal*/) const {
  for (size_t a = 0; a < Dim + 1; ++a) {
    xi_correction->get(a) = 0.0;
  }
}

template <size_t Dim>
void SingleBhOuterBoundary<Dim>::pup(PUP::er& p) {
  elliptic::BoundaryConditions::BoundaryCondition<Dim>::pup(p);
  p | surface_h5_file_;
  p | surface_subfile_;
  p | match_time_;
  p | match_time_epsilon_;
  p | check_frame_;
  p | affine_map_file_;
  p | area_gauge_relabeling_;
  p | area_gauge_alpha_override_;
  p | boundary_data_;
  p | mass_;
  metric_data_.pup(p);
  p | metric_affine_map_file_;
  xi_data_.pup(p);
  p | xi_derivative_step_;
  if (p.isUnpacking()) {
    affine_map_ = DQ::detail::BbhAffineMap::from_file(affine_map_file_);
    metric_affine_map_ =
        DQ::detail::BbhAffineMap::from_file(metric_affine_map_file_);
  }
  p | surface_;
  p | area_gauge_alpha_;
  p | area_gauge_dipole_before_;
  p | area_gauge_dipole_after_;
  p | area_gauge_sign_flipped_;
}

template <size_t Dim>
bool operator==(const SingleBhOuterBoundary<Dim>& lhs,
                const SingleBhOuterBoundary<Dim>& rhs) {
  return lhs.surface_h5_file() == rhs.surface_h5_file() and
         lhs.surface_subfile() == rhs.surface_subfile() and
         lhs.match_time() == rhs.match_time() and
         lhs.match_time_epsilon() == rhs.match_time_epsilon() and
         lhs.check_frame() == rhs.check_frame() and
         lhs.affine_map_file() == rhs.affine_map_file() and
         lhs.area_gauge_relabeling() == rhs.area_gauge_relabeling() and
         lhs.area_gauge_alpha_override() == rhs.area_gauge_alpha_override() and
         lhs.boundary_data() == rhs.boundary_data();
}

template <size_t Dim>
bool operator!=(const SingleBhOuterBoundary<Dim>& lhs,
                const SingleBhOuterBoundary<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
PUP::able::PUP_ID SingleBhOuterBoundary<Dim>::my_PUP_ID = 0;  // NOLINT

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                         \
  template class SingleBhOuterBoundary<DIM(data)>;                   \
  template bool operator==(const SingleBhOuterBoundary<DIM(data)>&,  \
                           const SingleBhOuterBoundary<DIM(data)>&); \
  template bool operator!=(const SingleBhOuterBoundary<DIM(data)>&,  \
                           const SingleBhOuterBoundary<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (3))

#undef INSTANTIATE
#undef DIM

}  // namespace DQ::BoundaryConditions
