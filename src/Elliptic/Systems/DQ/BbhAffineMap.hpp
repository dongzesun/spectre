// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace DQ::detail {

struct BbhAffineMap {
  bool active{false};
  std::array<double, 3> center{{0.0, 0.0, 0.0}};
  std::array<double, 3> center_velocity{{0.0, 0.0, 0.0}};
  std::array<double, 3> center_acceleration{{0.0, 0.0, 0.0}};
  std::array<double, 3> corot_shift{{0.0, 0.0, 0.0}};
  std::array<double, 3> corot_shift_velocity{{0.0, 0.0, 0.0}};
  std::array<double, 3> corot_shift_acceleration{{0.0, 0.0, 0.0}};
  // Constant boost from the moving corotating frame to the local right-hole
  // frame, represented by the velocity components in the corotating axes.
  std::array<double, 3> boost_velocity{{0.0, 0.0, 0.0}};
  // Approximate companion potential m_2 / D used in the local PN frame.
  double companion_potential{0.0};
  // Coefficients in
  // Z^0 = (1 + c0 v^2 + c1 Phi) Y^0 + c2 v_i Y^i
  // Z^i = (1 + c3 Phi) Y^i + c4 (Y^j v_j) v^i
  //       + c5 gamma v^i Y^0.
  // The defaults reproduce the unoptimized PN-style map.
  std::array<double, 6> local_frame_coefficients{
      {-0.5, -1.0, -1.0, 1.0, 0.5, -1.0}};
  // Additional symmetric spatial affine correction S_ij added to the spatial
  // block of the local-frame map. Stored as xx, xy, xz, yy, yz, zz.
  std::array<double, 6> local_frame_spatial_affine_correction{
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  // Constrained time-tilt composition:
  // \tilde Z^0 = Z^0 - tau_i Z^i, \tilde Z^i = Z^i.
  std::array<double, 3> local_frame_time_tilt{{0.0075, 0.0, 0.0}};
  // Constant spatial shift applied before the time tilt:
  // Z_shift^i = Z_base^i - d^i.
  std::array<double, 3> local_frame_spatial_shift{{0.0, 0.0, 0.0}};
  // Quadratic pre-map coefficients in
  // Z^a = M^a_b Y^b + 1/2 Q^a_ij Y^i Y^j, with i,j spatial.
  // Stored by component a=t,x,y,z, each as xx, xy, xz, yy, yz, zz.
  std::array<std::array<double, 6>, 4> local_frame_quadratic{{
      {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
  }};
  std::array<std::array<double, 3>, 3> corot_to_inertial{{
      {{1.0, 0.0, 0.0}},
      {{0.0, 1.0, 0.0}},
      {{0.0, 0.0, 1.0}},
  }};
  std::array<std::array<double, 3>, 3> corot_to_inertial_dt{{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};
  std::array<std::array<double, 3>, 3> corot_to_inertial_ddt{{
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
      {{0.0, 0.0, 0.0}},
  }};

  static BbhAffineMap from_file(const std::string& file_name) {
    BbhAffineMap result{};
    if (file_name.empty()) {
      return result;
    }
    std::ifstream input(file_name);
    ASSERT(input.good(), "Could not open BBH affine map file: " << file_name);
    auto read_line = [&input](const std::string& expected_key, double* values,
                              const size_t count) {
      std::string line{};
      std::getline(input, line);
      ASSERT(not line.empty(),
             "Unexpected end of BBH affine map file while reading "
                 << expected_key);
      std::istringstream stream(line);
      std::string key{};
      stream >> key;
      ASSERT(key == expected_key,
             "Expected key '" << expected_key << "' in BBH affine map file but "
                              << "found '" << key << "'");
      for (size_t i = 0; i < count; ++i) {
        stream >> values[i];
        ASSERT(not stream.fail(), "Failed to parse value "
                                      << i << " for key '" << expected_key
                                      << "' in BBH affine map file");
      }
    };

    read_line("center", result.center.data(), 3);
    read_line("center_velocity", result.center_velocity.data(), 3);
    read_line("corot_shift", result.corot_shift.data(), 3);
    read_line("corot_shift_velocity", result.corot_shift_velocity.data(), 3);
    read_line("corot_to_inertial",
              reinterpret_cast<double*>(result.corot_to_inertial.data()), 9);
    read_line("corot_to_inertial_dt",
              reinterpret_cast<double*>(result.corot_to_inertial_dt.data()), 9);
    std::string optional_line{};
    while (std::getline(input, optional_line)) {
      if (optional_line.empty()) {
        continue;
      }
      std::istringstream stream(optional_line);
      std::string key{};
      stream >> key;
      auto read_optional_values = [&stream, &key](double* values,
                                                  const size_t count) {
        for (size_t i = 0; i < count; ++i) {
          stream >> values[i];
          ASSERT(not stream.fail(), "Failed to parse optional value "
                                        << i << " for key '" << key
                                        << "' in BBH affine map file");
        }
      };
      if (key == "center_acceleration") {
        read_optional_values(result.center_acceleration.data(), 3);
      } else if (key == "corot_shift_acceleration") {
        read_optional_values(result.corot_shift_acceleration.data(), 3);
      } else if (key == "corot_to_inertial_ddt") {
        read_optional_values(
            reinterpret_cast<double*>(result.corot_to_inertial_ddt.data()), 9);
      } else if (key == "boost_velocity") {
        read_optional_values(result.boost_velocity.data(), 3);
      } else if (key == "companion_potential") {
        stream >> result.companion_potential;
        ASSERT(not stream.fail(),
               "Failed to parse optional value for key 'companion_potential' "
               "in BBH affine map file");
      } else if (key == "local_frame_coefficients") {
        read_optional_values(result.local_frame_coefficients.data(), 6);
      } else if (key == "local_frame_spatial_affine_correction") {
        read_optional_values(
            result.local_frame_spatial_affine_correction.data(), 6);
      } else if (key == "local_frame_time_tilt") {
        read_optional_values(result.local_frame_time_tilt.data(), 3);
      } else if (key == "local_frame_spatial_shift") {
        read_optional_values(result.local_frame_spatial_shift.data(), 3);
      } else if (key == "local_frame_quadratic") {
        read_optional_values(
            reinterpret_cast<double*>(result.local_frame_quadratic.data()), 24);
      } else {
        ERROR("Unknown optional key '" << key << "' in BBH affine map file");
      }
    }
    for (const double tau : result.local_frame_time_tilt) {
      ASSERT(std::abs(tau) <= 0.02,
             "local_frame_time_tilt must satisfy |tau_i| <= 0.02, got " << tau);
    }
    for (const double shift : result.local_frame_spatial_shift) {
      ASSERT(std::abs(shift) <= 0.01,
             "local_frame_spatial_shift must satisfy |d_i| <= 0.01, got "
                 << shift);
    }
    result.active = true;
    return result;
  }

  double boost_speed_squared() const {
    double result = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      result += boost_velocity[i] * boost_velocity[i];
    }
    ASSERT(result < 1.0,
           "The BBH local-frame boost has |v|^2 >= 1: " << result);
    return result;
  }

  double boost_gamma() const {
    const double v2 = boost_speed_squared();
    return 1.0 / sqrt(1.0 - v2);
  }

  std::array<std::array<double, 4>, 4> local_from_corot_matrix() const {
    std::array<std::array<double, 4>, 4> result{{
        {{0.0, 0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
    }};
    const double v2 = boost_speed_squared();
    const double gamma = boost_gamma();
    const std::array<std::array<double, 3>, 3> spatial_affine{{
        {{local_frame_spatial_affine_correction[0],
          local_frame_spatial_affine_correction[1],
          local_frame_spatial_affine_correction[2]}},
        {{local_frame_spatial_affine_correction[1],
          local_frame_spatial_affine_correction[3],
          local_frame_spatial_affine_correction[4]}},
        {{local_frame_spatial_affine_correction[2],
          local_frame_spatial_affine_correction[4],
          local_frame_spatial_affine_correction[5]}},
    }};
    result[0][0] = 1.0 + local_frame_coefficients[0] * v2 +
                   local_frame_coefficients[1] * companion_potential;
    for (size_t i = 0; i < 3; ++i) {
      result[0][i + 1] = local_frame_coefficients[2] * boost_velocity[i];
      result[i + 1][0] =
          local_frame_coefficients[5] * gamma * boost_velocity[i];
      for (size_t j = 0; j < 3; ++j) {
        result[i + 1][j + 1] =
            (i == j ? 1.0 + local_frame_coefficients[3] * companion_potential
                    : 0.0) +
            local_frame_coefficients[4] * boost_velocity[i] *
                boost_velocity[j] +
            spatial_affine[i][j];
      }
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t b = 0; b < 4; ++b) {
        result[0][b] -= local_frame_time_tilt[i] * result[i + 1][b];
      }
    }
    return result;
  }

  static std::array<std::array<double, 4>, 4> invert_matrix(
      const std::array<std::array<double, 4>, 4>& matrix) {
    std::array<std::array<double, 8>, 4> augmented{};
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        augmented[i][j] = matrix[i][j];
      }
      augmented[i][i + 4] = 1.0;
    }
    for (size_t pivot = 0; pivot < 4; ++pivot) {
      ASSERT(std::abs(augmented[pivot][pivot]) > 1.0e-14,
             "Singular BBH local-frame transform matrix.");
      const double inverse_pivot = 1.0 / augmented[pivot][pivot];
      for (size_t j = 0; j < 8; ++j) {
        augmented[pivot][j] *= inverse_pivot;
      }
      for (size_t row = 0; row < 4; ++row) {
        if (row == pivot) {
          continue;
        }
        const double factor = augmented[row][pivot];
        for (size_t j = 0; j < 8; ++j) {
          augmented[row][j] -= factor * augmented[pivot][j];
        }
      }
    }
    std::array<std::array<double, 4>, 4> inverse{};
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        inverse[i][j] = augmented[i][j + 4];
      }
    }
    return inverse;
  }

  static std::array<std::array<double, 3>, 3> invert_matrix(
      const std::array<std::array<double, 3>, 3>& matrix) {
    const double determinant =
        matrix[0][0] *
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
        matrix[0][1] *
            (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
        matrix[0][2] *
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    ASSERT(std::abs(determinant) > 1.0e-14,
           "Singular BBH local-frame spatial transform matrix.");
    const double inv_det = 1.0 / determinant;
    std::array<std::array<double, 3>, 3> inverse{};
    inverse[0][0] =
        inv_det * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]);
    inverse[0][1] =
        inv_det * (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]);
    inverse[0][2] =
        inv_det * (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]);
    inverse[1][0] =
        inv_det * (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]);
    inverse[1][1] =
        inv_det * (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]);
    inverse[1][2] =
        inv_det * (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]);
    inverse[2][0] =
        inv_det * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    inverse[2][1] =
        inv_det * (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]);
    inverse[2][2] =
        inv_det * (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]);
    return inverse;
  }

  std::array<std::array<double, 4>, 4> corot_from_local_matrix() const {
    return invert_matrix(local_from_corot_matrix());
  }

  std::array<std::array<double, 3>, 3> local_to_corot_spatial_matrix() const {
    const auto full_inverse = corot_from_local_matrix();
    std::array<std::array<double, 3>, 3> result{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        result[i][j] = full_inverse[i + 1][j + 1];
      }
    }
    return result;
  }

  std::array<std::array<double, 3>, 3> corot_to_local_spatial_matrix() const {
    return invert_matrix(local_to_corot_spatial_matrix());
  }

  double quadratic_coefficient(const size_t component, const size_t i,
                               const size_t j) const {
    const size_t min_i = std::min(i, j);
    const size_t max_i = std::max(i, j);
    const size_t pair_index = min_i == 0 ? max_i : (min_i == 1 ? max_i + 2 : 5);
    double coefficient = local_frame_quadratic[component][pair_index];
    if (component == 0) {
      for (size_t i = 0; i < 3; ++i) {
        coefficient -=
            local_frame_time_tilt[i] * local_frame_quadratic[i + 1][pair_index];
      }
    }
    return coefficient;
  }

  std::array<std::array<DataVector, 3>, 3> local_from_corot_spatial_jacobian(
      const tnsr::I<DataVector, 3, Frame::Inertial>& corot_coords) const {
    const auto affine_spatial = corot_to_local_spatial_matrix();
    std::array<std::array<DataVector, 3>, 3> jacobian{{
        {{DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0}}},
        {{DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0}}},
        {{DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0}}},
    }};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        jacobian[i][j] = affine_spatial[i][j];
        for (size_t k = 0; k < 3; ++k) {
          jacobian[i][j] +=
              quadratic_coefficient(i + 1, j, k) * corot_coords.get(k);
        }
      }
    }
    return jacobian;
  }

  tnsr::I<DataVector, 3, Frame::Inertial> local_to_corot_coords(
      const tnsr::I<DataVector, 3, Frame::Inertial>& local_coords) const {
    tnsr::I<DataVector, 3, Frame::Inertial> corot_coords{
        get_size(local_coords.get(0)), 0.0};
    const auto local_to_corot = local_to_corot_spatial_matrix();
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        corot_coords.get(i) +=
            local_to_corot[i][j] *
            (local_coords.get(j) + local_frame_spatial_shift[j]);
      }
    }
    for (size_t iteration = 0; iteration < 10; ++iteration) {
      const auto local_from_current = corot_to_local_coords(corot_coords);
      std::array<DataVector, 3> residual{
          {local_from_current.get(0) - local_coords.get(0),
           local_from_current.get(1) - local_coords.get(1),
           local_from_current.get(2) - local_coords.get(2)}};
      const auto jacobian = local_from_corot_spatial_jacobian(corot_coords);
      const DataVector determinant =
          jacobian[0][0] * (jacobian[1][1] * jacobian[2][2] -
                            jacobian[1][2] * jacobian[2][1]) -
          jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] -
                            jacobian[1][2] * jacobian[2][0]) +
          jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] -
                            jacobian[1][1] * jacobian[2][0]);
      ASSERT(min(abs(determinant)) > 1.0e-14,
             "Singular point-dependent BBH quadratic local-frame Jacobian.");
      std::array<DataVector, 3> step{{
          ((jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1]) *
               residual[0] +
           (jacobian[0][2] * jacobian[2][1] - jacobian[0][1] * jacobian[2][2]) *
               residual[1] +
           (jacobian[0][1] * jacobian[1][2] - jacobian[0][2] * jacobian[1][1]) *
               residual[2]) /
              determinant,
          ((jacobian[1][2] * jacobian[2][0] - jacobian[1][0] * jacobian[2][2]) *
               residual[0] +
           (jacobian[0][0] * jacobian[2][2] - jacobian[0][2] * jacobian[2][0]) *
               residual[1] +
           (jacobian[0][2] * jacobian[1][0] - jacobian[0][0] * jacobian[1][2]) *
               residual[2]) /
              determinant,
          ((jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0]) *
               residual[0] +
           (jacobian[0][1] * jacobian[2][0] - jacobian[0][0] * jacobian[2][1]) *
               residual[1] +
           (jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]) *
               residual[2]) /
              determinant,
      }};
      for (size_t i = 0; i < 3; ++i) {
        corot_coords.get(i) -= step[i];
      }
    }
    return corot_coords;
  }

  tnsr::I<DataVector, 3, Frame::Inertial> corot_to_local_coords(
      const tnsr::I<DataVector, 3, Frame::Inertial>& corot_coords) const {
    tnsr::I<DataVector, 3, Frame::Inertial> local_coords{
        get_size(corot_coords.get(0)), 0.0};
    const auto corot_to_local = corot_to_local_spatial_matrix();
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        local_coords.get(i) += corot_to_local[i][j] * corot_coords.get(j);
      }
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 3; ++k) {
          local_coords.get(i) += 0.5 * quadratic_coefficient(i + 1, j, k) *
                                 corot_coords.get(j) * corot_coords.get(k);
        }
      }
      local_coords.get(i) -= local_frame_spatial_shift[i];
    }
    return local_coords;
  }

  tnsr::I<DataVector, 3, Frame::Inertial> unboosted_map_point(
      const tnsr::I<DataVector, 3, Frame::Inertial>& corot_coords) const {
    tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{
        get_size(corot_coords.get(0))};
    for (size_t i = 0; i < 3; ++i) {
      inertial_coords.get(i) = center[i];
      for (size_t j = 0; j < 3; ++j) {
        inertial_coords.get(i) +=
            corot_to_inertial[i][j] * (corot_coords.get(j) + corot_shift[j]);
      }
    }
    return inertial_coords;
  }

  tnsr::I<DataVector, 3, Frame::Inertial> map_point(
      const tnsr::I<DataVector, 3, Frame::Inertial>& local_coords) const {
    if (not active) {
      return local_coords;
    }
    return unboosted_map_point(local_to_corot_coords(local_coords));
  }

  tnsr::I<DataVector, 3, Frame::Inertial> inverse_map_point(
      const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords) const {
    if (not active) {
      return inertial_coords;
    }
    tnsr::I<DataVector, 3, Frame::Inertial> corot_coords{
        get_size(inertial_coords.get(0)), 0.0};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        corot_coords.get(i) +=
            corot_to_inertial[j][i] * (inertial_coords.get(j) - center[j]);
      }
      corot_coords.get(i) -= corot_shift[i];
    }
    return corot_to_local_coords(corot_coords);
  }

  tnsr::I<DataVector, 3, Frame::Inertial> dt_spatial_map(
      const tnsr::I<DataVector, 3, Frame::Inertial>& corot_coords) const {
    tnsr::I<DataVector, 3, Frame::Inertial> dt_map{
        get_size(corot_coords.get(0)), 0.0};
    if (not active) {
      return dt_map;
    }
    for (size_t i = 0; i < 3; ++i) {
      dt_map.get(i) = center_velocity[i];
      for (size_t j = 0; j < 3; ++j) {
        dt_map.get(i) += corot_to_inertial_dt[i][j] * corot_coords.get(j);
        dt_map.get(i) += corot_to_inertial_dt[i][j] * corot_shift[j];
        dt_map.get(i) += corot_to_inertial[i][j] * corot_shift_velocity[j];
      }
    }
    return dt_map;
  }

  tnsr::a<DataVector, 3, Frame::Inertial> unboosted_map_covector(
      const tnsr::a<DataVector, 3, Frame::Inertial>& inertial_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& corot_coords) const {
    const auto dt_map = dt_spatial_map(corot_coords);
    tnsr::a<DataVector, 3, Frame::Inertial> corot_covector{
        get_size(inertial_covector.get(0)), 0.0};
    corot_covector.get(0) = inertial_covector.get(0);
    for (size_t i = 0; i < 3; ++i) {
      corot_covector.get(0) += dt_map.get(i) * inertial_covector.get(i + 1);
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        corot_covector.get(i + 1) +=
            corot_to_inertial[j][i] * inertial_covector.get(j + 1);
      }
    }
    return corot_covector;
  }

  tnsr::a<DataVector, 3, Frame::Inertial> map_covector(
      const tnsr::a<DataVector, 3, Frame::Inertial>& inertial_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& local_coords) const {
    if (not active) {
      return inertial_covector;
    }
    const auto corot_coords = local_to_corot_coords(local_coords);
    const auto corot_covector =
        unboosted_map_covector(inertial_covector, corot_coords);
    tnsr::a<DataVector, 3, Frame::Inertial> local_covector{
        get_size(inertial_covector.get(0)), 0.0};
    const auto corot_from_local = corot_from_local_matrix();
    for (size_t a = 0; a < 4; ++a) {
      local_covector.get(a) = corot_from_local[0][a] * corot_covector.get(0);
      for (size_t b = 1; b < 4; ++b) {
        local_covector.get(a) += corot_from_local[b][a] * corot_covector.get(b);
      }
    }
    return local_covector;
  }

  tnsr::A<DataVector, 3, Frame::Inertial> unboosted_map_vector(
      const tnsr::A<DataVector, 3, Frame::Inertial>& inertial_vector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& corot_coords) const {
    const auto dt_map = dt_spatial_map(corot_coords);
    tnsr::A<DataVector, 3, Frame::Inertial> corot_vector{
        get_size(inertial_vector.get(0)), 0.0};
    corot_vector.get(0) = inertial_vector.get(0);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        corot_vector.get(i + 1) +=
            corot_to_inertial[j][i] * (inertial_vector.get(j + 1) -
                                       dt_map.get(j) * inertial_vector.get(0));
      }
    }
    return corot_vector;
  }

  tnsr::A<DataVector, 3, Frame::Inertial> local_vector_from_corot(
      const tnsr::A<DataVector, 3, Frame::Inertial>& corot_vector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& corot_coords) const {
    tnsr::A<DataVector, 3, Frame::Inertial> local_vector{
        get_size(corot_vector.get(0)), 0.0};
    const auto local_from_corot = local_from_corot_matrix();
    for (size_t a = 0; a < 4; ++a) {
      local_vector.get(a) = local_from_corot[a][0] * corot_vector.get(0);
      for (size_t b = 1; b < 4; ++b) {
        DataVector jacobian_component = make_with_value<DataVector>(
            corot_vector.get(0), local_from_corot[a][b]);
        for (size_t k = 0; k < 3; ++k) {
          jacobian_component +=
              quadratic_coefficient(a, b - 1, k) * corot_coords.get(k);
        }
        local_vector.get(a) += jacobian_component * corot_vector.get(b);
      }
    }
    return local_vector;
  }

  tnsr::A<DataVector, 3, Frame::Inertial> map_vector(
      const tnsr::A<DataVector, 3, Frame::Inertial>& inertial_vector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& local_coords) const {
    if (not active) {
      return inertial_vector;
    }
    const auto corot_coords = local_to_corot_coords(local_coords);
    return local_vector_from_corot(
        unboosted_map_vector(inertial_vector, corot_coords), corot_coords);
  }

  template <typename InverseSpacetimeMetric>
  tnsr::A<DataVector, 3, Frame::Inertial> wave_map_correction(
      const InverseSpacetimeMetric& inverse_spacetime_metric,
      const tnsr::I<DataVector, 3, Frame::Inertial>& local_coords) const {
    tnsr::A<DataVector, 3, Frame::Inertial> correction{
        get_size(local_coords.get(0)), 0.0};
    if (not active) {
      return correction;
    }
    const auto corot_coords = local_to_corot_coords(local_coords);

    std::array<DataVector, 3> x_minus_center{
        {DataVector{get_size(corot_coords.get(0)), 0.0},
         DataVector{get_size(corot_coords.get(0)), 0.0},
         DataVector{get_size(corot_coords.get(0)), 0.0}}};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        x_minus_center[i] +=
            corot_to_inertial[i][j] * (corot_coords.get(j) + corot_shift[j]);
      }
    }

    for (size_t i = 0; i < 3; ++i) {
      DataVector d2_corot_coord{get_size(corot_coords.get(0)), 0.0};
      for (size_t j = 0; j < 3; ++j) {
        // y^i = R^T{}^i_j (x^j - C^j) - S^i. This is
        // d_t^2 y^i at fixed inertial x.
        d2_corot_coord += corot_to_inertial_ddt[j][i] * x_minus_center[j];
        d2_corot_coord -= 2.0 * corot_to_inertial_dt[j][i] * center_velocity[j];
        d2_corot_coord -= corot_to_inertial[j][i] * center_acceleration[j];
      }
      d2_corot_coord -= corot_shift_acceleration[i];
      correction.get(i + 1) =
          inverse_spacetime_metric.get(0, 0) * d2_corot_coord;
      for (size_t j = 0; j < 3; ++j) {
        correction.get(i + 1) += 2.0 * inverse_spacetime_metric.get(0, j + 1) *
                                 corot_to_inertial_dt[j][i];
      }
    }
    auto local_correction = local_vector_from_corot(correction, corot_coords);

    // The optimized local-frame map can include
    //   z^a = M^a_b y^b + 1/2 Q^a_ij y^i y^j,
    // so Box z^a contains the additional term g_y^{ij} Q^a_ij.  The
    // rotation/translation contribution above accounts only for Box y^a.
    const auto dt_map = dt_spatial_map(corot_coords);
    std::array<std::array<DataVector, 3>, 3> inverse_corot_spatial_metric{{
        {{DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0}}},
        {{DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0}}},
        {{DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0},
          DataVector{get_size(corot_coords.get(0)), 0.0}}},
    }};
    std::array<DataVector, 3> dy_dt{
        {DataVector{get_size(corot_coords.get(0)), 0.0},
         DataVector{get_size(corot_coords.get(0)), 0.0},
         DataVector{get_size(corot_coords.get(0)), 0.0}}};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        dy_dt[i] -= corot_to_inertial[j][i] * dt_map.get(j);
      }
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        auto& gij = inverse_corot_spatial_metric[i][j];
        gij += dy_dt[i] * dy_dt[j] * inverse_spacetime_metric.get(0, 0);
        for (size_t k = 0; k < 3; ++k) {
          gij += dy_dt[i] * corot_to_inertial[k][j] *
                 inverse_spacetime_metric.get(0, k + 1);
          gij += corot_to_inertial[k][i] * dy_dt[j] *
                 inverse_spacetime_metric.get(k + 1, 0);
          for (size_t l = 0; l < 3; ++l) {
            gij += corot_to_inertial[k][i] * corot_to_inertial[l][j] *
                   inverse_spacetime_metric.get(k + 1, l + 1);
          }
        }
      }
    }
    for (size_t a = 0; a < 4; ++a) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          local_correction.get(a) += inverse_corot_spatial_metric[i][j] *
                                     quadratic_coefficient(a, i, j);
        }
      }
    }
    return local_correction;
  }
};

}  // namespace DQ::detail
