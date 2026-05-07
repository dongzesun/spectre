// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/DQ/BoundaryConditions/SingleBhOuterBoundary.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

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
}  // namespace

template <size_t Dim>
SingleBhOuterBoundary<Dim>::SingleBhOuterBoundary(
    std::string surface_h5_file, std::string surface_subfile,
    const double match_time, const double match_time_epsilon,
    const bool check_frame, std::string affine_map_file,
    const Options::Context& context)
    : surface_h5_file_(std::move(surface_h5_file)),
      surface_subfile_(std::move(surface_subfile)),
      match_time_(match_time),
      match_time_epsilon_(match_time_epsilon),
      check_frame_(check_frame),
      affine_map_file_(std::move(affine_map_file)),
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
}

template <size_t Dim>
void SingleBhOuterBoundary<Dim>::apply(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
    /*n_dot_flux_xi*/,
    const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*face_normal*/) const {
  const auto inertial_face_coords = affine_map_.map_point(face_inertial_coords);
  const auto theta_phi =
      theta_phi_of(inertial_face_coords, surface_.expansion_center());
  const auto interpolation_info =
      surface_.ylm_spherepack().set_up_interpolation_info(theta_phi);
  DataVector surface_radius{get_size(face_inertial_coords.get(0))};
  surface_.ylm_spherepack().interpolate_from_coefs(
      make_not_null(&surface_radius), surface_.coefficients(),
      interpolation_info);

  xi->get(0) = make_with_value<DataVector>(surface_radius, 0.0);
  const auto& center = surface_.expansion_center();
  const DataVector dx = inertial_face_coords.get(0) - center[0];
  const DataVector dy = inertial_face_coords.get(1) - center[1];
  const DataVector dz = inertial_face_coords.get(2) - center[2];
  const DataVector inertial_radius = sqrt(dx * dx + dy * dy + dz * dz);
  tnsr::I<DataVector, Dim, Frame::Inertial> inertial_surface_coords{
      get_size(face_inertial_coords.get(0))};
  inertial_surface_coords.get(0) =
      center[0] + surface_radius * dx / inertial_radius;
  inertial_surface_coords.get(1) =
      center[1] + surface_radius * dy / inertial_radius;
  inertial_surface_coords.get(2) =
      center[2] + surface_radius * dz / inertial_radius;
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
  if (p.isUnpacking()) {
    affine_map_ = DQ::detail::BbhAffineMap::from_file(affine_map_file_);
  }
  p | surface_;
}

template <size_t Dim>
bool operator==(const SingleBhOuterBoundary<Dim>& lhs,
                const SingleBhOuterBoundary<Dim>& rhs) {
  return lhs.surface_h5_file() == rhs.surface_h5_file() and
         lhs.surface_subfile() == rhs.surface_subfile() and
         lhs.match_time() == rhs.match_time() and
         lhs.match_time_epsilon() == rhs.match_time_epsilon() and
         lhs.check_frame() == rhs.check_frame() and
         lhs.affine_map_file() == rhs.affine_map_file();
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
