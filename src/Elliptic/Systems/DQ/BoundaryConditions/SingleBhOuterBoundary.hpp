// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/DQ/BbhAffineMap.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace DQ::BoundaryConditions {

/*!
 * \brief Outer Dirichlet boundary condition derived from a precomputed
 * SingleBH `K=const` surface.
 *
 * \details The surface is represented as a `ylm::Strahlkorper` written to an
 * H5 Ylm subfile by a precompute step that uses the native SingleBH volume data
 * at a selected time. On the DQ outer boundary this condition imposes
 * \f[
 *   \xi_t = 0,\qquad
 *   \xi_i = X^{\mathrm{surface}}_i - X^{\mathrm{DQ}}_i,
 * \f]
 * where \(X^{\mathrm{surface}}_i\) is evaluated from the precomputed
 * Strahlkorper at the angular location of the DQ boundary point.
 */
template <size_t Dim>
class SingleBhOuterBoundary
    : public elliptic::BoundaryConditions::BoundaryCondition<Dim> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<Dim>;

 public:
  static constexpr Options::String help =
      "Outer Dirichlet boundary condition for Xi_a derived from a precomputed "
      "SingleBH GaussBonnetScalar isosurface written as Ylm coefficients. It "
      "imposes Xi_t = 0 and Xi_i = X_surface^i - X_DQ^i on the DQ outer "
      "boundary.";

  struct SurfaceH5File {
    using type = std::string;
    static constexpr Options::String help =
        "H5 file containing the precomputed SingleBH outer-surface Ylm data.";
  };
  struct SurfaceSubfile {
    using type = std::string;
    static constexpr Options::String help =
        "Dat subfile in 'SurfaceH5File' that stores the surface Ylm data.";
  };
  struct MatchTime {
    using type = double;
    static constexpr Options::String help =
        "Time at which to read the precomputed surface from the Ylm file.";
  };
  struct MatchTimeEpsilon {
    using type = double;
    static constexpr Options::String help =
        "Tolerance for matching 'MatchTime' in the surface Ylm file.";
  };
  struct CheckFrame {
    using type = bool;
    static constexpr Options::String help =
        "Whether to verify the frame metadata in the stored Ylm file.";
  };
  struct AffineMapFile {
    using type = std::string;
    static constexpr Options::String help =
        "Optional text file describing the corotating-to-inertial affine map "
        "used to evaluate the stored inertial BBH surface on the DQ outer "
        "boundary. Leave empty for the identity map.";
  };

  using options = tmpl::list<SurfaceH5File, SurfaceSubfile, MatchTime,
                             MatchTimeEpsilon, CheckFrame, AffineMapFile>;

  SingleBhOuterBoundary() = default;
  SingleBhOuterBoundary(const SingleBhOuterBoundary&) = default;
  SingleBhOuterBoundary& operator=(const SingleBhOuterBoundary&) = default;
  SingleBhOuterBoundary(SingleBhOuterBoundary&&) = default;
  SingleBhOuterBoundary& operator=(SingleBhOuterBoundary&&) = default;
  ~SingleBhOuterBoundary() override = default;

  /// \cond
  explicit SingleBhOuterBoundary(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SingleBhOuterBoundary);
  /// \endcond

  SingleBhOuterBoundary(std::string surface_h5_file,
                        std::string surface_subfile, double match_time,
                        double match_time_epsilon, bool check_frame,
                        std::string affine_map_file,
                        const Options::Context& context = {});

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<SingleBhOuterBoundary>(*this);
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return std::vector<elliptic::BoundaryConditionType>(
        Dim + 1, elliptic::BoundaryConditionType::Dirichlet);
  }

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                     Dim, Frame::Inertial>>>;
  using volume_tags = tmpl::list<>;

  void apply(
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi,
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> n_dot_flux_xi,
      const tnsr::ia<DataVector, Dim, Frame::Inertial>& deriv_xi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal) const;

  using argument_tags_linearized =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                     Dim, Frame::Inertial>>>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi_correction,
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
          n_dot_flux_xi_correction,
      const tnsr::ia<DataVector, Dim, Frame::Inertial>& deriv_xi_correction,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal) const;

  void pup(PUP::er& p) override;

  const std::string& surface_h5_file() const { return surface_h5_file_; }
  const std::string& surface_subfile() const { return surface_subfile_; }
  double match_time() const { return match_time_; }
  double match_time_epsilon() const { return match_time_epsilon_; }
  bool check_frame() const { return check_frame_; }
  const std::string& affine_map_file() const { return affine_map_file_; }

 private:
  std::string surface_h5_file_{};
  std::string surface_subfile_{};
  double match_time_{std::numeric_limits<double>::signaling_NaN()};
  double match_time_epsilon_{std::numeric_limits<double>::signaling_NaN()};
  bool check_frame_{true};
  std::string affine_map_file_{};
  DQ::detail::BbhAffineMap affine_map_{};
  ylm::Strahlkorper<Frame::Inertial> surface_{};
};

template <size_t Dim>
bool operator==(const SingleBhOuterBoundary<Dim>& lhs,
                const SingleBhOuterBoundary<Dim>& rhs);

template <size_t Dim>
bool operator!=(const SingleBhOuterBoundary<Dim>& lhs,
                const SingleBhOuterBoundary<Dim>& rhs);

}  // namespace DQ::BoundaryConditions
