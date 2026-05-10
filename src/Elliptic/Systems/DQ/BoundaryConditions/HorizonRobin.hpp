// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/DQ/BbhAffineMap.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/NumericData.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace DQ::BoundaryConditions {

/*!
 * \brief Near-horizon boundary condition for the coordinate correction
 * \f$\xi_a\f$ on an inner excision surface at \f$r_\mathrm{in} > 2m\f$.
 *
 * \details This imposes the full surface equation
 *
 * \f{align*}
 * \partial_r \xi_t + \frac{1}{2m}\Delta_{S^2}\xi_t &= -1 + 2m H^t \\
 * \partial_r \xi_i + \frac{1}{2m}\Delta_{S^2}\xi_i &= n_i + 2m H^i
 * \f}
 *
 * converted to the DQ normal flux
 * \f$n_j F^j{}_a = (1 - 2m/r) \partial_r \xi_a\f$,
 * where the surface Laplacian is evaluated on each boundary face using the
 * induced 2-metric from the face embedding.
 */
template <size_t Dim>
class HorizonRobin
    : public elliptic::BoundaryConditions::BoundaryCondition<Dim> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<Dim>;

 public:
  static constexpr Options::String help =
      "Near-horizon boundary condition for Xi_a on an inner boundary at r > "
      "2m, imposing the horizon-regular Robin condition for all components.";

  struct Mass {
    using type = double;
    static constexpr Options::String help =
        "The mass m appearing in the near-horizon regular boundary condition.";
  };
  struct FileGlob {
    using type = std::string;
    static constexpr Options::String help =
        "Path or glob pattern to volume data containing GaugeH_a and "
        "SpacetimeMetric. Leave empty when UseGaugeH is false.";
  };
  struct Subgroup {
    using type = std::string;
    static constexpr Options::String help =
        "The subgroup within the volume data file, excluding extensions.";
  };
  struct ObservationStep {
    using type = int;
    static constexpr Options::String help =
        "Observation step used to read GaugeH_a and SpacetimeMetric.";
  };
  struct ExtrapolateIntoExcisions {
    using type = bool;
    static constexpr Options::String help =
        "Whether to extrapolate GaugeH_a and SpacetimeMetric into excised "
        "regions.";
  };
  struct UseGaugeH {
    using type = bool;
    static constexpr Options::String help =
        "If true, use -1 + 2m H^t and n_i + 2m H^i in the horizon Robin "
        "condition. H_a is read from volume data, raised with the observed "
        "metric, and transformed as a vector to the DQ frame.";
  };
  struct AffineMapFile {
    using type = std::string;
    static constexpr Options::String help =
        "Optional text file describing the corotating-to-inertial affine map. "
        "Leave empty for the identity map.";
  };
  struct ZeroSource {
    using type = bool;
    static constexpr Options::String help =
        "If true, impose the homogeneous horizon-regular condition, i.e. "
        "use zero source on the right-hand side for all components.";
  };
  using options =
      tmpl::list<Mass, FileGlob, Subgroup, ObservationStep,
                 ExtrapolateIntoExcisions, UseGaugeH, AffineMapFile,
                 ZeroSource>;

  HorizonRobin() = default;
  HorizonRobin(const HorizonRobin&) = default;
  HorizonRobin& operator=(const HorizonRobin&) = default;
  HorizonRobin(HorizonRobin&&) = default;
  HorizonRobin& operator=(HorizonRobin&&) = default;
  ~HorizonRobin() override = default;

  /// \cond
  explicit HorizonRobin(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(HorizonRobin);
  /// \endcond

  HorizonRobin(const double mass, std::string file_glob, std::string subgroup,
               const int observation_step,
               const bool extrapolate_into_excisions, const bool use_gauge_h,
               std::string affine_map_file, const bool zero_source)
      : mass_(mass),
        numeric_data_(std::move(file_glob), std::move(subgroup),
                      observation_step, extrapolate_into_excisions),
        use_gauge_h_(use_gauge_h),
        affine_map_file_(std::move(affine_map_file)),
        affine_map_(DQ::detail::BbhAffineMap::from_file(affine_map_file_)),
        zero_source_(zero_source) {
    ASSERT(Dim == 3, "HorizonRobin is implemented only in 3D.");
  }

  double mass() const { return mass_; }

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<HorizonRobin>(*this);
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return std::vector<elliptic::BoundaryConditionType>(
        Dim + 1, elliptic::BoundaryConditionType::Neumann);
  }

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                     Dim, Frame::Inertial>>,
                 domain::Tags::Direction<Dim>, domain::Tags::Mesh<Dim>>;
  using volume_tags = tmpl::list<domain::Tags::Mesh<Dim>>;

  void apply(
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi,
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> n_dot_flux_xi,
      const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal,
      const Direction<Dim>& direction, const Mesh<Dim>& volume_mesh) const;

  void apply(
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi,
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> n_dot_flux_xi,
      const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal,
      const Direction<Dim>& direction,
      const DirectionalIdMap<Dim, Mesh<Dim>>& volume_meshes) const;

  using argument_tags_linearized =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                     Dim, Frame::Inertial>>,
                 domain::Tags::Direction<Dim>, domain::Tags::Mesh<Dim>>;
  using volume_tags_linearized = tmpl::list<domain::Tags::Mesh<Dim>>;

  void apply_linearized(
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi_correction,
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
          n_dot_flux_xi_correction,
      const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi_correction*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal,
      const Direction<Dim>& direction, const Mesh<Dim>& volume_mesh) const;

  void apply_linearized(
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi_correction,
      gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
          n_dot_flux_xi_correction,
      const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi_correction*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& face_inertial_coords,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal,
      const Direction<Dim>& direction,
      const DirectionalIdMap<Dim, Mesh<Dim>>& volume_meshes) const;

  void pup(PUP::er& p) override;

 private:
  tnsr::A<DataVector, Dim, Frame::Inertial> gauge_h(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords) const;

  double mass_ = std::numeric_limits<double>::signaling_NaN();
  ::NumericData numeric_data_{};
  bool use_gauge_h_{false};
  std::string affine_map_file_{};
  DQ::detail::BbhAffineMap affine_map_{};
  bool zero_source_{false};
};

template <size_t Dim>
bool operator==(const HorizonRobin<Dim>& lhs, const HorizonRobin<Dim>& rhs);

template <size_t Dim>
bool operator!=(const HorizonRobin<Dim>& lhs, const HorizonRobin<Dim>& rhs);

}  // namespace DQ::BoundaryConditions
