// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/RadiallyCompressedCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/RunEventsAndTriggers.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/Executables/Solver.hpp"
#include "Elliptic/Systems/DQ/BbhAffineMap.hpp"
#include "Elliptic/Systems/DQ/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/DQ/FirstOrderSystem.hpp"
#include "Elliptic/Systems/DQ/Tags.hpp"
#include "Elliptic/Triggers/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendAmrDiagnostics.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Factory.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/DQ/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "PointwiseFunctions/InitialDataUtilities/NumericData.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

#include <optional>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/CallWithDynamicType.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP

namespace DQ::Backgrounds {

namespace detail {

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

}  // namespace detail

template <size_t Dim>
class SingleBhGaugeH final : public elliptic::analytic_data::AnalyticSolution {
  static_assert(Dim == 3,
                "The SingleBhGaugeH background is only implemented in 3D.");

 public:
  struct PlusConstant {
    using type = double;
    static constexpr Options::String help{"Constant added to the solution."};
  };
  struct FileGlob {
    using type = std::string;
    static constexpr Options::String help =
        "Path or glob pattern to the SingleBH volume data file.";
  };
  struct Subgroup {
    using type = std::string;
    static constexpr Options::String help =
        "The subgroup within the SingleBH volume file, excluding extensions.";
  };
  struct ObservationStep {
    using type = int;
    static constexpr Options::String help =
        "Observation step in the SingleBH volume file used to read GaugeH_a "
        "and SpacetimeMetric.";
  };
  struct ExtrapolateIntoExcisions {
    using type = bool;
    static constexpr Options::String help =
        "Whether to extrapolate SingleBH GaugeH_t into excised regions.";
  };
  struct UseSingleBhGaugeH {
    using type = bool;
    static constexpr Options::String help =
        "If true, raise the observed covariant GaugeH_a to H^a, transform "
        "H^a to the DQ frame, add the wave-map correction from the "
        "time-dependent corotating affine map when present, and use sources "
        "-H^t + 2m/r^2 and -H^i - 2m/r^2 n^i. If false, use 2m/r^2 and "
        "-2m/r^2 n^i.";
  };
  struct Mass {
    using type = double;
    static constexpr Options::String help =
        "The mass m used in the source term 2m/r^2.";
  };
  struct AffineMapFile {
    using type = std::string;
    static constexpr Options::String help =
        "Optional text file describing the corotating-to-inertial affine map "
        "used to pull BBH volume data into the DQ coordinates. Leave empty "
        "for the identity map.";
  };
  struct ZeroSource {
    using type = bool;
    static constexpr Options::String help =
        "If true, ignore GaugeH and the analytic Kerr-Schild source and use "
        "FixedSource(Xi_a)=0. This is intended for homogeneous solves such as "
        "dtXi_a.";
  };
  struct DtXiFileGlob {
    using type = std::string;
    static constexpr Options::String help =
        "Optional path or glob pattern to DQ volume data containing dtXi_a. "
        "Leave empty when UseDtXi is false.";
  };
  struct DtXiSubgroup {
    using type = std::string;
    static constexpr Options::String help =
        "The subgroup in the dtXi volume file, excluding extensions.";
  };
  struct DtXiObservationStep {
    using type = int;
    static constexpr Options::String help =
        "Observation step used to read dtXi_a.";
  };
  struct DtXiExtrapolateIntoExcisions {
    using type = bool;
    static constexpr Options::String help =
        "Whether to extrapolate dtXi_a into excised regions.";
  };
  struct UseDtXi {
    using type = bool;
    static constexpr Options::String help =
        "If true, add +(2m/r^2) dtXi_a to the fixed source.";
  };

  using options =
      tmpl::list<PlusConstant, FileGlob, Subgroup, ObservationStep,
                 ExtrapolateIntoExcisions, UseSingleBhGaugeH, Mass,
                 AffineMapFile, ZeroSource, DtXiFileGlob, DtXiSubgroup,
                 DtXiObservationStep, DtXiExtrapolateIntoExcisions, UseDtXi>;
  static constexpr Options::String help =
      "Analytic xi_a solution with a source that is either "
      "(2m/r^2, -2m/r^2 n^i) or (-H^t + 2m/r^2, "
      "-H^i - 2m/r^2 n^i). The observed GaugeH_a is raised with the "
      "observed spacetime metric before being transformed by the optional "
      "affine map, including g^{AB} d_A d_B y when map acceleration data is "
      "available.";

  SingleBhGaugeH() = default;
  SingleBhGaugeH(const SingleBhGaugeH&) = default;
  SingleBhGaugeH& operator=(const SingleBhGaugeH&) = default;
  SingleBhGaugeH(SingleBhGaugeH&&) = default;
  SingleBhGaugeH& operator=(SingleBhGaugeH&&) = default;
  ~SingleBhGaugeH() override = default;

  SingleBhGaugeH(const double plus_constant, std::string file_glob,
                 std::string subgroup, const int observation_step,
                 const bool extrapolate_into_excisions,
                 const bool use_single_bh_gauge_h, const double mass,
                 std::string affine_map_file, const bool zero_source,
                 std::string dtxi_file_glob, std::string dtxi_subgroup,
                 const int dtxi_observation_step,
                 const bool dtxi_extrapolate_into_excisions,
                 const bool use_dtxi)
      : lorentzian_solution_(mass, plus_constant),
        numeric_data_(std::move(file_glob), std::move(subgroup),
                      observation_step, extrapolate_into_excisions),
        use_single_bh_gauge_h_(use_single_bh_gauge_h),
        mass_(mass),
        affine_map_file_(std::move(affine_map_file)),
        affine_map_(DQ::detail::BbhAffineMap::from_file(affine_map_file_)),
        zero_source_(zero_source),
        dtxi_data_(std::move(dtxi_file_glob), std::move(dtxi_subgroup),
                   dtxi_observation_step, dtxi_extrapolate_into_excisions),
        use_dtxi_(use_dtxi) {}

  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<SingleBhGaugeH>(*this);
  }

  /// \cond
  explicit SingleBhGaugeH(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SingleBhGaugeH);
  /// \endcond

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return {compute_variable<RequestedTags>(x)...};
  }

  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    p | lorentzian_solution_;
    numeric_data_.pup(p);
    p | use_single_bh_gauge_h_;
    p | mass_;
    p | affine_map_file_;
    p | zero_source_;
    dtxi_data_.pup(p);
    p | use_dtxi_;
    if (p.isUnpacking()) {
      affine_map_ = DQ::detail::BbhAffineMap::from_file(affine_map_file_);
    }
  }

 private:
  tnsr::a<DataVector, Dim, Frame::Inertial> dtxi(
      const tnsr::I<DataVector, Dim>& x) const {
    tnsr::a<DataVector, Dim, Frame::Inertial> result{get_size(x.get(0)), 0.0};
    if (not use_dtxi_) {
      return result;
    }
    return tuples::get<DQ::Tags::Xi<DataVector, Dim>>(
        dtxi_data_.variables(x, tmpl::list<DQ::Tags::Xi<DataVector, Dim>>{}));
  }

  template <typename RequestedTag>
  typename RequestedTag::type compute_variable(
      const tnsr::I<DataVector, Dim>& x) const {
    static_assert(
        std::is_same_v<RequestedTag,
                       ::Tags::FixedSource<DQ::Tags::Xi<DataVector, Dim>>>,
        "SingleBhGaugeH only supports the DQ fixed source tag.");
    const DataVector r2 = get(dot_product(x, x));
    const DataVector r = sqrt(r2);
    tnsr::a<DataVector, Dim, Frame::Inertial> result{get_size(r2)};
    if (zero_source_) {
      for (size_t a = 0; a < Dim + 1; ++a) {
        result.get(a) = 0.0;
      }
      return result;
    }
    const auto source_coords = affine_map_.map_point(x);
    if (use_single_bh_gauge_h_) {
      const auto gauge_h_vars = numeric_data_.variables(
          source_coords,
          tmpl::list<
              gh::Tags::GaugeH<DataVector, Dim>,
              gr::Tags::SpacetimeMetric<DataVector, Dim, Frame::Inertial>>{});
      auto gauge_h_inertial =
          tuples::get<gh::Tags::GaugeH<DataVector, Dim>>(gauge_h_vars);
      auto spacetime_metric = tuples::get<
          gr::Tags::SpacetimeMetric<DataVector, Dim, Frame::Inertial>>(
          gauge_h_vars);
      detail::sanitize_gauge_h_and_metric(make_not_null(&gauge_h_inertial),
                                          make_not_null(&spacetime_metric));
      const auto det_and_inverse = determinant_and_inverse(spacetime_metric);
      const auto& inverse_spacetime_metric = det_and_inverse.second;
      tnsr::A<DataVector, Dim, Frame::Inertial> gauge_h_raised{get_size(r2),
                                                               0.0};
      for (size_t a = 0; a < Dim + 1; ++a) {
        for (size_t b = 0; b < Dim + 1; ++b) {
          gauge_h_raised.get(a) +=
              inverse_spacetime_metric.get(a, b) * gauge_h_inertial.get(b);
        }
      }
      auto gauge_h = affine_map_.map_vector(gauge_h_raised, x);
      const auto wave_map_correction =
          affine_map_.wave_map_correction(inverse_spacetime_metric, x);
      for (size_t a = 0; a < Dim + 1; ++a) {
        gauge_h.get(a) += wave_map_correction.get(a);
      }
      const auto dtxi_value = dtxi(x);
      result.get(0) = -gauge_h.get(0) + 2.0 * mass_ / r2 +
                      2.0 * mass_ * dtxi_value.get(0) / r2;
      for (size_t d = 0; d < Dim; ++d) {
        result.get(d + 1) = -gauge_h.get(d + 1) -
                            2.0 * mass_ * x.get(d) / (r2 * r) +
                            2.0 * mass_ * dtxi_value.get(d + 1) / r2;
      }
    } else {
      result.get(0) = 2.0 * mass_ / r2;
      for (size_t d = 0; d < Dim; ++d) {
        result.get(d + 1) = -2.0 * mass_ * x.get(d) / (r2 * r);
      }
    }
    return result;
  }

  template <typename RequestedTag>
  typename RequestedTag::type compute_variable(
      const tnsr::I<DataVector, Dim>& x) const
    requires(not std::is_same_v<
             RequestedTag, ::Tags::FixedSource<DQ::Tags::Xi<DataVector, Dim>>>)
  {
    return tuples::get<RequestedTag>(
        lorentzian_solution_.variables(x, tmpl::list<RequestedTag>{}));
  }

  DQ::Solutions::Lorentzian<Dim, DataVector> lorentzian_solution_{};
  ::NumericData numeric_data_{};
  bool use_single_bh_gauge_h_{false};
  double mass_{std::numeric_limits<double>::signaling_NaN()};
  std::string affine_map_file_{};
  DQ::detail::BbhAffineMap affine_map_{};
  bool zero_source_{false};
  ::NumericData dtxi_data_{};
  bool use_dtxi_{false};
};

template <size_t Dim>
PUP::able::PUP_ID SingleBhGaugeH<Dim>::my_PUP_ID = 0;  // NOLINT

}  // namespace DQ::Backgrounds

namespace DQ::Actions {

template <size_t Dim, typename BackgroundTag, typename FixedSourcesTag>
struct SetObservedSource : tt::ConformsTo<::amr::protocols::Projector> {
  using const_global_cache_tags = tmpl::list<BackgroundTag>;
  using simple_tags = tmpl::list<DQ::Tags::ObservedSource<DataVector, Dim>>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<SetObservedSource>(make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

  using return_tags = tmpl::list<DQ::Tags::ObservedSource<DataVector, Dim>>;
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>, BackgroundTag,
                 Parallel::Tags::Metavariables>;

  template <typename Background, typename Metavariables, typename... AmrData>
  static void apply(
      const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
          observed_source,
      const tnsr::I<DataVector, Dim>& inertial_coords,
      const Background& background, const Metavariables& /*meta*/,
      const AmrData&... /*amr_data*/) {
    using factory_classes =
        typename std::decay_t<Metavariables>::factory_creation::factory_classes;
    const auto fixed_sources =
        call_with_dynamic_type<Variables<typename FixedSourcesTag::tags_list>,
                               tmpl::at<factory_classes, Background>>(
            &background, [&inertial_coords](const auto* const derived) {
              return variables_from_tagged_tuple(derived->variables(
                  inertial_coords, typename FixedSourcesTag::tags_list{}));
            });
    *observed_source =
        get<::Tags::FixedSource<DQ::Tags::Xi<DataVector, Dim>>>(fixed_sources);
  }
};

}  // namespace DQ::Actions

template <size_t Dim>
struct Metavariables {
  static constexpr Options::String help{
      "Find the solution to a Poisson problem."};

  static constexpr size_t volume_dim = Dim;
  using system =
      DQ::FirstOrderSystem<Dim, DQ::Geometry::FlatCartesian>;
  using solver = elliptic::Solver<Metavariables, Dim, system>;

  using analytic_solution_fields = typename system::primal_fields;
  using error_compute = ::Tags::ErrorsCompute<analytic_solution_fields>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_solution_fields>;
  using observe_fields = tmpl::append<
      analytic_solution_fields, error_tags,
      typename solver::fixed_sources_tag::tags_list,
      tmpl::list<DQ::Tags::ObservedSource<DataVector, volume_dim>>,
      tmpl::list<domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                 ::Events::Tags::ObserverDetInvJacobianCompute<
                     Frame::ElementLogical, Frame::Inertial>,
                 domain::Tags::RadiallyCompressedCoordinatesCompute<
                     volume_dim, Frame::Inertial>>>;
  using observer_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 error_compute>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<domain::Tags::RadiallyCompressedCoordinatesOptions,
                 DQ::Tags::Mass>;

  using supported_dq_solutions =
      tmpl::list<DQ::Solutions::Zero<volume_dim>,
                 DQ::Solutions::Lorentzian<volume_dim>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            elliptic::analytic_data::Background,
            tmpl::push_back<supported_dq_solutions,
                            elliptic::analytic_data::NumericData,
                            DQ::Backgrounds::SingleBhGaugeH<volume_dim>>>,
        tmpl::pair<elliptic::analytic_data::InitialGuess,
                   tmpl::push_back<supported_dq_solutions,
                                   DQ::Backgrounds::SingleBhGaugeH<volume_dim>,
                                   elliptic::analytic_data::NumericData>>,
        tmpl::pair<
            elliptic::analytic_data::AnalyticSolution,
            tmpl::push_back<supported_dq_solutions,
                            DQ::Backgrounds::SingleBhGaugeH<volume_dim>>>,
        tmpl::pair<
            ::MathFunction<volume_dim, Frame::Inertial>,
            MathFunctions::all_math_functions<volume_dim, Frame::Inertial>>,
        tmpl::pair<
            elliptic::BoundaryConditions::BoundaryCondition<volume_dim>,
            DQ::BoundaryConditions::standard_boundary_conditions<system>>,
        tmpl::pair<
            ::amr::Criterion,
            ::amr::Criteria::standard_criteria<
                volume_dim, tmpl::list<DQ::Tags::Xi<DataVector, volume_dim>>>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       dg::Events::field_observations<
                           volume_dim, observe_fields, observer_compute_tags,
                           ::amr::Tags::IsFinestGrid>>>>,
        tmpl::pair<Trigger, elliptic::Triggers::all_triggers<
                                ::amr::OptionTags::AmrGroup>>,
        tmpl::pair<
            PhaseChange,
            tmpl::list<
                // Phases for AMR
                PhaseControl::VisitAndReturn<
                    Parallel::Phase::EvaluateAmrCriteria>,
                PhaseControl::VisitAndReturn<Parallel::Phase::AdjustDomain>,
                PhaseControl::VisitAndReturn<Parallel::Phase::UpdateSections>,
                PhaseControl::VisitAndReturn<Parallel::Phase::CheckDomain>>>>;
  };

  // Collect all reduction tags for observers
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          solver>>>;

  using initialization_actions =
      tmpl::push_back<typename solver::initialization_actions,
                      DQ::Actions::SetObservedSource<
                          volume_dim, typename solver::background_tag,
                          typename solver::fixed_sources_tag>,
                      Parallel::Actions::TerminatePhase>;

  using register_actions =
      tmpl::push_back<typename solver::register_actions,
                      observers::Actions::RegisterEventsWithObservers>;

  using solve_actions =
      tmpl::push_front<typename solver::template solve_actions<tmpl::list<>>,
                       DQ::Actions::SetObservedSource<
                           volume_dim, typename solver::background_tag,
                           typename solver::fixed_sources_tag>>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        initialization_actions>,
                 Parallel::PhaseActions<
                     Parallel::Phase::Register,
                     tmpl::push_back<register_actions,
                                     Parallel::Actions::TerminatePhase>>,
                 Parallel::PhaseActions<
                     Parallel::Phase::Restart,
                     tmpl::push_back<register_actions,
                                     Parallel::Actions::TerminatePhase>>,
                 Parallel::PhaseActions<Parallel::Phase::Solve, solve_actions>,
                 Parallel::PhaseActions<
                     Parallel::Phase::CheckDomain,
                     tmpl::list<::amr::Actions::SendAmrDiagnostics,
                                Parallel::Actions::TerminatePhase>>>,
      LinearSolver::multigrid::ElementsAllocator<
          volume_dim, typename solver::multigrid::options_group>>;

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = dg_element_array;
    using projectors =
        tmpl::push_back<typename solver::amr_projectors,
                        DQ::Actions::SetObservedSource<
                            volume_dim, typename solver::background_tag,
                            typename solver::fixed_sources_tag>>;
    static constexpr bool keep_coarse_grids = true;
    static constexpr bool p_refine_only_in_event = false;
  };

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array, register_actions>>;
  };

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::flatten<
      tmpl::list<dg_element_array, typename solver::component_list,
                 observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>>>;

  static constexpr std::array<Parallel::Phase, 6> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::UpdateSections, Parallel::Phase::CheckDomain,
       Parallel::Phase::Solve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
/// \endcond
