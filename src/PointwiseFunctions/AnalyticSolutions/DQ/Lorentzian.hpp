// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/DQ/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace DQ::Solutions {

namespace detail {
template <typename DataType, size_t Dim>
struct LorentzianVariables {
  using Cache = CachedTempBuffer<
      Tags::Xi<DataType, Dim>,
      ::Tags::deriv<Tags::Xi<DataType, Dim>, tmpl::size_t<Dim>,
                    Frame::Inertial>,
      ::Tags::Flux<Tags::Xi<DataType, Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::FixedSource<Tags::Xi<DataType, Dim>>>;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const tnsr::I<DataVector, Dim>& x;
  double mass;
  double constant;
  double complex_phase;

  void operator()(gsl::not_null<tnsr::a<DataType, Dim, Frame::Inertial>*> xi,
                  gsl::not_null<Cache*> cache,
                  Tags::Xi<DataType, Dim> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::ia<DataType, Dim, Frame::Inertial>*> xi_gradient,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::Xi<DataType, Dim>, tmpl::size_t<Dim>,
                    Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<typename ::Tags::Flux<
          Tags::Xi<DataType, Dim>, tmpl::size_t<Dim>, Frame::Inertial>::type*>
          flux_for_xi,
      gsl::not_null<Cache*> cache,
      ::Tags::Flux<Tags::Xi<DataType, Dim>, tmpl::size_t<Dim>,
                   Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::a<DataType, Dim, Frame::Inertial>*>
                      fixed_source_for_xi,
                  gsl::not_null<Cache*> cache,
                  ::Tags::FixedSource<Tags::Xi<DataType, Dim>> /*meta*/) const;
};
}  // namespace detail

/*!
 * \brief An analytic solution for the damped-harmonic coordinate correction
 * \f$\xi_a\f$.
 *
 * \details This implements the analytic solution
 * \f$\xi_t = 2m \log(2m/r) + C\f$ and \f$\xi_i = -m x_i / r\f$ in 3D.
 */
template <size_t Dim, typename DataType = DataVector>
class Lorentzian : public elliptic::analytic_data::AnalyticSolution {
  static_assert(
      Dim == 3,
      "This solution is currently implemented in 3 spatial dimensions only");

 public:
  struct PlusConstant {
    using type = double;
    static constexpr Options::String help{"Constant added to the solution."};
  };
  struct Mass {
    using type = double;
    static constexpr Options::String help{
        "Mass parameter m in the DQ solution."};
  };

  struct ComplexPhase {
    using type = double;
    static constexpr Options::String help{
        "Phase 'phi' of a complex exponential 'exp(i phi)' that rotates the "
        "solution in the complex plane."};
  };

  using options = tmpl::flatten<tmpl::list<
      Mass, PlusConstant,
      tmpl::conditional_t<std::is_same_v<DataType, ComplexDataVector>,
                          ComplexPhase, tmpl::list<>>>>;
  static constexpr Options::String help{
      "A Lorentzian solution to the Poisson equation."};

  Lorentzian() = default;
  Lorentzian(const Lorentzian&) = default;
  Lorentzian& operator=(const Lorentzian&) = default;
  Lorentzian(Lorentzian&&) = default;
  Lorentzian& operator=(Lorentzian&&) = default;
  ~Lorentzian() override = default;

  explicit Lorentzian(const double mass, const double constant,
                      const double complex_phase = 0.)
      : mass_(mass), constant_(constant), complex_phase_(complex_phase) {
    ASSERT((std::is_same_v<DataType, ComplexDataVector> or complex_phase == 0.),
           "The complex phase is only supported for ComplexDataVector.");
  }

  double mass() const { return mass_; }
  double constant() const { return constant_; }
  double complex_phase() const { return complex_phase_; }

  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<Lorentzian>(*this);
  }

  /// \cond
  explicit Lorentzian(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Lorentzian);  // NOLINT
  /// \endcond

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::LorentzianVariables<DataType, Dim>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{x, mass_, constant_, complex_phase_};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    p | mass_;
    p | constant_;
    p | complex_phase_;
  }

 private:
  double mass_ = std::numeric_limits<double>::signaling_NaN();
  double constant_ = std::numeric_limits<double>::signaling_NaN();
  double complex_phase_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <size_t Dim, typename DataType>
PUP::able::PUP_ID Lorentzian<Dim, DataType>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <size_t Dim, typename DataType>
bool operator==(const Lorentzian<Dim, DataType>& lhs,
                const Lorentzian<Dim, DataType>& rhs) {
  return lhs.mass() == rhs.mass() and lhs.constant() == rhs.constant() and
         lhs.complex_phase() == rhs.complex_phase();
}

template <size_t Dim, typename DataType>
bool operator!=(const Lorentzian<Dim, DataType>& lhs,
                const Lorentzian<Dim, DataType>& rhs) {
  return not(lhs == rhs);
}

}  // namespace DQ::Solutions
