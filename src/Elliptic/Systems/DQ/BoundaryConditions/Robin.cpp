// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/DQ/BoundaryConditions/Robin.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace DQ::BoundaryConditions {

template <size_t Dim>
Robin<Dim>::Robin(const double dirichlet_weight, const double neumann_weight,
                  const double constant, const Options::Context& context)
    : dirichlet_weight_(dirichlet_weight),
      neumann_weight_(neumann_weight),
      constant_(constant) {
  if (dirichlet_weight == 0. and neumann_weight == 0.) {
    PARSE_ERROR(
        context,
        "Either the dirichlet_weight or the neumann_weight must be non-zero "
        "(or both).");
  }
}

template <size_t Dim>
void Robin<Dim>::apply(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> xi,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        n_dot_deriv_xi,
    const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi*/) const {
  if (neumann_weight_ == 0.) {
    ASSERT(
        not equal_within_roundoff(dirichlet_weight_, 0.),
        "The dirichlet_weight is close to zero. Set it to a non-zero value to "
        "avoid divisions by small numbers.");
    for (size_t a = 0; a < Dim + 1; ++a) {
      xi->get(a) = constant_ / dirichlet_weight_;
    }
  } else {
    ASSERT(not equal_within_roundoff(neumann_weight_, 0.),
           "The neumann_weight is close to zero. Set it to a non-zero value to "
           "avoid divisions by small numbers.");
    for (size_t a = 0; a < Dim + 1; ++a) {
      n_dot_deriv_xi->get(a) =
          (constant_ - dirichlet_weight_ * xi->get(a)) / neumann_weight_;
    }
  }
}

template <size_t Dim>
void Robin<Dim>::apply_linearized(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        xi_correction,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        n_dot_deriv_xi_correction,
    const tnsr::ia<DataVector, Dim, Frame::Inertial>& /*deriv_xi_correction*/)
    const {
  if (neumann_weight_ == 0.) {
    for (size_t a = 0; a < Dim + 1; ++a) {
      xi_correction->get(a) = 0.;
    }
  } else {
    ASSERT(not equal_within_roundoff(neumann_weight_, 0.),
           "The neumann_weight is close to zero. Set it to a non-zero value to "
           "avoid divisions by small numbers.");
    for (size_t a = 0; a < Dim + 1; ++a) {
      n_dot_deriv_xi_correction->get(a) =
          -dirichlet_weight_ / neumann_weight_ * xi_correction->get(a);
    }
  }
}

template <size_t Dim>
void Robin<Dim>::pup(PUP::er& p) {
  p | dirichlet_weight_;
  p | neumann_weight_;
  p | constant_;
}

template <size_t Dim>
bool operator==(const Robin<Dim>& lhs, const Robin<Dim>& rhs) {
  return lhs.dirichlet_weight() == rhs.dirichlet_weight() and
         lhs.neumann_weight() == rhs.neumann_weight() and
         lhs.constant() == rhs.constant();
}

template <size_t Dim>
bool operator!=(const Robin<Dim>& lhs, const Robin<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
PUP::able::PUP_ID Robin<Dim>::my_PUP_ID = 0;  // NOLINT

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template class Robin<DIM(data)>;                                            \
  template bool operator==(const Robin<DIM(data)>&, const Robin<DIM(data)>&); \
  template bool operator!=(const Robin<DIM(data)>&, const Robin<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace Poisson::BoundaryConditions
