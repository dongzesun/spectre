// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/DQ/Lorentzian.hpp"

#include <array>
#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace DQ::Solutions::detail {

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> field,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::Field<DataType> /*meta*/) const {

  constexpr double m = 1.0;  // hard-coded mass for now

  const DataVector r2 = get(dot_product(x, x));
  const DataVector r = sqrt(r2);

  // u = 2 m log(r) + constant  (here m = 1)
  get(*field) = make_with_value<DataType>(r, 0.0)
                - make_with_value<DataType>(r, 2.0 * m)
                * log(make_with_value<DataType>(r, 2.0 * m)/r)
                + make_with_value<DataType>(r, constant);

  if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
    get(*field) *= std::complex<double>{cos(complex_phase), sin(complex_phase)};
  }
}

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*> field_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::Field<DataType>, tmpl::size_t<Dim>,
                  Frame::Inertial> /*meta*/) const {

  constexpr double m = 1.0;  // hard-coded mass for now

  const DataVector r2 = get(dot_product(x, x));
  const DataVector r = sqrt(r2);

  DataType prefactor = make_with_value<DataType>(r2, (2.0 * m)) / r2;

  if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
    prefactor *= std::complex<double>{cos(complex_phase), sin(complex_phase)};
  }

  for (size_t d = 0; d < Dim; ++d) {
    field_gradient->get(d) = prefactor * x.get(d);
  }
}

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::Field<DataType>, tmpl::size_t<Dim>,
                 Frame::Inertial> /*meta*/) const {

  constexpr double m = 1.0;  // hard-coded mass for now

  const auto& grad_u = cache->get_var(
      *this, ::Tags::deriv<Tags::Field<DataType>, tmpl::size_t<Dim>,
                           Frame::Inertial>{});

  const DataVector r2 = get(dot_product(x, x));
  const DataVector r = sqrt(r2);
  const DataVector inv_r2 = 1.0 / r2;
  const DataVector inv_r = 1.0 / r;

  // n·grad u = (x·grad u)/r
  const DataType x_dot_grad_u = get(dot_product(x, grad_u));
  const DataType n_dot_grad_u = x_dot_grad_u * inv_r;

  // flux F^i = grad u^i - (2m/r) n^i (n·grad u)
  // (2m/r) n^i = 2m x^i / r^2  (here m = 1)
  const DataVector coeff = (2.0 * m) * inv_r2;

  for (size_t d = 0; d < Dim; ++d) {
    flux_for_field->get(d) =
        grad_u.get(d) - (coeff * x.get(d)) * n_dot_grad_u;
  }
}

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> fixed_source_for_field,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::Field<DataType>> /*meta*/) const {

  constexpr double m = 1.0;  // hard-coded mass for now

  const DataVector r2 = get(dot_product(x, x));

  // RHS f = -2m/r^2  (here m = 1)
  get(*fixed_source_for_field) = -make_with_value<DataType>(r2, 2.0 * m) / r2;

  if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
    get(*fixed_source_for_field) *=
        std::complex<double>{cos(complex_phase), sin(complex_phase)};
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class LorentzianVariables<DTYPE(data), DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, ComplexDataVector), (3))

#undef DTYPE
#undef DIM
#undef INSTANTIATE

}  // namespace Poisson::Solutions::detail
