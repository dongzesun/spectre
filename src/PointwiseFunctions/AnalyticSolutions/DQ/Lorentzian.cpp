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
#include "Elliptic/Systems/DQ/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace DQ::Solutions::detail {

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::a<DataType, Dim, Frame::Inertial>*> xi,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::Xi<DataType, Dim> /*meta*/) const {
  const DataVector r2 = get(dot_product(x, x));
  const DataVector r = sqrt(r2);
  xi->get(0) = make_with_value<DataType>(r, 2.0 * mass) *
                   log(make_with_value<DataType>(r, 2.0 * mass) / r) +
               make_with_value<DataType>(r, constant);
  for (size_t d = 0; d < Dim; ++d) {
    xi->get(d + 1) = -make_with_value<DataType>(r, mass) * x.get(d) / r;
  }
  if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
    const std::complex<double> phase{cos(complex_phase), sin(complex_phase)};
    for (size_t a = 0; a < Dim + 1; ++a) {
      xi->get(a) *= phase;
    }
  }
}

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::ia<DataType, Dim, Frame::Inertial>*> xi_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::Xi<DataType, Dim>, tmpl::size_t<Dim>,
                  Frame::Inertial> /*meta*/) const {
  const DataVector r2 = get(dot_product(x, x));
  const DataVector r = sqrt(r2);
  DataType time_prefactor = -make_with_value<DataType>(r2, 2.0 * mass) / r2;
  if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
    time_prefactor *=
        std::complex<double>{cos(complex_phase), sin(complex_phase)};
  }
  for (size_t d = 0; d < Dim; ++d) {
    xi_gradient->get(d, 0) = time_prefactor * x.get(d);
  }

  for (size_t s = 0; s < Dim; ++s) {
    for (size_t d = 0; d < Dim; ++d) {
      xi_gradient->get(d, s + 1) =
          make_with_value<DataType>(r, 0.0) +
          make_with_value<DataType>(r, mass) * x.get(s) * x.get(d) / (r2 * r);
      if (d == s) {
        xi_gradient->get(d, s + 1) -= make_with_value<DataType>(r, mass) / r;
      }
      if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
        xi_gradient->get(d, s + 1) *=
            std::complex<double>{cos(complex_phase), sin(complex_phase)};
      }
    }
  }
}

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<DQ::FluxXiTensor<DataType, Dim>*> flux_for_xi,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::Xi<DataType, Dim>, tmpl::size_t<Dim>,
                 Frame::Inertial> /*meta*/) const {
  const auto& xi_gradient = cache->get_var(
      *this, ::Tags::deriv<Tags::Xi<DataType, Dim>, tmpl::size_t<Dim>,
                           Frame::Inertial>{});
  dq_fluxes_flat_cartesian<DataType, Dim>(flux_for_xi, mass, x, xi_gradient);
}

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::a<DataType, Dim, Frame::Inertial>*>
        fixed_source_for_xi,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::Xi<DataType, Dim>> /*meta*/) const {
  const DataVector r2 = get(dot_product(x, x));
  const DataVector r = sqrt(r2);
  fixed_source_for_xi->get(0) = make_with_value<DataType>(r2, 2.0 * mass) / r2;
  for (size_t d = 0; d < Dim; ++d) {
    fixed_source_for_xi->get(d + 1) =
        -make_with_value<DataType>(r2, 2.0 * mass) * x.get(d) / (r2 * r);
  }
  if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
    const std::complex<double> phase{cos(complex_phase), sin(complex_phase)};
    for (size_t a = 0; a < Dim + 1; ++a) {
      fixed_source_for_xi->get(a) *= phase;
    }
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

}  // namespace DQ::Solutions::detail
