// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/DQ/Equations.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace DQ {

template <typename DataType, size_t Dim>
void dq_fluxes_flat_cartesian(
    const gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
    const double mass, const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
    const DerivXiTensor<DataType, Dim>& xi_gradient) {
  // F^i = (delta^{ij} - 2m/r n^i n^j) d_j u
  //     = d^i u - 2m * x^i (x^j d_j u) / r^3

  // r^2 and r^{-3} are real (DataVector)
  DataVector r2 = x.get(0) * x.get(0);
  for (size_t d = 1; d < Dim; ++d) {
    r2 += x.get(d) * x.get(d);
  }
  const DataVector r = sqrt(r2);
  const DataVector inv_r3 = 1.0 / (r2 * r);  // = 1/r^3

  for (size_t a = 0; a < Dim + 1; ++a) {
    DataType x_dot_grad_xi_a = x.get(0) * xi_gradient.get(0, a);
    for (size_t d = 1; d < Dim; ++d) {
      x_dot_grad_xi_a += x.get(d) * xi_gradient.get(d, a);
    }
    for (size_t d = 0; d < Dim; ++d) {
      flux_for_xi->get(d, a) = xi_gradient.get(d, a) - 2.0 * mass *
                                                           (x.get(d) * inv_r3) *
                                                           x_dot_grad_xi_a;
    }
  }
}

template <typename DataType, size_t Dim>
void curved_fluxes(
    const gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
    const DerivXiTensor<DataType, Dim>& xi_gradient) {
  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t i = 0; i < Dim; ++i) {
      flux_for_xi->get(i, a) =
          xi_gradient.get(0, a) * inv_spatial_metric.get(i, 0);
      for (size_t j = 1; j < Dim; ++j) {
        flux_for_xi->get(i, a) +=
            xi_gradient.get(j, a) * inv_spatial_metric.get(i, j);
      }
    }
  }
}

template <typename DataType, size_t Dim>
void fluxes_on_face(gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
                    const tnsr::I<DataVector, Dim>& face_normal_vector,
                    const XiTensor<DataType, Dim>& xi) {
  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t d = 0; d < Dim; d++) {
      flux_for_xi->get(d, a) = face_normal_vector.get(d) * xi.get(a);
    }
  }
}

template <typename DataType, size_t Dim>
void add_curved_sources(
    const gsl::not_null<XiTensor<DataType, Dim>*> source_for_xi,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const FluxXiTensor<DataType, Dim>& flux_for_xi) {
  for (size_t a = 0; a < Dim + 1; ++a) {
    source_for_xi->get(a) -=
        christoffel_contracted.get(0) * flux_for_xi.get(0, a);
    for (size_t i = 1; i < Dim; ++i) {
      source_for_xi->get(a) -=
          christoffel_contracted.get(i) * flux_for_xi.get(i, a);
    }
  }
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::FlatCartesian, DataType>::apply(
    const gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
    const double& mass, const XiTensor<DataType, Dim>& /*xi*/,
    const DerivXiTensor<DataType, Dim>& xi_gradient) {
  dq_fluxes_flat_cartesian<DataType, Dim>(flux_for_xi, mass, inertial_coords,
                                          xi_gradient);
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::FlatCartesian, DataType>::apply(
    const gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
    /*inertial_coords_on_face*/,
    const double& /*mass*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*face_normal*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_normal_vector,
    const XiTensor<DataType, Dim>& xi) {
  fluxes_on_face<DataType, Dim>(flux_for_field, face_normal_vector, xi);
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::Curved, DataType>::apply(
    const gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
    const XiTensor<DataType, Dim>& /*xi*/,
    const DerivXiTensor<DataType, Dim>& xi_gradient) {
  curved_fluxes<DataType, Dim>(flux_for_xi, inv_spatial_metric, xi_gradient);
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::Curved, DataType>::apply(
    const gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
    const tnsr::II<DataVector, Dim>& /*inv_spatial_metric*/,
    const tnsr::i<DataVector, Dim>& /*face_normal*/,
    const tnsr::I<DataVector, Dim>& face_normal_vector,
    const XiTensor<DataType, Dim>& xi) {
  fluxes_on_face<DataType, Dim>(flux_for_xi, face_normal_vector, xi);
}

template <size_t Dim, typename DataType>
void Sources<Dim, Geometry::Curved, DataType>::apply(
    const gsl::not_null<XiTensor<DataType, Dim>*> equation_for_xi,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const XiTensor<DataType, Dim>& /*xi*/,
    const FluxXiTensor<DataType, Dim>& xi_flux) {
  add_curved_sources<DataType, Dim>(equation_for_xi, christoffel_contracted,
                                    xi_flux);
}

}  // namespace DQ

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template void DQ::dq_fluxes_flat_cartesian<DTYPE(data), DIM(data)>(       \
      const gsl::not_null<DQ::FluxXiTensor<DTYPE(data), DIM(data)>*>,       \
      const double, const tnsr::I<DataVector, DIM(data), Frame::Inertial>&, \
      const DQ::DerivXiTensor<DTYPE(data), DIM(data)>&);                    \
  template void DQ::curved_fluxes<DTYPE(data), DIM(data)>(                  \
      const gsl::not_null<DQ::FluxXiTensor<DTYPE(data), DIM(data)>*>,       \
      const tnsr::II<DataVector, DIM(data)>&,                               \
      const DQ::DerivXiTensor<DTYPE(data), DIM(data)>&);                    \
  template void DQ::fluxes_on_face<DTYPE(data), DIM(data)>(                 \
      const gsl::not_null<DQ::FluxXiTensor<DTYPE(data), DIM(data)>*>,       \
      const tnsr::I<DataVector, DIM(data)>&,                                \
      const DQ::XiTensor<DTYPE(data), DIM(data)>&);                         \
  template void DQ::add_curved_sources<DTYPE(data), DIM(data)>(             \
      const gsl::not_null<DQ::XiTensor<DTYPE(data), DIM(data)>*>,           \
      const tnsr::i<DataVector, DIM(data)>&,                                \
      const DQ::FluxXiTensor<DTYPE(data), DIM(data)>&);                     \
  template class DQ::Fluxes<DIM(data), DQ::Geometry::FlatCartesian,         \
                            DTYPE(data)>;                                   \
  template class DQ::Fluxes<DIM(data), DQ::Geometry::Curved, DTYPE(data)>;  \
  template class DQ::Sources<DIM(data), DQ::Geometry::Curved, DTYPE(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, ComplexDataVector), (1, 2, 3))

#undef INSTANTIATE
#undef DIM
#undef DTYPE
