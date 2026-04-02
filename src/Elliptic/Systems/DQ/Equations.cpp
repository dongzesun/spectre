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
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
    const tnsr::i<DataType, Dim, Frame::Inertial>& field_gradient) {

  // F^i = (delta^{ij} - 2m/r n^i n^j) d_j u
  //     = d^i u - 2m * x^i (x^j d_j u) / r^3

  // r^2 and r^{-3} are real (DataVector)
  DataVector r2 = x.get(0) * x.get(0);
  for (size_t d = 1; d < Dim; ++d) {
    r2 += x.get(d) * x.get(d);
  }
  const DataVector r = sqrt(r2);
  const DataVector inv_r3 = 1.0 / (r2 * r);  // = 1/r^3

  // x · grad(u) has type DataType (DataVector or ComplexDataVector)
  DataType x_dot_grad_u = x.get(0) * field_gradient.get(0);
  for (size_t d = 1; d < Dim; ++d) {
    x_dot_grad_u += x.get(d) * field_gradient.get(d);
  }

  for (size_t d = 0; d < Dim; ++d) {
    flux_for_field->get(d) =
        field_gradient.get(d)
        - (2.0 * 1.0) * (x.get(d) * inv_r3) * x_dot_grad_u;
  }
}

template <typename DataType, size_t Dim>
void curved_fluxes(const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                   const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                   const tnsr::i<DataType, Dim>& field_gradient) {
  raise_or_lower_index(flux_for_field, field_gradient, inv_spatial_metric);
}

template <typename DataType, size_t Dim>
void fluxes_on_face(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                    const tnsr::I<DataVector, Dim>& face_normal_vector,
                    const Scalar<DataType>& field) {
  std::copy(face_normal_vector.begin(), face_normal_vector.end(),
            flux_for_field->begin());
  for (size_t d = 0; d < Dim; d++) {
    flux_for_field->get(d) *= get(field);
  }
}

template <typename DataType, size_t Dim>
void add_curved_sources(const gsl::not_null<Scalar<DataType>*> source_for_field,
                        const tnsr::i<DataVector, Dim>& christoffel_contracted,
                        const tnsr::I<DataType, Dim>& flux_for_field) {
  get(*source_for_field) -=
      get(dot_product(christoffel_contracted, flux_for_field));
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::FlatCartesian, DataType>::apply(
    const gsl::not_null<tnsr::I<DataType, Dim,
                  Frame::Inertial>*> flux_for_field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
    const Scalar<DataType>& /*field*/,
    const tnsr::i<DataType, Dim, Frame::Inertial>& field_gradient) {
  dq_fluxes_flat_cartesian(flux_for_field, inertial_coords, field_gradient);
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::FlatCartesian, DataType>::apply(
    const gsl::not_null<tnsr::I<DataType, Dim,
                      Frame::Inertial>*> flux_for_field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
                      /*inertial_coords_on_face*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*face_normal*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& face_normal_vector,
    const Scalar<DataType>& field) {
  fluxes_on_face(flux_for_field, face_normal_vector, field);
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::Curved, DataType>::apply(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
    const Scalar<DataType>& /*field*/,
    const tnsr::i<DataType, Dim>& field_gradient) {
  curved_fluxes(flux_for_field, inv_spatial_metric, field_gradient);
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::Curved, DataType>::apply(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const tnsr::II<DataVector, Dim>& /*inv_spatial_metric*/,
    const tnsr::i<DataVector, Dim>& /*face_normal*/,
    const tnsr::I<DataVector, Dim>& face_normal_vector,
    const Scalar<DataType>& field) {
  fluxes_on_face(flux_for_field, face_normal_vector, field);
}

template <size_t Dim, typename DataType>
void Sources<Dim, Geometry::Curved, DataType>::apply(
    const gsl::not_null<Scalar<DataType>*> equation_for_field,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const Scalar<DataType>& /*field*/,
    const tnsr::I<DataType, Dim>& field_flux) {
  add_curved_sources(equation_for_field, christoffel_contracted, field_flux);
}

}  // namespace DQ

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                  \
  template void DQ::dq_fluxes_flat_cartesian(                               \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data)>*>,                  \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DTYPE(data), DIM(data), Frame::Inertial>&);               \
  template void DQ::curved_fluxes(                                       \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data)>*>,                  \
      const tnsr::II<DataVector, DIM(data)>&,                                 \
      const tnsr::i<DTYPE(data), DIM(data)>&);                                \
  template void DQ::fluxes_on_face(                                      \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data)>*>,                  \
      const tnsr::I<DataVector, DIM(data)>&, const Scalar<DTYPE(data)>&);     \
  template void DQ::add_curved_sources(                                  \
      const gsl::not_null<Scalar<DTYPE(data)>*>,                              \
      const tnsr::i<DataVector, DIM(data)>&,                                  \
      const tnsr::I<DTYPE(data), DIM(data)>&);                                \
  template class DQ::Fluxes<DIM(data), DQ::Geometry::FlatCartesian, \
                                 DTYPE(data)>;                                \
  template class DQ::Fluxes<DIM(data), DQ::Geometry::Curved,        \
                                 DTYPE(data)>;                                \
  template class DQ::Sources<DIM(data), DQ::Geometry::Curved,       \
                                  DTYPE(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, ComplexDataVector), (1, 2, 3))

#undef INSTANTIATE
#undef DIM
#undef DTYPE
