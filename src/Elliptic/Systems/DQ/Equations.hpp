// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/DQ/Geometry.hpp"
#include "Elliptic/Systems/DQ/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace DQ {
template <typename DataType, size_t Dim>
using XiTensor = typename Tags::Xi<DataType, Dim>::type;
template <typename DataType, size_t Dim>
using DerivXiTensor =
    typename ::Tags::deriv<Tags::Xi<DataType, Dim>, tmpl::size_t<Dim>,
                           Frame::Inertial>::type;
template <typename DataType, size_t Dim>
using FluxXiTensor =
    typename ::Tags::Flux<Tags::Xi<DataType, Dim>, tmpl::size_t<Dim>,
                          Frame::Inertial>::type;
template <size_t Dim, Geometry BackgroundGeometry,
          typename DataType = DataVector>
struct Fluxes;
template <size_t Dim, Geometry BackgroundGeometry,
          typename DataType = DataVector>
struct Sources;
template <size_t Dim, typename DataType = DataVector>
struct FlatCartesianSources;
}  // namespace Poisson
/// \endcond

namespace DQ {

/*!
 * \brief Compute the fluxes \f$F^i=\partial_i u(x)\f$ for the Poisson
 * equation on a flat spatial metric in Cartesian coordinates.
 */

template <typename DataType, size_t Dim>
void dq_fluxes_flat_cartesian(
    gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi, const double mass,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coordinates,
    const DerivXiTensor<DataType, Dim>& xi_gradient);

/*!
 * \brief Compute the fluxes \f$F^i=\gamma^{ij}\partial_j u(x)\f$
 * for the curved-space Poisson equation on a spatial metric \f$\gamma_{ij}\f$.
 */
template <typename DataType, size_t Dim>
void curved_fluxes(gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
                   const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                   const DerivXiTensor<DataType, Dim>& xi_gradient);

/*!
 * \brief Compute the fluxes $F^i=\gamma^{ij} n_j u$ where $n_j$ is the
 * `face_normal`.
 *
 * The `face_normal_vector` is $\gamma^{ij} n_j$.
 */
template <typename DataType, size_t Dim>
void fluxes_on_face(gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
                    const tnsr::I<DataVector, Dim>& face_normal_vector,
                    const XiTensor<DataType, Dim>& xi);

/*!
 * \brief Add the sources \f$S=-\Gamma^i_{ij}v^j\f$
 * for the curved-space Poisson equation on a spatial metric \f$\gamma_{ij}\f$.
 *
 * These sources arise from the non-principal part of the Laplacian on a
 * non-Euclidean background.
 */
template <typename DataType, size_t Dim>
void add_curved_sources(gsl::not_null<XiTensor<DataType, Dim>*> source_for_xi,
                        const tnsr::i<DataVector, Dim>& christoffel_contracted,
                        const FluxXiTensor<DataType, Dim>& flux_for_xi);

/*!
 * \brief Compute the fluxes \f$F^i\f$ for the Poisson equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim, typename DataType>
struct Fluxes<Dim, Geometry::FlatCartesian, DataType> {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 DQ::Tags::Mass>;
  using volume_tags = tmpl::list<DQ::Tags::Mass>;
  using const_global_cache_tags = tmpl::list<DQ::Tags::Mass>;
  static constexpr bool is_trivial = true;
  static constexpr bool is_discontinuous = false;
  static void apply(
      gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
      const double& mass, const XiTensor<DataType, Dim>& xi,
      const DerivXiTensor<DataType, Dim>& xi_gradient);
  static void apply(
      gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords_on_face,
      const double& mass,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& face_normal_vector,
      const XiTensor<DataType, Dim>& xi);
};

/*!
 * \brief Compute the fluxes \f$F^i\f$ for the curved-space Poisson equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim, typename DataType>
struct Fluxes<Dim, Geometry::Curved, DataType> {
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, Dim>>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = true;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
                    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                    const XiTensor<DataType, Dim>& xi,
                    const DerivXiTensor<DataType, Dim>& xi_gradient);
  static void apply(gsl::not_null<FluxXiTensor<DataType, Dim>*> flux_for_xi,
                    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                    const tnsr::i<DataVector, Dim>& face_normal,
                    const tnsr::I<DataVector, Dim>& face_normal_vector,
                    const XiTensor<DataType, Dim>& xi);
};

/*!
 * \brief Add the sources \f$S\f$ for the curved-space Poisson equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim, typename DataType>
struct Sources<Dim, Geometry::Curved, DataType> {
  using argument_tags = tmpl::list<
      gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, Dim>>;
  using const_global_cache_tags = tmpl::list<>;
  static void apply(gsl::not_null<XiTensor<DataType, Dim>*> equation_for_xi,
                    const tnsr::i<DataVector, Dim>& christoffel_contracted,
                    const XiTensor<DataType, Dim>& xi,
                    const FluxXiTensor<DataType, Dim>& xi_flux);
};

template <size_t Dim, typename DataType>
struct FlatCartesianSources {
  using argument_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;

  template <typename... Args>
  static void apply(gsl::not_null<XiTensor<DataType, Dim>*> equation_for_xi,
                    const Args&... /*unused*/) {
    for (size_t a = 0; a < Dim + 1; ++a) {
      equation_for_xi->get(a) = 0.;
    }
  }
};

}  // namespace DQ
