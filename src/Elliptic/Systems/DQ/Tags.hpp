// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the Poisson system

#pragma once

#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving a Poisson equation \f$-\Delta u(x)=f(x)\f$.
 */
namespace DQ {

namespace OptionTags {

/*!
 * \brief The parameter \f$m\f$ in the DQ operator.
 *
 * Parsed from the input file as `Mass: <double>`.
 */
struct Mass {
  using type = double;
  static constexpr Options::String help =
      "Parameter m used in the DQ elliptic operator.";
};

}  // namespace OptionTags

namespace Tags {

/*!
 * \brief The scalar field \f$u(x)\f$ to solve for
 */
template <typename DataType>
struct Field : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The raw continuum source \f$f(x) = -2m/r^2\f$ used in the DQ
 * equation.
 *
 * This tag is used for observation so volume output can show the physical
 * source rather than the linear-solver RHS after boundary-condition
 * contributions have been folded in.
 */
template <typename DataType>
struct ObservedSource : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() { return "PhysicalSource(Field)"; }
};

template <typename DataType, size_t Dim>
struct ObservedSourceCompute : ObservedSource<DataType>, db::ComputeTag {
  using base = ObservedSource<DataType>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  static std::string name() { return base::name(); }

  static void function(
      const gsl::not_null<return_type*> observed_source,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
    const DataVector r2 = get(dot_product(inertial_coords, inertial_coords));
    get(*observed_source) = -2.0 / r2;
  }
};

struct Mass : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::Mass>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& mass) { return mass; }
};

}  // namespace Tags
}  // namespace Poisson
