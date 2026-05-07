// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the Poisson system

#pragma once

#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
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
 * \brief The coordinate correction \f$\xi_a(x)\f$ to solve for
 */
template <typename DataType, size_t Dim>
struct Xi : db::SimpleTag {
  using type = tnsr::a<DataType, Dim, Frame::Inertial>;
};

template <typename DataType>
struct Field : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The observed source for the \f$\xi_a\f$ equation.
 *
 * This tag is populated from the background so volume output can show the
 * source being supplied to the solve for all four spacetime components.
 */
template <typename DataType, size_t Dim>
struct ObservedSource : db::SimpleTag {
  using type = tnsr::a<DataType, Dim, Frame::Inertial>;
  static std::string name() { return "PhysicalSource(Xi)"; }
};

struct Mass : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::Mass>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& mass) { return mass; }
};

}  // namespace Tags
}  // namespace DQ
