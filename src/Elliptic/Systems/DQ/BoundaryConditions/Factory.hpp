// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/Systems/DQ/BoundaryConditions/HorizonRobin.hpp"
#include "Elliptic/Systems/DQ/BoundaryConditions/Robin.hpp"
#include "Elliptic/Systems/DQ/BoundaryConditions/SingleBhOuterBoundary.hpp"
#include "Utilities/TMPL.hpp"

namespace DQ::BoundaryConditions {

template <typename System>
using standard_boundary_conditions =
    tmpl::list<elliptic::BoundaryConditions::AnalyticSolution<System>,
               Robin<System::volume_dim>, HorizonRobin<System::volume_dim>,
               SingleBhOuterBoundary<System::volume_dim>>;
}
