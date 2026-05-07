# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DQBoundaryConditions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  HorizonRobin.cpp
  Robin.cpp
  SingleBhOuterBoundary.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  HorizonRobin.hpp
  Robin.hpp
  SingleBhOuterBoundary.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Elliptic
  GeneralizedHarmonic
  GeneralRelativity
  InitialDataUtilities
  Options
  Parallel
  SphericalHarmonics
  SphericalHarmonicsIO
  Utilities
  )
