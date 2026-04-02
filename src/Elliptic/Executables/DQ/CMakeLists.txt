# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Charmxx::main
  AmrCriteria
  Convergence
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Elliptic
  EllipticDg
  EllipticDgSubdomainOperator
  EllipticSubdomainPreconditioners
  Events
  EventsAndTriggers
  FunctionsOfTime
  Informer
  InitialDataUtilities
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelAmr
  ParallelLinearSolver
  ParallelMultigrid
  ParallelSchwarz
  PhaseControl
  DQ
  DQBoundaryConditions
  DQSolutions
  Utilities
  )

function(add_DQ_executable DIM)
set(EXECUTABLE "DQ${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    SolveDQ.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction(add_DQ_executable)

add_DQ_executable(3)
