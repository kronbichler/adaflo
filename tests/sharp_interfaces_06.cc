// --------------------------------------------------------------------------
//
// Copyright (C) 2021 by the adaflo authors
//
// This file is part of the adaflo library.
//
// The adaflo library is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.  The full text of the
// license can be found in the file LICENSE at the top level of the adaflo
// distribution.
//
// --------------------------------------------------------------------------

// runs a simulation on a static bubble where the velocities ideally should be
// zero but where we actually get some velocities which are due to
// inaccuracies in the scheme

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <adaflo/level_set_okz.h>
#include <adaflo/level_set_okz_matrix.h>
#include <adaflo/parameters.h>
#include <adaflo/phase_field.h>

#include "sharp_interfaces_util.h"


using namespace dealii;

struct TwoPhaseParameters : public FlowParameters
{
  TwoPhaseParameters(const std::string &parameter_filename)
  {
    ParameterHandler prm;
    FlowParameters::declare_parameters(prm);
    prm.enter_subsection("Problem-specific");
    prm.declare_entry(
      "two-phase method",
      "front tracking",
      Patterns::Selection(
        "front tracking|mixed level set|sharp level set|level set|level set 2"),
      "Defines the two-phase method to be used");
    prm.leave_subsection();
    check_for_file(parameter_filename, prm);
    parse_parameters(parameter_filename, prm);
    prm.enter_subsection("Problem-specific");
    solver_method = prm.get("two-phase method");
    prm.leave_subsection();
  }

  std::string solver_method;
};

template <int dim>
void
create_annulus(Triangulation<dim> &tria, const unsigned int n_refinements)
{
  GridGenerator::hyper_cube(tria, -2.5, +2.5);

  if (n_refinements == 0)
    return;

  for (int i = 0; i < static_cast<int>(n_refinements) - 3; i++)
    tria.refine_global();

  Point<dim> center;
  for (unsigned int d = 0; d < dim; ++d)
    center[d] = 0.02 + 0.01 * d;

  if (n_refinements >= 1)
    {
      for (auto cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          if ((cell->center() - center).norm() < 0.5)
            cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }

  if (n_refinements >= 2)
    {
      for (auto cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          if ((cell->center() - center).norm() < 0.5)
            cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }

  if (n_refinements >= 3)
    {
      for (auto cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          if ((cell->center() - center).norm() < 0.5)
            cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }
}



template <int dim>
class InitialValuesLS : public Function<dim>
{
public:
  InitialValuesLS()
    : Function<dim>(1, 0)
  {}

  double
  value(const Point<dim> &p, const unsigned int) const
  {
    // set radius of bubble to 0.5, slightly shifted away from the center
    Point<dim> center;
    for (unsigned int d = 0; d < dim; ++d)
      center[d] = 0.02 + 0.01 * d;
    return p.distance(center) - 0.5;
  }
};



template <int dim>
class MicroFluidicProblem
{
public:
  MicroFluidicProblem(const TwoPhaseParameters &parameters);
  void
  run();

private:
  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;

  mutable TimerOutput timer;

  TwoPhaseParameters                        parameters;
  parallel::distributed::Triangulation<dim> triangulation;
  parallel::distributed::Triangulation<dim> triangulation_ls;
};

template <int dim>
MicroFluidicProblem<dim>::MicroFluidicProblem(const TwoPhaseParameters &parameters)
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , timer(pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  , parameters(parameters)
  , triangulation(mpi_communicator)
  , triangulation_ls(mpi_communicator)
{}

template <int dim>
void
MicroFluidicProblem<dim>::run()
{
  GridGenerator::hyper_cube(triangulation, -2.5, 2.5);

  if (false)
    {
      GridGenerator::hyper_cube(triangulation_ls, -2.5, 2.5);
    }
  else
    {
      create_annulus(triangulation_ls, parameters.global_refinements);
    }

  NavierStokes<dim> navier_stokes_solver(parameters, triangulation, &timer);

  navier_stokes_solver.set_no_slip_boundary(0);
  navier_stokes_solver.fix_pressure_constant(0);
  navier_stokes_solver.setup_problem(Functions::ZeroFunction<dim>(dim));
  navier_stokes_solver.output_solution(parameters.output_filename);

  Triangulation<dim - 1, dim> surface_mesh;
  GridGenerator::hyper_sphere(surface_mesh, Point<dim>(0.02, 0.03), 0.5);
  surface_mesh.refine_global(5);

  std::unique_ptr<SharpInterfaceSolver> solver;

  if (parameters.solver_method == "front tracking")
    solver =
      std::make_unique<FrontTrackingSolver<dim>>(navier_stokes_solver, surface_mesh);
  else if (parameters.solver_method == "mixed level set")
    solver = std::make_unique<MixedLevelSetSolver<dim>>(navier_stokes_solver,
                                                        surface_mesh,
                                                        InitialValuesLS<dim>());
  else if (parameters.solver_method == "sharp level set")
    solver = std::make_unique<MixedLevelSetSolver<dim>>(navier_stokes_solver,
                                                        InitialValuesLS<dim>());
  else if (parameters.solver_method == "level set")
    solver = std::make_unique<MixedLevelSetSolver<dim>>(navier_stokes_solver,
                                                        InitialValuesLS<dim>(),
                                                        false);
  else if (parameters.solver_method == "level set 2")
    solver = std::make_unique<MixedLevelSetSolver<dim>>(navier_stokes_solver,
                                                        triangulation_ls,
                                                        InitialValuesLS<dim>());
  else
    AssertThrow(false, ExcNotImplemented());

  solver->output_solution(parameters.output_filename);

  while (navier_stokes_solver.time_stepping.at_end() == false)
    {
      solver->advance_time_step();

      solver->output_solution(parameters.output_filename);
    }
}

int
main(int argc, char **argv)
{
  using namespace dealii;


  try
    {
      deallog.depth_console(0);
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

      std::string paramfile;
      if (argc > 1)
        paramfile = argv[1];
      else
        paramfile = "sharp_interfaces_04.prm";

      TwoPhaseParameters parameters(paramfile);
      if (parameters.dimension == 2)
        {
          MicroFluidicProblem<2> flow_problem(parameters);
          flow_problem.run();
        }
      else
        AssertThrow(false, ExcNotImplemented());
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;
}
