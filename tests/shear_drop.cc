// --------------------------------------------------------------------------
//
// Copyright (C) 2010 - 2016 by the adaflo authors
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

#include <deal.II/grid/grid_generator.h>

#include <adaflo/level_set_okz.h>
#include <adaflo/level_set_okz_matrix.h>
#include <adaflo/parameters.h>
#include <adaflo/phase_field.h>


using namespace dealii;


struct TwoPhaseParameters : public FlowParameters
{
  TwoPhaseParameters(const std::string &parameter_filename)
  {
    ParameterHandler prm;
    FlowParameters::declare_parameters(prm);
    prm.enter_subsection("Problem-specific");
    prm.declare_entry("two-phase method",
                      "level set okz",
                      Patterns::Selection(
                        "level set okz|level set okz matrix|phase field"),
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
class InitialValuesLS : public Function<dim>
{
public:
  InitialValuesLS()
    : Function<dim>(1, 0)
  {}

  double
  value(const Point<dim> &p, const unsigned int /*component*/) const
  {
    const double radius = 1.0;
    Point<dim>   origin(4.0, 4.0);
    return p.distance(origin) - radius;
  }
};


template <int dim>
class BCVelocityField : public Function<dim>
{
public:
  BCVelocityField()
    : Function<dim>(dim, 0)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const
  {
    if (component == 0)
      return p[1] - 4.0;
    else
      return 0.0;
  }
};



// @sect3{The <code>MicroFluidicProblem</code> class template}
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

  TwoPhaseParameters                        parameters;
  parallel::distributed::Triangulation<dim> triangulation;

  std::unique_ptr<TwoPhaseBaseAlgorithm<dim>> solver;
};


template <int dim>
MicroFluidicProblem<dim>::MicroFluidicProblem(const TwoPhaseParameters &parameters)
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  ,

  parameters(parameters)
  , triangulation(mpi_communicator)
{
  if (parameters.solver_method == "level set okz")
    solver = std::make_unique<LevelSetOKZSolver<dim>>(parameters, triangulation);
  else if (parameters.solver_method == "level set okz matrix")
    solver = std::make_unique<LevelSetOKZMatrixSolver<dim>>(parameters, triangulation);
  else if (parameters.solver_method == "phase field")
    solver = std::make_unique<PhaseFieldSolver<dim>>(parameters, triangulation);
  else
    AssertThrow(false, ExcMessage("Unknown solver selected in parameter file"));
}



template <int dim>
void
MicroFluidicProblem<dim>::run()
{
  // create mesh
  std::vector<unsigned int> subdivisions(dim, 5);

  const Point<dim> bottom_left;
  const Point<dim> top_right = (dim == 2 ? Point<dim>(8, 8) : Point<dim>(8, 8, 8));
  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            subdivisions,
                                            bottom_left,
                                            top_right);

  // set boundary indicator to 2 on left and right face -> symmetry boundary
  typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin(),
    endc = triangulation.end();

  for (; cell != endc; ++cell)
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() == false)
          continue;

        if (std::fabs(cell->face(face)->center()[0] - 8) < 1e-14)
          cell->face(face)->set_boundary_id(1);
        if (std::fabs(cell->face(face)->center()[0]) < 1e-14)
          cell->face(face)->set_boundary_id(2);
      }

  AssertThrow(parameters.global_refinements < 12, ExcInternalError());

  solver->set_velocity_dirichlet_boundary(0, std::make_shared<BCVelocityField<dim>>());
  solver->set_periodic_direction(0, 1, 2);

  solver->fix_pressure_constant(0);

  solver->setup_problem(BCVelocityField<dim>(), InitialValuesLS<dim>());
  solver->output_solution(parameters.output_filename);

  std::vector<std::vector<double>> solution_data;
  solution_data.push_back(solver->compute_bubble_statistics(0));

  // time loop
  bool first_output = true;
  while (solver->get_time_stepping().at_end() == false)
    {
      solver->advance_time_step();

      solver->output_solution(parameters.output_filename);

      solver->refine_grid();

      solution_data.push_back(solver->compute_bubble_statistics());

      if (solution_data.size() > 0 &&
          Utilities::MPI::this_mpi_process(triangulation.get_communicator()) == 0 &&
          solver->get_time_stepping().at_tick(parameters.output_frequency))
        {
          const int time_step = 1.000001e4 * solver->get_time_stepping().step_size();

          std::ostringstream filename3;
          filename3 << parameters.output_filename << "-"
                    << Utilities::int_to_string((int)parameters.adaptive_refinements, 1)
                    << "-" << Utilities::int_to_string(parameters.global_refinements, 3)
                    << "-" << Utilities::int_to_string(time_step, 4) << ".txt";

          std::fstream output_positions3(filename3.str().c_str(),
                                         first_output ? std::ios::out :
                                                        std::ios::out | std::ios::app);

          output_positions3.precision(14);
          if (first_output)
            output_positions3
              << "#    time        area      perimeter   circularity   bubble_xvel   bubble_yvel   bubble_xpos    bubble_ypos"
              << std::endl;
          for (unsigned int i = 0; i < solution_data.size(); ++i)
            {
              output_positions3 << " ";
              for (unsigned int j = 0; j < solution_data[i].size(); ++j)
                output_positions3 << solution_data[i][j] << "   ";
              output_positions3 << std::endl;
            }
          solution_data.clear();
          first_output = false;
        }
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
        paramfile = "rising_bubble.prm";

      TwoPhaseParameters parameters(paramfile);
      if (parameters.dimension == 2)
        {
          MicroFluidicProblem<2> flow_problem(parameters);
          flow_problem.run();
        }
      else if (parameters.dimension == 3)
        {
          MicroFluidicProblem<3> flow_problem(parameters);
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
