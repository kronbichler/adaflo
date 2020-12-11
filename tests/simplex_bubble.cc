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

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/simplex/grid_generator.h>

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
    const double radius               = 0.25;
    Point<dim>   distance_from_origin = p;
    for (unsigned int i = 0; i < dim; ++i)
      distance_from_origin[i] = 0.5;
    return p.distance(distance_from_origin) - radius;
  }
};



// @sect3{The <code>MicroFluidicProblem</code> class template}
template <int dim>
class MicroFluidicProblem
{
public:
  MicroFluidicProblem(TwoPhaseParameters &parameters);
  void
  run();

  unsigned int
  fix_n_refinements(FlowParameters &parameters)
  {
    unsigned int temp = parameters.global_refinements;

    parameters.global_refinements = 0;

    return temp;
  }

private:
  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;

  const unsigned int                   n_refinements;
  TwoPhaseParameters                   parameters;
  parallel::shared::Triangulation<dim> triangulation;


  std::unique_ptr<TwoPhaseBaseAlgorithm<dim>> solver;
};


template <int dim>
MicroFluidicProblem<dim>::MicroFluidicProblem(TwoPhaseParameters &parameters)
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , n_refinements(fix_n_refinements(parameters))
  , parameters(parameters)
  , triangulation(mpi_communicator,
                  ::Triangulation<dim>::none,
                  true,
                  parallel::shared::Triangulation<dim>::partition_zorder)
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
  const unsigned int        n = Utilities::pow(2, n_refinements);
  std::vector<unsigned int> subdivisions(dim, 5 * n);
  subdivisions[dim - 1] = 10 * n;

  const Point<dim> bottom_left;
  const Point<dim> top_right = (dim == 2 ? Point<dim>(1, 2) : Point<dim>(1, 1, 2));

  if (parameters.use_simplex_mesh)
    {
      if (false)
        {
          GridGenerator::subdivided_hyper_rectangle_with_simplices(triangulation,
                                                                   subdivisions,
                                                                   bottom_left,
                                                                   top_right);
        }
      else
        {
          GridIn<dim> grid_in;
          grid_in.attach_triangulation(triangulation);

          std::string file_name;

          if (dim == 2)
            file_name = "simplex_bubble_" + std::to_string(n_refinements) + ".msh";
          else
            file_name = "simplex_bubble_3D_" + std::to_string(n_refinements) + ".msh";

          std::ifstream input_file(file_name);
          grid_in.read_msh(input_file);
        }
    }
  else
    {
      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                subdivisions,
                                                bottom_left,
                                                top_right);
    }


  // set boundary indicator to 2 on left and right face -> symmetry boundary
  for (const auto &cell : triangulation.active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && (std::fabs(face->center()[0] - 1) < 1e-14 ||
                                  std::fabs(face->center()[0]) < 1e-14))
        face->set_boundary_id(2);

  AssertDimension(parameters.global_refinements, 0);

  solver->set_no_slip_boundary(0);
  solver->fix_pressure_constant(0);
  solver->set_symmetry_boundary(2);

  solver->setup_problem(Functions::ZeroFunction<dim>(dim), InitialValuesLS<dim>());
  solver->output_solution(parameters.output_filename, 2);

  // time loop
  while (solver->get_time_stepping().at_end() == false)
    {
      solver->advance_time_step();

      solver->output_solution(parameters.output_filename, 2);
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
