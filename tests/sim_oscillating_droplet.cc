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

static constexpr double side_length      = 2.5;
static constexpr double larger_semiaxis  = 3. / 2. * 0.5;
static constexpr double smaller_semiaxis = 2. / 3. * 0.5;
static constexpr double eps              = 0.70710678118;

namespace DistanceFunctions
{
  template <int dim>
  inline double
  ellipsoidal_manifold(const Point<dim> &p,
                       const Point<dim> &center,
                       const double      radius_x,
                       const double      radius_y,
                       const double      radius_z = 0)
  {
    if (dim == 3)
      return -std::pow(p[0] - center[0], 2) / std::pow(radius_x, 2) -
             std::pow(p[1] - center[1], 2) / std::pow(radius_y, 2) -
             std::pow(p[2] - center[2], 2) / std::pow(radius_z, 2) + 1;
    else if (dim == 2)
      return -std::pow(p[0] - center[0], 2) / std::pow(radius_x, 2) -
             std::pow(p[1] - center[1], 2) / std::pow(radius_y, 2) + 1;
    else if (dim == 1)
      return -std::pow(p[0] - center[0], 2) / std::pow(radius_x, 2) + 1;
    else
      AssertThrow(false, ExcMessage("Ellipsoidal manifold: dim must be 1, 2 or 3."));
  }
} // namespace DistanceFunctions

namespace UtilityFunctions::CharacteristicFunctions
{
  inline double
  tanh_characteristic_function(const double &distance, const double &eps)
  {
    return std::tanh(distance / (2. * eps));
  }
} // namespace UtilityFunctions::CharacteristicFunctions

struct TwoPhaseParameters : public FlowParameters
{
  TwoPhaseParameters(const std::string &parameter_filename)
  {
    ParameterHandler prm;
    FlowParameters::declare_parameters(prm);
    prm.enter_subsection("Problem-specific");
    prm.declare_entry("two-phase method",
                      "front tracking",
                      Patterns::Selection(
                        "front tracking|mixed level set|sharp level set|level set"),
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
  value(const Point<dim> &p, const unsigned int) const
  {
    const Point<dim> center;

    return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
      DistanceFunctions::ellipsoidal_manifold<dim>(
        p, center, larger_semiaxis, smaller_semiaxis),
      eps);
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
  void
  evaluate_spurious_velocities(NavierStokes<dim> &navier_stokes_solver);

  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;

  mutable TimerOutput timer;

  TwoPhaseParameters                        parameters;
  parallel::distributed::Triangulation<dim> triangulation;

  std::vector<std::vector<double>> solution_data;
};

template <int dim>
MicroFluidicProblem<dim>::MicroFluidicProblem(const TwoPhaseParameters &parameters)
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , timer(pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  , parameters(parameters)
  , triangulation(mpi_communicator)
{}


template <int dim>
void
MicroFluidicProblem<dim>::evaluate_spurious_velocities(
  NavierStokes<dim> &navier_stokes_solver)
{
  double               local_norm_velocity, norm_velocity;
  const QIterated<dim> quadrature_formula(QTrapez<1>(), parameters.velocity_degree + 2);
  const unsigned int   n_q_points = quadrature_formula.size();

  const MPI_Comm &            mpi_communicator = triangulation.get_communicator();
  FEValues<dim>               fe_values(navier_stokes_solver.get_fe_u(),
                          quadrature_formula,
                          update_values);
  std::vector<Tensor<1, dim>> velocity_values(n_q_points);
  local_norm_velocity = 0;

  const FEValuesExtractors::Vector velocities(0);

  typename DoFHandler<dim>::active_cell_iterator
    cell = navier_stokes_solver.get_dof_handler_u().begin_active(),
    endc = navier_stokes_solver.get_dof_handler_u().end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values[velocities].get_function_values(navier_stokes_solver.solution.block(0),
                                                  velocity_values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          local_norm_velocity = std::max(local_norm_velocity, velocity_values[q].norm());
      }
  norm_velocity = Utilities::MPI::max(local_norm_velocity, mpi_communicator);

  double pressure_jump = 0;
  {
    QGauss<dim>       quadrature_formula(parameters.velocity_degree + 1);
    QGauss<dim - 1>   face_quadrature_formula(parameters.velocity_degree + 1);
    FEValues<dim>     ns_values(navier_stokes_solver.get_fe_p(),
                            quadrature_formula,
                            update_values | update_JxW_values);
    FEFaceValues<dim> fe_face_values(navier_stokes_solver.get_fe_p(),
                                     face_quadrature_formula,
                                     update_values | update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    std::vector<double> p_values(n_q_points);
    std::vector<double> p_face_values(face_quadrature_formula.size());

    // With all this in place, we can go on with the loop over all cells and
    // add the local contributions.
    //
    // The first thing to do is to evaluate the FE basis functions at the
    // quadrature points of the cell, as well as derivatives and the other
    // quantities specified above.  Moreover, we need to reset the local
    // matrices and right hand side before filling them with new information
    // from the current cell.
    const FEValuesExtractors::Scalar p(dim);
    double pressure_average = 0, one_average = 0, press_b = 0, one_b = 0;
    typename DoFHandler<dim>::active_cell_iterator
      endc    = navier_stokes_solver.get_dof_handler_p().end(),
      ns_cell = navier_stokes_solver.get_dof_handler_p().begin_active();

    for (; ns_cell != endc; ++ns_cell)
      if (ns_cell->is_locally_owned())
        {
          ns_values.reinit(ns_cell);

          if (ns_cell->center().norm() < 0.1)
            {
              ns_values.get_function_values(navier_stokes_solver.solution.block(1),
                                            p_values);
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  pressure_average += p_values[q] * ns_values.JxW(q);
                  one_average += ns_values.JxW(q);
                }
            }
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
            if (ns_cell->face(face)->at_boundary())
              {
                fe_face_values.reinit(ns_cell, face);
                fe_face_values.get_function_values(navier_stokes_solver.solution.block(1),
                                                   p_face_values);
                for (unsigned int q = 0; q < face_quadrature_formula.size(); ++q)
                  {
                    press_b += p_face_values[q] * fe_face_values.JxW(q);
                    one_b += fe_face_values.JxW(q);
                  }
              }
        }

    const double global_p_avg = Utilities::MPI::sum(pressure_average, mpi_communicator);
    const double global_o_avg = Utilities::MPI::sum(one_average, mpi_communicator);
    const double global_p_bou = Utilities::MPI::sum(press_b, mpi_communicator);
    const double global_o_bou = Utilities::MPI::sum(one_b, mpi_communicator);
    pressure_jump = ((global_p_avg / global_o_avg - global_p_bou / global_o_bou) -
                     2. * (dim - 1) * parameters.surface_tension) /
                    (2 * (dim - 1) * parameters.surface_tension) * 100.;
    std::cout.precision(8);
    // output relative pressure error
    pcout << "  Error in pressure jump: " << pressure_jump << " %" << std::endl;
  }

  // output spurious currents
  pcout << "  Size spurious currents, absolute: " << norm_velocity << std::endl;
}



template <int dim>
void
MicroFluidicProblem<dim>::run()
{
  GridGenerator::subdivided_hyper_cube(triangulation,
                                       parameters.global_refinements,
                                       -side_length / 2,
                                       +side_length / 2);

  NavierStokes<dim> navier_stokes_solver(parameters, triangulation, &timer);

  navier_stokes_solver.set_no_slip_boundary(0);
  navier_stokes_solver.fix_pressure_constant(0);
  navier_stokes_solver.setup_problem(Functions::ZeroFunction<dim>(dim));
  navier_stokes_solver.output_solution(parameters.output_filename);

  parallel::shared::Triangulation<dim - 1, dim> surface_mesh(mpi_communicator);
  GridGenerator::hyper_sphere(surface_mesh, Point<dim>(0.02, 0.03), 0.5);
  surface_mesh.refine_global(5);

  std::unique_ptr<SharpInterfaceSolver> solver;

  Assert(parameters.solver_method != "front tracking", ExcNotImplemented());
  Assert(parameters.solver_method != "mixed level set", ExcNotImplemented());

  if (parameters.solver_method == "front tracking")
    {
      AssertDimension(Utilities::MPI::n_mpi_processes(mpi_communicator), 1);
      solver =
        std::make_unique<FrontTrackingSolver<dim>>(navier_stokes_solver, surface_mesh);
    }
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
  else
    AssertThrow(false, ExcNotImplemented());

  solver->output_solution(parameters.output_filename);

  while (navier_stokes_solver.time_stepping.at_end() == false)
    {
      solver->advance_time_step();

      solver->output_solution(parameters.output_filename);

      // evaluate velocity norm and pressure jump
      evaluate_spurious_velocities(navier_stokes_solver);
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
