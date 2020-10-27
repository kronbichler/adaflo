// --------------------------------------------------------------------------
//
// Copyright (C) 2009 - 2016 by the adaflo authors
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

#include <adaflo/parameters.h>
#include <adaflo/level_set_okz.h>
#include <adaflo/phase_field.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>


using namespace dealii;


struct TwoPhaseParameters : public FlowParameters
{
  TwoPhaseParameters (const std::string &parameter_filename)
  {
    ParameterHandler prm;
    FlowParameters::declare_parameters (prm);
    prm.enter_subsection ("Problem-specific");
    prm.declare_entry ("two-phase method","level set okz",
                       Patterns::Selection("level set okz|phase field"),
                       "Defines the two-phase method to be used");
    prm.leave_subsection();
    check_for_file(parameter_filename, prm);
    parse_parameters (parameter_filename, prm);
    prm.enter_subsection ("Problem-specific");
    solver_method = prm.get("two-phase method");
    prm.leave_subsection();
  }

  std::string solver_method;
};


template <int dim>
class InitialValuesLS : public Function<dim>
{
public:
  InitialValuesLS ()
    :
    Function<dim>(1, 0)
  {}

  double value (const Point<dim> &p,
                const unsigned int component = 0) const
  {
    // set radius of bubble to 0.5, slightly shifted away from the center
    Point<dim> center;
    for (unsigned int d=0; d<dim; ++d)
      center[d] = 0.02+0.01*d;
    return p.distance(center) - 0.5;
  }
};



// @sect3{The <code>MicroFluidicProblem</code> class template}
template <int dim>
class MicroFluidicProblem
{
public:
  MicroFluidicProblem (const TwoPhaseParameters &parameters);
  void run ();

private:

  void evaluate_spurious_velocities();

  MPI_Comm mpi_communicator;
  ConditionalOStream  pcout;

  TwoPhaseParameters        parameters;
  parallel::distributed::Triangulation<dim>   triangulation;

  std::shared_ptr<TwoPhaseBaseAlgorithm<dim> > solver;

  std::vector<std::vector<double> > solution_data;
};


template <int dim>
MicroFluidicProblem<dim>::MicroFluidicProblem (const TwoPhaseParameters &parameters)
  :
  mpi_communicator(MPI_COMM_WORLD),
  pcout (std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),

  parameters (parameters),
  triangulation (mpi_communicator)
{
  if (parameters.solver_method == "level set okz")
    solver.reset(new LevelSetOKZSolver<dim>(parameters, triangulation));
  else if (parameters.solver_method == "phase field")
    solver.reset(new PhaseFieldSolver<dim>(parameters, triangulation));
  else
    AssertThrow(false, ExcMessage("Unknown solver selected in parameter file"));
}



template <int dim>
void MicroFluidicProblem<dim>::evaluate_spurious_velocities()
{

  double local_norm_velocity, norm_velocity;
  const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                           parameters.velocity_degree+2);
  const unsigned int n_q_points = quadrature_formula.size();

  const NavierStokes<dim> &navier_stokes = solver->get_navier_stokes();
  const MPI_Comm &mpi_communicator = triangulation.get_communicator();
  FEValues<dim> fe_values (navier_stokes.get_fe_u(), quadrature_formula,
                           update_values);
  std::vector<Tensor<1,dim> > velocity_values(n_q_points);
  local_norm_velocity = 0;

  const FEValuesExtractors::Vector velocities (0);

  typename DoFHandler<dim>::active_cell_iterator
  cell = navier_stokes.get_dof_handler_u().begin_active(),
  endc = navier_stokes.get_dof_handler_u().end();
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit (cell);
        fe_values[velocities].get_function_values (navier_stokes.solution.block(0),
                                                   velocity_values);
        for (unsigned int q=0; q<n_q_points; ++q)
          local_norm_velocity = std::max (local_norm_velocity,
                                          velocity_values[q].norm());
      }
  norm_velocity = Utilities::MPI::max(local_norm_velocity, mpi_communicator);

  double pressure_jump = 0;
  {
    QGauss<dim>   quadrature_formula(parameters.velocity_degree+1);
    QGauss<dim-1> face_quadrature_formula(parameters.velocity_degree+1);
    FEValues<dim> ns_values (navier_stokes.get_fe_p(), quadrature_formula,
                             update_values | update_JxW_values);
    FEFaceValues<dim> fe_face_values (navier_stokes.get_fe_p(),
                                      face_quadrature_formula,
                                      update_values | update_JxW_values);

    const unsigned int   n_q_points = quadrature_formula.size();

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
    const FEValuesExtractors::Scalar p (dim);
    double pressure_average = 0, one_average = 0, press_b = 0, one_b = 0;
    typename DoFHandler<dim>::active_cell_iterator
    endc = navier_stokes.get_dof_handler_p().end(),
    ns_cell = navier_stokes.get_dof_handler_p().begin_active();
    for ( ; ns_cell!=endc; ++ns_cell)
      if (ns_cell->is_locally_owned())
        {
          ns_values.reinit (ns_cell);
          if (ns_cell->center().norm() < 0.1)
            {
              ns_values.get_function_values(navier_stokes.solution.block(1),
                                            p_values);
              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  pressure_average += p_values[q] * ns_values.JxW(q);
                  one_average += ns_values.JxW(q);
                }
            }
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            if (ns_cell->face(face)->at_boundary())
              {
                fe_face_values.reinit (ns_cell, face);
                fe_face_values.get_function_values(navier_stokes.solution.block(1),
                                                   p_face_values);
                for (unsigned int q=0; q<face_quadrature_formula.size(); ++q)
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
    pressure_jump = ((global_p_avg / global_o_avg - global_p_bou/global_o_bou)
                     - 2.*(dim-1)*parameters.surface_tension)/
                    (2*(dim-1)*parameters.surface_tension) * 100. ;
    std::cout.precision (8);
    pcout << "  Error in pressure jump: " << pressure_jump << " %" << std::endl;
  }

  // calculate spurious currents
  pcout << "  Size spurious currents, absolute: "
        << norm_velocity << std::endl;

  std::vector<double> data(3);
  data[0] = solver->get_time_stepping().now();
  data[1] = norm_velocity;
  data[2] = pressure_jump;
  if (solution_data.size() && data[0] == solution_data.back()[0])
    solution_data.back().insert (solution_data.back().end(),
                                 data.begin()+1, data.end());
  else
    solution_data.push_back (data);
}



template <int dim>
void MicroFluidicProblem<dim>::run ()
{
  // create mesh
  GridGenerator::subdivided_hyper_cube (triangulation, parameters.global_refinements,
                                        -2.5, 2.5);
  solver->set_no_slip_boundary(0);
  solver->fix_pressure_constant(0);
  solver->setup_problem(Functions::ZeroFunction<dim>(dim), InitialValuesLS<dim>());
  solver->output_solution(parameters.output_filename);

  // time loop
  bool first_output = true;
  while (solver->get_time_stepping().at_end() == false)
    {
      solver->advance_time_step();

      solver->output_solution(parameters.output_filename);

      evaluate_spurious_velocities();

      if (solution_data.size() > 0 &&
          Utilities::MPI::this_mpi_process(triangulation.get_communicator()) == 0 &&
          solver->get_time_stepping().at_tick(parameters.output_frequency))
        {
          const int time_step = 1.000001e4*solver->get_time_stepping().step_size();

          std::ostringstream filename3;
          filename3 << parameters.output_filename << "-"
                    << Utilities::int_to_string((int)parameters.epsilon,1) << "-";

          filename3 << Utilities::int_to_string((int)(100./parameters.surface_tension), 4) << "-";
          filename3 << Utilities::int_to_string((int)parameters.adaptive_refinements, 1) << "-"
                    << Utilities::int_to_string(parameters.global_refinements, 3) << "-"
                    << Utilities::int_to_string(time_step, 4)
                    << ".txt";

          std::fstream output_positions3 (filename3.str().c_str(),
                                          first_output ? std::ios::out : std::ios::out
                                          | std::ios::app);

          output_positions3.precision (14);
          for (unsigned int i=0; i<solution_data.size(); ++i)
            {
              output_positions3 << " ";
              for (unsigned int j=0; j<solution_data[i].size(); ++j)
                output_positions3 << solution_data[i][j] << "   ";
              output_positions3 << std::endl;
            }
          solution_data.clear();
          first_output = false;
        }
    }
}




int main (int argc, char **argv)
{
  using namespace dealii;


  try
    {
      deallog.depth_console (0);
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      std::string paramfile;
      if (argc>1)
        paramfile = argv[1];
      else
        paramfile = "spurious_currents.prm";

      TwoPhaseParameters parameters (paramfile);
      if (parameters.dimension == 2)
        {
          MicroFluidicProblem<2> flow_problem (parameters);
          flow_problem.run ();
        }
      else if (parameters.dimension == 3)
        {
          MicroFluidicProblem<3> flow_problem (parameters);
          flow_problem.run ();
        }
      else
        AssertThrow (false, ExcNotImplemented());
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
