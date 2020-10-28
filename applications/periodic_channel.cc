// --------------------------------------------------------------------------
//
// Copyright (C) 2014 - 2016 by the adaflo authors
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

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_time.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>


#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/matrix_free/fe_evaluation.h>


#include <adaflo/navier_stokes.h>
#include <adaflo/time_stepping.h>

#include <fstream>
#include <iostream>
#include <iomanip>

using namespace dealii;



/* ----------------------------------------------------------------------- */


// Initial velocity field
template <int dim>
class InitialChannel : public Function<dim>
{
public:
  InitialChannel ()
    :
    Function<dim>(dim, 0.)
  {}

  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &values) const;
};

template <int dim>
void
InitialChannel<dim>::vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const
{
  AssertThrow(dim == 2 || dim == 3, ExcNotImplemented());

  // initial value is a quartic profile in y-direction and some perturbation
  // into the other directions to trigger the transition to turbulence
  values(0) = (1.-p[1]*p[1]*p[1]*p[1])*1.25;
  values(1) = 0.2*(1.-p[1]*p[1]*p[1]*p[1])*std::cos(p[2]*3.);
  if (dim == 3)
    values(2) = 0.2*(1.-p[1]*p[1]*p[1]*p[1])*std::sin(p[2]*3.);
}




// @sect3{The <code>PeriodicChannelproblem</code> class template}
template <int dim>
class PeriodicChannelProblem
{

public:
  PeriodicChannelProblem (const FlowParameters &parameters);
  void run ();

private:

  void output_results () const;

  void perform_data_exchange (std::vector<double> &positions,
                              std::vector<Tensor<1,dim> > &velocities) const;
  template <int velocity_degree>
  void
  local_compute_force (const MatrixFree<dim,double> &data,
                       LinearAlgebra::distributed::Vector<double> &dst,
                       const LinearAlgebra::distributed::Vector<double> &,
                       const std::pair<unsigned int,unsigned int> &cell_range) const;

  void compute_force();

  ConditionalOStream  pcout;

  mutable TimerOutput timer;

  parallel::distributed::Triangulation<dim>   triangulation;
  NavierStokes<dim>    navier_stokes;

  const double cavity_depth;
};



template <int dim>
PeriodicChannelProblem<dim>::PeriodicChannelProblem (const FlowParameters &parameters)
  :
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  timer (pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
  triangulation(MPI_COMM_WORLD),
  navier_stokes (parameters, triangulation,
                 &timer),
  cavity_depth  (3.)
{}



template <int dim>
void PeriodicChannelProblem<dim>::output_results () const
{
  if (!navier_stokes.time_stepping.at_tick(navier_stokes.get_parameters().output_frequency))
    return;

  timer.enter_subsection("Output solution.");

  navier_stokes.solution.update_ghost_values();

  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  vector_component_interpretation
  (dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector (navier_stokes.get_dof_handler_u(),
                            navier_stokes.solution.block(0),
                            std::vector<std::string>(dim,"velocity"),
                            vector_component_interpretation);
  data_out.add_data_vector (navier_stokes.get_dof_handler_p(),
                            navier_stokes.solution.block(1),
                            "pressure");
  data_out.build_patches (navier_stokes.get_parameters().velocity_degree);

  navier_stokes.write_data_output(navier_stokes.get_parameters().output_filename,
                                  navier_stokes.time_stepping,
                                  navier_stokes.get_parameters().output_frequency,
                                  data_out);

  timer.leave_subsection();
}



template <int dim>
void PeriodicChannelProblem<dim>::perform_data_exchange(std::vector<double> &positions,
                                                        std::vector<Tensor<1,dim> > &velocities) const
{
  // Exchange data with other processors
  std::vector<unsigned int>
  receive_count(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));

  unsigned int n_send_elements = velocities.size();
  MPI_Gather(&n_send_elements, 1, MPI_UNSIGNED, &receive_count[0],
             1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  for (unsigned int i = 1;
       i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
    {
      // Each processor sends the interface_points he deals with to processor
      // 0
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == i)
        {
          // put data into a std::vector<double> to create a data type that
          // MPI understands
          std::vector<double> send_data((dim+1)*velocities.size());
          for (unsigned int j=0; j<velocities.size(); ++j)
            {
              for (unsigned int d=0; d<dim; ++d)
                send_data[j*(dim+1)+d] = velocities[j][d];
              send_data[j*(dim+1)+dim] = positions[j];
            }

          MPI_Send (&send_data[0], send_data.size(), MPI_DOUBLE, 0, i,
                    MPI_COMM_WORLD);
        }

      // Processor 0 receives data from the other processors
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::vector<double> receive_data((dim+1)*receive_count[i]);
          int ierr = MPI_Recv (&receive_data[0], receive_data.size(),
                               MPI_DOUBLE, i, i, MPI_COMM_WORLD,
                               MPI_STATUSES_IGNORE);
          AssertThrow (ierr == MPI_SUCCESS, ExcInternalError());
          for (unsigned int j=0; j<receive_count[i]; ++j)
            {
              Tensor<1,dim> velocity;
              for (unsigned int d=0; d<dim; ++d)
                velocity[d] = receive_data[j*(dim+1)+d];
              positions.push_back(receive_data[j*(dim+1)+dim]);
              velocities.push_back(velocity);
            }
        }
    }
}



template <int dim>
Point<dim>
grid_transform (const Point<dim> &in)
{
  Point<dim> out = in;
  out[1] = std::tanh(1.*(2.*in(1)-1))/std::tanh(1);
  return out;
}



template <int dim>
template <int velocity_degree>
void
PeriodicChannelProblem<dim>::local_compute_force (const MatrixFree<dim,double> &data,
                                                  LinearAlgebra::distributed::Vector<double> &dst,
                                                  const LinearAlgebra::distributed::Vector<double> &,
                                                  const std::pair<unsigned int,unsigned int> &cell_range) const
{
  FEEvaluation<dim,velocity_degree,velocity_degree+1,dim> vel_values(data,0,0);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      vel_values.reinit(cell);
      Tensor<1,dim,VectorizedArray<double> > force;
      force[0] = make_vectorized_array(-0.00337204);
      for (unsigned int q=0; q<vel_values.n_q_points; ++q)
        {
          vel_values.submit_value(force, q);
        }
      vel_values.integrate(true, false);
      vel_values.distribute_local_to_global(dst);
    }
}



template <int dim>
void
PeriodicChannelProblem<dim>::compute_force()
{
  this->navier_stokes.user_rhs = 0;
#define OPERATION(velocity_degree)                                      \
  navier_stokes.get_matrix().get_matrix_free()                          \
  .cell_loop (&PeriodicChannelProblem<dim>::template local_compute_force<velocity_degree>, \
              this, navier_stokes.user_rhs.block(0),                  \
              navier_stokes.user_rhs.block(0))

  const unsigned int degree_u =
    this->navier_stokes.get_dof_handler_u().get_fe().degree;
  if (degree_u == 2)
    OPERATION(2);
  else if (degree_u == 3)
    OPERATION(3);
  else if (degree_u == 4)
    OPERATION(4);
  else if (degree_u == 5)
    OPERATION(5);
  else if (degree_u == 6)
    OPERATION(6);
  else
    AssertThrow(false, ExcNotImplemented());
}



template <int dim>
void PeriodicChannelProblem<dim>::run ()
{
  timer.enter_subsection ("Setup grid and initial condition.");

  pcout << "Running a " << dim << "D channel flow problem "
        << "using " << navier_stokes.time_stepping.name()
        << ", Q"  << navier_stokes.get_fe_u().degree
        << "/Q" << navier_stokes.get_fe_p().degree
        << (navier_stokes.get_parameters().augmented_taylor_hood ? "+" : "")
        << " elements on "
        << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << " processes"
        << std::endl;

  {
    Point<dim> coordinates;
    coordinates[0] = 2*numbers::PI;
    coordinates[1] = 1.;
    if (dim == 3)
      coordinates[2] = 2./3.*numbers::PI;
    AssertThrow(navier_stokes.get_parameters().global_refinements % 4 == 0,
                ExcMessage("The number of elements per direction must be divisible by 4"));
    std::vector<unsigned int> refinements(dim, navier_stokes.get_parameters().
                                          global_refinements / 4);
    GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,
                                               Point<dim>(), coordinates);

    std::vector<unsigned int> face_to_indicator_list(6);
    face_to_indicator_list[0] = 1;
    face_to_indicator_list[1] = 3;
    face_to_indicator_list[2] = face_to_indicator_list[3] = 0;
    face_to_indicator_list[4] = 2;
    face_to_indicator_list[5] = 4;
    for (typename Triangulation<dim>::cell_iterator cell = triangulation.begin();
         cell!= triangulation.end(); ++cell)
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary())
          cell->face(f)->set_all_boundary_ids(face_to_indicator_list[f]);

    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
    periodic_faces;
    GridTools::collect_periodic_faces(triangulation, 1, 3, 0, periodic_faces);
    GridTools::collect_periodic_faces(triangulation, 2, 4, 2, periodic_faces);
    triangulation.add_periodicity(periodic_faces);

    triangulation.refine_global(2);

    GridTools::transform (&grid_transform<dim>, triangulation);
  }

  navier_stokes.set_velocity_dirichlet_boundary(0, std::shared_ptr<Function<dim> > (new Functions::ZeroFunction<dim>(dim)));
  navier_stokes.fix_pressure_constant(0, std::shared_ptr<Function<dim> > (new Functions::ZeroFunction<dim>()));
  navier_stokes.set_periodic_direction(0, 1, 3);
  navier_stokes.set_periodic_direction(2, 2, 4);

  navier_stokes.distribute_dofs();

  timer.leave_subsection();

  navier_stokes.time_stepping.restart();
  navier_stokes.print_n_dofs();

  navier_stokes.initialize_data_structures();
  navier_stokes.initialize_matrix_free();

  VectorTools::interpolate (navier_stokes.get_dof_handler_u(),
                            InitialChannel<dim>(),
                            navier_stokes.solution.block(0));
  output_results();

  // @sect5{Time loop}
  while (navier_stokes.time_stepping.at_end() == false)
    {
      navier_stokes.init_time_advance();
      compute_force();
      navier_stokes.evaluate_time_step();
      output_results();
    }
}


/* ----------------------------------------------------------------------- */



/* ----------------------------------------------------------------------- */



int main (int argc, char **argv)
{
  /* we initialize MPI at the start of the program. Since we will in general
   * mix MPI parallelization with threads, we also set the third argument in
   * MPI_InitFinalize that controls the number of threads to an invalid
   * number, which means that the TBB library chooses the number of threads
   * automatically, typically to the number of available cores in the
   * system. As an alternative, you can also set this number manually if you
   * want to set a specific number of threads (e.g. when MPI-only is required)
   * (cf step-40 and 48).
   */
  try
    {
      using namespace dealii;


      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, -1);
      deallog.depth_console (0);

      std::string paramfile;
      if (argc>1)
        paramfile = argv[1];
      else
        paramfile = "periodic_channel.prm";

      FlowParameters parameters (paramfile);

      if (parameters.dimension == 2)
        {
          PeriodicChannelProblem<2> beltrami_problem (parameters);
          beltrami_problem.run ();
        }
      else if (parameters.dimension == 3)
        {
          PeriodicChannelProblem<3> beltrami_problem (parameters);
          beltrami_problem.run ();
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
