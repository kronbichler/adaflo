// --------------------------------------------------------------------------
//
// Copyright (C) 2013 - 2016 by the adaflo authors
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

#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <adaflo/phase_field.h>
#include <adaflo/time_stepping.h>

#include <fstream>
#include <iostream>
#include <iomanip>

using namespace dealii;



template <int dim>
class InitialValuesLS : public Function<dim>
{
public:
  InitialValuesLS (const double center = 2.5)
    :
    Function<dim>(1, 0),
    center(center)
  {}

  double value (const Point<dim> &p,
                const unsigned int /*component*/) const
  {
    return -p[0]+center;
  }

private:
  const double center;
};


template <int dim>
class SimilarityVelocity : public Function<dim>
{
public:
  SimilarityVelocity (const double wall_velocity = 0.,
                      const double viscosity_ratio = 0.,
                      const double phi = numbers::PI_2,
                      const double shift = 0.,
                      const double scaling = 1.)
    :
    Function<dim>(dim, 0),
    wall_velocity (wall_velocity),
    viscosity_ratio (viscosity_ratio),
    phi (phi),
    scaling (scaling)
  {
    x_wall = shift/scaling;
    const double pi = numbers::PI;
    const double C = std::cos(phi), S = std::sin(phi), delta = phi - pi;
    const double R = viscosity_ratio;
    const double D = (S*C-phi)*(delta*delta-S*S)+R*(delta-S*C)*(phi*phi-S*S);
    c_A = S * S * (S*S - delta*phi + R*(phi*phi-S*S))/D;
    d_A = S * (C * (S*S - delta*phi + R*(phi*phi-S*S)) - pi*S)/D;
    a_A = -1. - pi * c_A - d_A;
    b_A = -pi * d_A;
    c_B = S * S * (S*S - delta*delta + R*(delta*phi-S*S))/D;
    d_B = S * (C * (S*S - delta*delta + R*(delta*phi-S*S)) - R*pi*S)/D;
    a_B = -1. - d_B;
    b_B = 0;
  }

  virtual void vector_value (const Point<dim>   &p,
                             Vector<double>     &values) const;
  virtual void vector_gradient (const Point<dim> &p,
                                std::vector<Tensor<1,dim> > &gradients) const;

private:
  const double wall_velocity, viscosity_ratio, phi, scaling;
  double x_wall, a_A, b_A, c_A, d_A, a_B, b_B, c_B, d_B;
};



template <int dim>
void SimilarityVelocity<dim>::vector_value (const Point<dim> &unscaled,
                                            Vector<double>   &values) const
{
  Assert (values.size() == dim,
          ExcDimensionMismatch (values.size(), dim));

  Assert (dim==2, ExcNotImplemented());
  const Point<dim> p = unscaled / scaling;

  // transform to polar coordinates
  const double pi = numbers::PI, pi2 = numbers::PI_2;
  const double theta = ((std::fabs(p[1]+0.5)<1e-12) ?
                        ((p[0]-x_wall)<0 ? pi : 0) :
                        std::atan((x_wall-p[0])/(p[1]+0.5))+pi2);

  // distinguish between left and right fluid
  double v_radial, v_angular;
  const double c_t = std::cos(theta), s_t = std::sin(theta);

  // parameters based on convention:
  // v_radial  = - 1/r (d psi) / (d theta)
  // v_angular = (d psi) / (d r)
  if (theta > phi)
    {
      v_radial = -((a_A+theta*c_A+d_A)*c_t-(b_A+theta*d_A-c_A)*s_t);
      v_angular = (a_A+theta*c_A)*s_t+(b_A+theta*d_A)*c_t;
    }
  else
    {
      v_radial = -((a_B+theta*c_B+d_B)*c_t-(b_B+theta*d_B-c_B)*s_t);
      v_angular = (a_B+theta*c_B)*s_t+(b_B+theta*d_B)*c_t;
    }

  // transform back to x-y coordinates and scale
  // by wall velocity
  values(0) = wall_velocity * (v_radial*c_t - v_angular*s_t);
  values(1) = wall_velocity * (v_radial*s_t + v_angular*c_t);
}



template <int dim>
void SimilarityVelocity<dim>::vector_gradient (const Point<dim> &p,
                                               std::vector<Tensor<1,dim> > &gradients) const
{
  AssertDimension(gradients.size(), dim);
  Vector<double> values (dim), values_shift (dim);
  const double eps = 1e-8;

  // get gradient by difference quotient. in y
  // direction, we can only search in positive
  // direction.
  vector_value(p, values);
  for (unsigned int e=0; e<dim; ++e)
    {
      Point<dim> p1 = p;
      p1[e] += eps;
      vector_value(p1, values_shift);
      for (unsigned int d=0; d<1+dim; ++d)
        gradients[d][e] = (values_shift(d)-values(d))/eps;
    }
}



// @sect3{The <code>Beltramiproblem</code> class template}
template <int dim>
class ChannelProblem
{
public:
  ChannelProblem (const FlowParameters &parameters);
  void run ();

private:
  void output_results () const;

  const double channel_width;

  ConditionalOStream  pcout;

  FlowParameters parameters;

  parallel::distributed::Triangulation<dim>   triangulation;
  PhaseFieldSolver<dim> twophase_flow;

  mutable std::vector<std::vector<double> > solution_data;
  mutable double contact_velocity, old_contact_position;
};



template <int dim>
ChannelProblem<dim>::ChannelProblem (const FlowParameters &parameters)
  :
  channel_width (1.0),
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  parameters (parameters),
  triangulation(MPI_COMM_WORLD),
  twophase_flow (parameters, triangulation),
  contact_velocity (0),
  old_contact_position (0)
{}



template <int dim>
void ChannelProblem<dim>::output_results () const
{
  std::cout.precision (5);
  const FiniteElement<dim> &fe = twophase_flow.get_dof_handler().get_fe();
  const unsigned int lsdegree = fe.degree;
  const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                           lsdegree+2);
  const QTrapez<dim-1> quadrature_formula_faces;
  const QGauss<dim-1> quadrature_formula_face2 (lsdegree+2);
  const unsigned int n_q_points = quadrature_formula.size();

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values);
  FEFaceValues<dim> fe_face_values (fe, quadrature_formula_faces,
                                    update_values | update_quadrature_points);
  FEFaceValues<dim> fe_face2_values (fe, quadrature_formula_face2,
                                     update_values | update_JxW_values);
  std::vector<double> concentration_values(n_q_points);
  std::vector<double> concentration_faces (quadrature_formula_faces.size());
  std::vector<double> concentration_faces2 (quadrature_formula_face2.size());

  double min_concentration = twophase_flow.solution.linfty_norm(),
         max_concentration = -min_concentration;

  bool found_interface_x = false;
  double pos_x = 0.;
  double pos_lower_1 = 0., pos_lower_2 = 0., pos_upper_1 = 0., pos_upper_2 = 0.;
  double value_x = 0., value_y = 0.;
  double h_lower = 0., h_upper = 0.;
  typename DoFHandler<dim>::active_cell_iterator
  cell = twophase_flow.get_dof_handler().begin_active(),
  endc = twophase_flow.get_dof_handler().end();
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit (cell);
        fe_values.get_function_values (twophase_flow.solution.block(0),
                                       concentration_values);

        for (unsigned int q=0; q<n_q_points; ++q)
          {
            const double concentration = concentration_values[q];

            min_concentration = std::min (min_concentration, concentration);
            max_concentration = std::max (max_concentration, concentration);
          }

        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          {
            if (cell->center()[1] < 0 &&
                std::fabs(cell->face(face)->center()[1]) < 1e-8)
              {
                fe_face2_values.reinit(cell,face);
                fe_face2_values.get_function_values (twophase_flow.solution.block(0),
                                                     concentration_faces2);
                for (unsigned int q=0; q<quadrature_formula_face2.size(); ++q)
                  value_x += 0.5*(1+concentration_faces2[q])*fe_face2_values.JxW(q);

                if (found_interface_x == false)
                  {
                    fe_face_values.reinit(cell,face);
                    fe_face_values.get_function_values (twophase_flow.solution.block(0),
                                                        concentration_faces);
                    if (concentration_faces[0]*concentration_faces[1] <= 0)
                      {
                        found_interface_x = true;
                        const double left_cell = fe_face_values.get_quadrature_points()[0][0];
                        const double right_cell = fe_face_values.get_quadrature_points()[1][0];
                        const double position_zero = -concentration_faces[0] *
                                                     (right_cell-left_cell) / (concentration_faces[1]-concentration_faces[0])
                                                     + left_cell;
                        pos_x = -position_zero;
                      }
                  }
              }

            // cell at boundary
            if (std::fabs(cell->face(face)->center()[1]+
                          channel_width) < 1e-8)
              {
                fe_face2_values.reinit(cell,face);
                fe_face2_values.get_function_values (twophase_flow.solution.block(0),
                                                     concentration_faces2);
                for (unsigned int q=0; q<quadrature_formula_face2.size(); ++q)
                  value_y += 0.5*(1+concentration_faces2[q])*fe_face2_values.JxW(q);
              }
            if (std::fabs(cell->face(face)->center()[1]+
                          channel_width) < 1e-8)
              {
                fe_face_values.reinit(cell,face);
                fe_face_values.get_function_values (twophase_flow.solution.block(0),
                                                    concentration_faces);
                concentration_faces[0] += 1e-22*channel_width;
                concentration_faces[1] += 1e-22*channel_width;
                if (concentration_faces[0]*concentration_faces[1] <= 0)
                  {
                    const double left_cell = fe_face_values.get_quadrature_points()[0][0];
                    const double right_cell = fe_face_values.get_quadrature_points()[1][0];
                    const double position_zero = -concentration_faces[0] *
                                                 (right_cell-left_cell) / (concentration_faces[1]-concentration_faces[0])
                                                 + left_cell;
                    pos_lower_1 = position_zero;
                  }

                unsigned int opposite_face = (face%2==0)?(face+1):(face-1);
                fe_face_values.reinit(cell,opposite_face);
                fe_face_values.get_function_values (twophase_flow.solution.block(0),
                                                    concentration_faces);
                concentration_faces[0] += 1e-22*channel_width;
                concentration_faces[1] += 1e-22*channel_width;
                if (concentration_faces[0]*concentration_faces[1] <= 0)
                  {
                    const double left_cell = fe_face_values.get_quadrature_points()[0][0];
                    const double right_cell = fe_face_values.get_quadrature_points()[1][0];
                    const double position_zero = -concentration_faces[0] *
                                                 (right_cell-left_cell) / (concentration_faces[1]-concentration_faces[0])
                                                 + left_cell;
                    pos_lower_2 = position_zero;
                    h_lower = std::fabs(cell->face(face)->center()[1]-
                                        cell->face(opposite_face)->center()[1]);
                  }
              }
            if (std::fabs(cell->face(face)->center()[1]-channel_width) < 1e-8)
              {
                fe_face_values.reinit(cell,face);
                fe_face_values.get_function_values (twophase_flow.solution.block(0),
                                                    concentration_faces);
                concentration_faces[0] += 1e-22*channel_width;
                concentration_faces[1] += 1e-22*channel_width;
                if (concentration_faces[0]*concentration_faces[1] <= 0)
                  {
                    const double left_cell = fe_face_values.get_quadrature_points()[0][0];
                    const double right_cell = fe_face_values.get_quadrature_points()[1][0];
                    const double position_zero = -concentration_faces[0] *
                                                 (right_cell-left_cell) / (concentration_faces[1]-concentration_faces[0])
                                                 + left_cell;
                    pos_upper_1 = position_zero;
                  }
                unsigned int opposite_face = (face%2==0)?(face+1):(face-1);
                fe_face_values.reinit(cell,opposite_face);
                fe_face_values.get_function_values (twophase_flow.solution.block(0),
                                                    concentration_faces);
                concentration_faces[0] += 1e-22*channel_width;
                concentration_faces[1] += 1e-22*channel_width;
                if (concentration_faces[0]*concentration_faces[1] <= 0)
                  {
                    const double left_cell = fe_face_values.get_quadrature_points()[0][0];
                    const double right_cell = fe_face_values.get_quadrature_points()[1][0];
                    const double position_zero = -concentration_faces[0] *
                                                 (right_cell-left_cell) / (concentration_faces[1]-concentration_faces[0])
                                                 + left_cell;
                    pos_upper_2 = position_zero;
                    h_upper = std::fabs(cell->face(face)->center()[1]-
                                        cell->face(opposite_face)->center()[1]);
                  }
              }
          }
      }

  std::cout.precision(4);
  std::vector<double> data;
  data.push_back(twophase_flow.get_navier_stokes().time_stepping.now());
  pos_x = -pos_x;

  // Exchange data with MPI: Only one processor should have identified the
  // above values, so simply sum them over all processors and let each
  // processor know about the sum.
  pos_x = Utilities::MPI::sum(pos_x, MPI_COMM_WORLD);
  pos_lower_1 = Utilities::MPI::sum(pos_lower_1, MPI_COMM_WORLD);
  pos_lower_2 = Utilities::MPI::sum(pos_lower_2, MPI_COMM_WORLD);
  pos_upper_1 = Utilities::MPI::sum(pos_upper_1, MPI_COMM_WORLD);
  pos_upper_2 = Utilities::MPI::sum(pos_upper_2, MPI_COMM_WORLD);
  h_lower = Utilities::MPI::sum(h_lower, MPI_COMM_WORLD);
  h_upper = Utilities::MPI::sum(h_upper, MPI_COMM_WORLD);
  value_x = Utilities::MPI::sum(value_x, MPI_COMM_WORLD);
  value_y = Utilities::MPI::sum(value_y, MPI_COMM_WORLD);

  const double angle_lower = -std::atan((pos_lower_2-pos_lower_1)/h_lower)*180./numbers::PI+90.;
  const double angle_upper = std::atan((pos_upper_2-pos_upper_1)/h_upper)*180./numbers::PI+90.;
  contact_velocity = (pos_lower_1 - old_contact_position) / twophase_flow.get_navier_stokes().time_stepping.step_size();
  old_contact_position = pos_lower_1;
  pcout << "  Interface location center pointwise:  "
        << pos_x << std::endl;
  pcout << "  Interface location lower wall/angle:  "
        << pos_lower_1 << " / " << angle_lower << std::endl;
  pcout << "  Interface location upper wall/angle:  "
        << pos_upper_1 << " / " << angle_upper << std::endl;
  pcout << "  Interface velocity:                   "
        << contact_velocity << std::endl;
  std::cout.precision(3);

  data.push_back(pos_x);
  data.push_back(value_x);
  data.push_back(value_y);
  data.push_back(pos_lower_1);
  data.push_back(angle_lower);
  data.push_back(pos_upper_1);
  data.push_back(angle_upper);
  if (solution_data.size() &&
      data[0] == solution_data.back()[0])
    solution_data.back().insert (solution_data.back().end(),
                                 data.begin()+1, data.end());
  else
    solution_data.push_back (data);

  pcout << "  Concentration range: "
        << min_concentration << " / "
        << max_concentration << std::endl;


  if (!twophase_flow.get_time_stepping().at_tick(parameters.output_frequency) &&
      twophase_flow.get_time_stepping().step_no() > 1)
    return;

  std::vector<Tensor<2,dim> > interface_points;
  twophase_flow.compute_bubble_statistics(&interface_points, 1);

  if (interface_points.size() > 0 &&
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ostringstream filename3;
      filename3 << parameters.output_filename << "-interface-"
                << Utilities::int_to_string(parameters.global_refinements,1) << "-"
                << Utilities::int_to_string(twophase_flow.get_time_stepping().step_no(), 4)
                << ".txt";

      std::ofstream output_positions3 (filename3.str().c_str());
      output_positions3.precision (14);
      output_positions3 << "# time = " << twophase_flow.get_time_stepping().now()
                        << " contact vel = " << contact_velocity << std::endl;
      for (unsigned int j=0; j<interface_points.size(); ++j)
        {
          for (unsigned int c=0; c<2; ++c)
            for (unsigned int d=0; d<dim; ++d)
              output_positions3 << interface_points[j][c][d] << "   ";
          output_positions3 << std::endl;
        }
    }

  if (parameters.print_solution_fields > 0)
    {
      DataOut<dim> data_out;

      const NavierStokes<dim> &navier_stokes = twophase_flow.get_navier_stokes();
      LinearAlgebra::distributed::Vector<double> velocity_relative(navier_stokes.solution.block(0)), velocity_shift;
      velocity_shift.reinit(velocity_relative);
      Vector<double> velocity_local(navier_stokes.get_fe_u().dofs_per_cell);
      for (unsigned int i=0; i<velocity_local.size(); ++i)
        if (navier_stokes.get_fe_u().system_to_component_index(i).first == 0)
          velocity_local(i) = -contact_velocity;
      for (typename DoFHandler<dim>::active_cell_iterator cell=navier_stokes.get_dof_handler_u().begin_active(); cell != navier_stokes.get_dof_handler_u().end(); ++cell)
        if (cell->is_locally_owned())
          {
            cell->set_dof_values(velocity_local, velocity_shift);
          }
      velocity_relative += velocity_shift;
      velocity_relative.update_ghost_values();

      VectorTools::interpolate(navier_stokes.get_dof_handler_u(),
                               SimilarityVelocity<dim> (-contact_velocity,
                                                        parameters.viscosity/(parameters.viscosity+parameters.viscosity_diff),
                                                        numbers::PI*angle_lower/180.,
                                                        old_contact_position, 2.),
                               velocity_shift);

      data_out.add_data_vector (navier_stokes.get_dof_handler_u(),
                                navier_stokes.solution.block(0),
                                std::vector<std::string>(dim, "velocity"),
                                std::vector<DataComponentInterpretation::DataComponentInterpretation>(dim, DataComponentInterpretation::component_is_part_of_vector));
      data_out.add_data_vector (navier_stokes.get_dof_handler_u(),
                                velocity_relative,
                                std::vector<std::string>(dim, "velocity_relative"),
                                std::vector<DataComponentInterpretation::DataComponentInterpretation>(dim, DataComponentInterpretation::component_is_part_of_vector));
      data_out.add_data_vector (navier_stokes.get_dof_handler_u(),
                                velocity_shift,
                                std::vector<std::string>(dim, "similarity_velocity"),
                                std::vector<DataComponentInterpretation::DataComponentInterpretation>(dim, DataComponentInterpretation::component_is_part_of_vector));
      data_out.add_data_vector (navier_stokes.get_dof_handler_p(),
                                navier_stokes.solution.block(1),
                                "pressure");
      data_out.add_data_vector (twophase_flow.get_dof_handler(),
                                twophase_flow.solution.block(0),
                                "concentration");
      data_out.add_data_vector (twophase_flow.get_dof_handler(),
                                twophase_flow.solution.block(1),
                                "chemical_potential");
      data_out.build_patches ();

      twophase_flow.write_data_output(parameters.output_filename, navier_stokes.time_stepping,
                                      parameters.output_frequency, data_out);
    }

  if (solution_data.size() > 0 &&
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      const int time_step = 1.000001e4*twophase_flow.get_time_stepping().step_size();
      std::ostringstream filename3;
      filename3 << parameters.output_filename << "-"
                <<Utilities::int_to_string(parameters.global_refinements,1) << "-"
                << Utilities::int_to_string(time_step, 4)
                << ".txt";

      std::ofstream output_positions3 (filename3.str().c_str(),
                                       twophase_flow.get_time_stepping().step_no()==0 ?
                                       std::ios_base::out :
                                       std::ios_base::out | std::ios_base::app);
      output_positions3.precision (14);
      for (unsigned int i=0; i<solution_data.size(); ++i)
        {
          output_positions3 << " ";
          for (unsigned int j=0; j<solution_data[i].size(); ++j)
            output_positions3 << solution_data[i][j] << "   ";
          output_positions3 << std::endl;
        }
      solution_data.clear();
    }
}



template <int dim>
void ChannelProblem<dim>::run ()
{
  pcout << "Running a " << dim << "D phase field channel flow problem "
        << "using " << twophase_flow.get_time_stepping().name()
        << ", Q"  << twophase_flow.get_navier_stokes().get_fe_u().degree
        << "/Q" << twophase_flow.get_navier_stokes().get_fe_p().degree
        << " elements" << std::endl;

  {
    const double length = 6;
    std::vector<unsigned int> subdivisions (dim, 1);
    subdivisions[0] = std::round(length/(2*channel_width));

    const Point<dim> bottom_left = (dim == 2 ?
                                    Point<dim>(0,-channel_width) :
                                    Point<dim>(0,-channel_width,-channel_width));
    const Point<dim> top_right   = (dim == 2 ?
                                    Point<dim>(length,channel_width) :
                                    Point<dim>(length,channel_width,channel_width));

    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                               subdivisions,
                                               bottom_left,
                                               top_right);

    // no need to check for owned cells here: on level 0 everything is locally
    // owned
    for (typename Triangulation<dim>::active_cell_iterator it=triangulation.begin();
         it != triangulation.end(); ++it)
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        if (it->face(face)->at_boundary())
          {
            if (std::abs(it->face(face)->center()[0]-length)<1e-13)
              it->face(face)->set_boundary_id(1);
            else if (std::abs(it->face(face)->center()[0]-0)<1e-13)
              it->face(face)->set_boundary_id(2);
            else
              it->face(face)->set_boundary_id(0);
          }
  }

  twophase_flow.set_no_slip_boundary(0);

  twophase_flow.set_open_boundary_with_normal_flux(1, std::shared_ptr<Function<dim> > (new Functions::ZeroFunction<dim>()), 1);
  twophase_flow.set_open_boundary_with_normal_flux(2, std::shared_ptr<Function<dim> > (new Functions::ZeroFunction<dim>()), -1);

  twophase_flow.setup_problem(Functions::ZeroFunction<dim>(dim),
                              InitialValuesLS<dim>());

  output_results ();

  // @sect5{Time loop}
  while (twophase_flow.get_time_stepping().at_end() == false)
    {
      twophase_flow.advance_time_step();

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

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      deallog.depth_console (0);

      std::string paramfile;
      if (argc>1)
        paramfile = argv[1];
      else
        paramfile = "phasefield_poiseuille.prm";

      FlowParameters parameters (paramfile);
      Assert(parameters.dimension == 2, ExcNotImplemented());

      ChannelProblem<2> channel(parameters);
      channel.run();
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
