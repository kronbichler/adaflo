// --------------------------------------------------------------------------
//
// Copyright (C) 2011 - 2016 by the adaflo authors
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
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/vector_tools.h>

#include <adaflo/phase_field.h>

#include <fstream>
#include <iostream>

using namespace dealii;



template <int dim>
template <int ls_degree, int velocity_degree>
void
PhaseFieldSolver<dim>::local_compute_force(
  const MatrixFree<dim, double> &             data,
  LinearAlgebra::distributed::Vector<double> &dst,
  const LinearAlgebra::distributed::Vector<double> &,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  FEEvaluation<dim, ls_degree, velocity_degree + 1, 1>           ls_values(data, 2, 0);
  FEEvaluation<dim, ls_degree, velocity_degree + 1, 1>           curv_values(data, 2, 0);
  FEEvaluation<dim, velocity_degree, velocity_degree + 1, dim>   vel_values(data, 0, 0);
  FEEvaluation<dim, velocity_degree - 1, velocity_degree + 1, 1> pre_values(data, 1, 0);

  typedef VectorizedArray<double> vector_t;
  const bool                      use_variable_parameters =
    this->parameters.density_diff != 0 || this->parameters.viscosity_diff != 0;

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      ls_values.reinit(cell);
      vel_values.reinit(cell);
      curv_values.reinit(cell);
      pre_values.reinit(cell);

      ls_values.read_dof_values(this->solution.block(0));

      // set variable parameters
      if (use_variable_parameters)
        {
          ls_values.evaluate(true, false);
          vector_t *densities = this->navier_stokes.get_matrix().begin_densities(cell);
          vector_t *viscosities =
            this->navier_stokes.get_matrix().begin_viscosities(cell);
          for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
            {
              vector_t heaviside_val = 0.5 * (ls_values.get_value(q) + 1.);
              heaviside_val =
                std::min(make_vectorized_array(1.),
                         std::max(make_vectorized_array(0.), heaviside_val));
              densities[q] =
                this->parameters.density + this->parameters.density_diff * heaviside_val;
              viscosities[q] = this->parameters.viscosity +
                               this->parameters.viscosity_diff * heaviside_val;
            }
        }

      // interpolate ls values onto pressure
      for (unsigned int i = 0; i < pre_values.dofs_per_cell; ++i)
        {
          vector_t projected_value = vector_t();
          for (unsigned int j = 0; j < ls_values.dofs_per_cell; ++j)
            projected_value +=
              interpolation_concentration_pressure(i, j) * ls_values.get_dof_value(j);
          pre_values.submit_dof_value(projected_value, i);
        }

      // evaluate curvature and level set gradient
      curv_values.read_dof_values(this->solution.block(1));
      curv_values.evaluate(true, false);
      pre_values.evaluate(false, true);

      // evaluate surface tension force and gravity force
      for (unsigned int q = 0; q < curv_values.n_q_points; ++q)
        {
          // surface tension
          Tensor<1, dim, vector_t> force =
            curv_values.get_value(q) * pre_values.get_gradient(q);

          // gravity
          vector_t actual_rho =
            use_variable_parameters ?
              *(this->navier_stokes.get_matrix().begin_densities(cell) + q) :
              make_vectorized_array(this->parameters.density);
          force[dim - 1] -= this->parameters.gravity * actual_rho;

          vel_values.submit_value(force, q);
        }
      vel_values.integrate(true, false);

      vel_values.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int ls_degree, int velocity_degree>
void
PhaseFieldSolver<dim>::local_residual(
  const MatrixFree<dim, double> &                        data,
  LinearAlgebra::distributed::BlockVector<double> &      dst,
  const LinearAlgebra::distributed::BlockVector<double> &src,
  const std::pair<unsigned int, unsigned int> &          cell_range) const
{
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1>         c_values(data, 2, 2);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1>         phi_values(data, 3, 2);
  FEEvaluation<dim, velocity_degree, 2 * ls_degree, dim> vel_values(data, 0, 2);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1>         old_c_values(data, 2, 2);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1>         old_old_c_values(data, 2, 2);

  const double inv_time_weight = 1. / this->time_stepping.weight();
  const double factor_mobility = inv_time_weight * this->parameters.diffusion_length *
                                 this->parameters.diffusion_length;
  const double factor_2 = (1.5 * this->parameters.surface_tension / this->epsilon_used);
  const double factor_4 = (0.75 * this->parameters.surface_tension * this->epsilon_used);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      Tensor<1, dim, VectorizedArray<double>> *velocities =
        &evaluated_convection[cell * c_values.n_q_points];
      VectorizedArray<double> *w_values = &evaluated_phi[cell * c_values.n_q_points];

      c_values.reinit(cell);
      phi_values.reinit(cell);

      vel_values.reinit(cell);
      old_c_values.reinit(cell);
      old_old_c_values.reinit(cell);

      phi_values.read_dof_values_plain(src.block(1));
      c_values.read_dof_values_plain(src.block(0));
      old_c_values.read_dof_values_plain(this->solution_old.block(0));
      old_old_c_values.read_dof_values_plain(this->solution_old_old.block(0));
      vel_values.read_dof_values_plain(*velocity_vector);

      vel_values.evaluate(true, false);
      old_old_c_values.evaluate(true, false);
      old_c_values.evaluate(true, false);

      c_values.evaluate(true, true);
      phi_values.evaluate(true, true);

      for (unsigned int q = 0; q < c_values.n_q_points; ++q)
        {
          const VectorizedArray<double>                 c_val  = c_values.get_value(q);
          const Tensor<1, dim, VectorizedArray<double>> c_grad = c_values.get_gradient(q);
          const VectorizedArray<double>                 w_val  = phi_values.get_value(q);
          const Tensor<1, dim, VectorizedArray<double>> w_grad =
            phi_values.get_gradient(q);
          Tensor<1, dim, VectorizedArray<double>> velocity = vel_values.get_value(q);
          VectorizedArray<double>                 val      = c_val;

          velocities[q] = velocity;
          w_values[q]   = c_val;
          val += this->time_stepping.weight_old() * inv_time_weight *
                 old_c_values.get_value(q);
          if (this->time_stepping.scheme() == TimeStepping::bdf_2 &&
              this->time_stepping.step_no() > 1)
            val += this->time_stepping.weight_old_old() * inv_time_weight *
                   old_old_c_values.get_value(q);

          val += (c_grad * velocity) * inv_time_weight;
          c_values.submit_value(val, q);
          c_values.submit_gradient(make_vectorized_array(factor_mobility) * w_grad, q);

          phi_values.submit_value(w_val - factor_2 * c_val * (c_val * c_val - 1.), q);
          phi_values.submit_gradient(make_vectorized_array(-factor_4) * c_grad, q);
        }

      c_values.integrate(true, true);
      c_values.distribute_local_to_global(dst.block(0));
      phi_values.integrate(true, true);
      phi_values.distribute_local_to_global(dst.block(1));
    }
}



template <int dim>
template <int ls_degree>
void
PhaseFieldSolver<dim>::local_vmult(
  const MatrixFree<dim, double> &                        data,
  LinearAlgebra::distributed::BlockVector<double> &      dst,
  const LinearAlgebra::distributed::BlockVector<double> &src,
  const std::pair<unsigned int, unsigned int> &          cell_range) const
{
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1> c_values(data, 2, 2);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1> phi_values(data, 3, 2);

  const double inv_time_weight = 1. / this->time_stepping.weight();
  const double factor_mobility = inv_time_weight * this->parameters.diffusion_length *
                                 this->parameters.diffusion_length;
  const double factor_2 = (1.5 * this->parameters.surface_tension / this->epsilon_used);
  const double factor_4 = (0.75 * this->parameters.surface_tension * this->epsilon_used);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      Tensor<1, dim, VectorizedArray<double>> *velocities =
        &evaluated_convection[cell * c_values.n_q_points];
      VectorizedArray<double> *w_values = &evaluated_phi[cell * c_values.n_q_points];

      c_values.reinit(cell);
      phi_values.reinit(cell);

      phi_values.read_dof_values(src.block(1));
      c_values.read_dof_values(src.block(0));
      c_values.evaluate(true, true);
      phi_values.evaluate(true, true);

      for (unsigned int q = 0; q < c_values.n_q_points; ++q)
        {
          const VectorizedArray<double>                 c_val  = c_values.get_value(q);
          const Tensor<1, dim, VectorizedArray<double>> c_grad = c_values.get_gradient(q);
          const VectorizedArray<double>                 w_val  = phi_values.get_value(q);
          const Tensor<1, dim, VectorizedArray<double>> w_grad =
            phi_values.get_gradient(q);
          VectorizedArray<double> val = c_val;
          val += (c_grad * velocities[q]) * inv_time_weight;
          c_values.submit_value(val, q);
          c_values.submit_gradient(make_vectorized_array(factor_mobility) * w_grad, q);

          phi_values.submit_value(w_val - factor_2 * c_val *
                                            (3. * w_values[q] * w_values[q] - 1.),
                                  q);
          phi_values.submit_gradient(make_vectorized_array(-factor_4) * c_grad, q);
        }

      c_values.integrate(true, true);
      c_values.distribute_local_to_global(dst.block(0));
      phi_values.integrate(true, true);
      phi_values.distribute_local_to_global(dst.block(1));
    }
}



template <int dim>
template <int ls_degree>
void
PhaseFieldSolver<dim>::local_mass(
  const MatrixFree<dim, double> &                   data,
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1> c_values(data, 2, 2);
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      c_values.reinit(cell);
      c_values.read_dof_values(src);
      c_values.evaluate(true, false);
      for (unsigned int q = 0; q < c_values.n_q_points; ++q)
        c_values.submit_value(c_values.get_value(q), q);
      c_values.integrate(true, false);
      c_values.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int operation>
void
PhaseFieldSolver<dim>::apply_contact_bc(
  LinearAlgebra::distributed::BlockVector<double> &      dst,
  const LinearAlgebra::distributed::BlockVector<double> &src) const
{
  if (this->parameters.contact_angle == 0.)
    return;

  const bool have_ghost_values = src.has_ghost_elements();
  if (!have_ghost_values)
    src.update_ghost_values();
  const unsigned int n_faces = face_indices.size() / this->fe->dofs_per_face;
  Assert(face_indices.size() == this->fe->dofs_per_face * n_faces, ExcInternalError());
  Vector<double>      local_dofs(this->fe->dofs_per_face);
  std::vector<double> evaluated;
  const unsigned int *indices  = &face_indices[0];
  const double *      JxW      = &face_JxW[0];
  double *            c_values = &face_evaluated_c[0];
  for (unsigned int f = 0; f < n_faces; ++f)
    {
      const unsigned int n_face_q_points = face_JxW.size() / n_faces;
      evaluated.resize(n_face_q_points);
      this->constraints.get_dof_values(src.block(0),
                                       indices,
                                       local_dofs.begin(),
                                       local_dofs.end());
      for (unsigned int q = 0; q < n_face_q_points; ++q)
        {
          double sum = 0;
          for (unsigned int i = 0; i < this->fe->dofs_per_face; ++i)
            sum += local_dofs[i] * face_matrix(i, q);
          if (operation == 1)
            {
              c_values[q]  = sum;
              evaluated[q] = this->parameters.contact_angle *
                             this->parameters.surface_tension * 0.75 * (sum * sum - 1.) *
                             JxW[q];
            }
          else
            evaluated[q] = this->parameters.contact_angle *
                           this->parameters.surface_tension * 1.5 * sum * c_values[q] *
                           JxW[q];
        }
      for (unsigned int i = 0; i < this->fe->dofs_per_face; ++i)
        {
          double sum = 0;
          for (unsigned int q = 0; q < n_face_q_points; ++q)
            sum += face_matrix(i, q) * evaluated[q];
          local_dofs[i] = sum;
        }
      this->constraints_curvature.distribute_local_to_global(local_dofs.begin(),
                                                             local_dofs.end(),
                                                             indices,
                                                             dst.block(1));
      indices += this->fe->dofs_per_face;
      c_values += n_face_q_points;
      JxW += n_face_q_points;
    }
  if (!have_ghost_values)
    const_cast<LinearAlgebra::distributed::BlockVector<double> &>(src).zero_out_ghosts();
}



#define EXPAND_OPERATIONS(OPERATION)                                                      \
  const unsigned int degree_u  = this->navier_stokes.get_dof_handler_u().get_fe().degree; \
  const unsigned int ls_degree = this->parameters.concentration_subdivisions;             \
                                                                                          \
  AssertThrow(degree_u >= 2 && degree_u <= 5, ExcNotImplemented());                       \
  AssertThrow(ls_degree >= 1 && ls_degree <= 4, ExcNotImplemented());                     \
  if (ls_degree == 1)                                                                     \
    {                                                                                     \
      if (degree_u == 2)                                                                  \
        OPERATION(1, 2);                                                                  \
      else if (degree_u == 3)                                                             \
        OPERATION(1, 3);                                                                  \
      else if (degree_u == 4)                                                             \
        OPERATION(1, 4);                                                                  \
      else if (degree_u == 5)                                                             \
        OPERATION(1, 5);                                                                  \
    }                                                                                     \
  else if (ls_degree == 2)                                                                \
    {                                                                                     \
      if (degree_u == 2)                                                                  \
        OPERATION(2, 2);                                                                  \
      else if (degree_u == 3)                                                             \
        OPERATION(2, 3);                                                                  \
      else if (degree_u == 4)                                                             \
        OPERATION(2, 4);                                                                  \
      else if (degree_u == 5)                                                             \
        OPERATION(2, 5);                                                                  \
    }                                                                                     \
  else if (ls_degree == 3)                                                                \
    {                                                                                     \
      if (degree_u == 2)                                                                  \
        OPERATION(3, 2);                                                                  \
      else if (degree_u == 3)                                                             \
        OPERATION(3, 3);                                                                  \
      else if (degree_u == 4)                                                             \
        OPERATION(3, 4);                                                                  \
      else if (degree_u == 5)                                                             \
        OPERATION(3, 5);                                                                  \
    }                                                                                     \
  else if (ls_degree == 4)                                                                \
    {                                                                                     \
      if (degree_u == 2)                                                                  \
        OPERATION(4, 2);                                                                  \
      else if (degree_u == 3)                                                             \
        OPERATION(4, 3);                                                                  \
      else if (degree_u == 4)                                                             \
        OPERATION(4, 4);                                                                  \
      else if (degree_u == 5)                                                             \
        OPERATION(4, 5);                                                                  \
    }



template <int dim>
void
PhaseFieldSolver<dim>::compute_force()
{
  this->timer->enter_subsection("Compute force.");

  this->navier_stokes.user_rhs = 0;
#define OPERATION(ls_degree, vel_degree)                                         \
  this->matrix_free.cell_loop(                                                   \
    &PhaseFieldSolver<dim>::template local_compute_force<ls_degree, vel_degree>, \
    this,                                                                        \
    this->navier_stokes.user_rhs.block(0),                                       \
    this->solution.block(0))

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  this->timer->leave_subsection();
}



template <int dim>
double
PhaseFieldSolver<dim>::compute_residual()
{
  this->timer->enter_subsection("Cahn-Hilliard residual.");
  this->system_rhs = 0;
#define OPERATION(c_degree, u_degree)                                    \
  this->matrix_free.cell_loop(                                           \
    &PhaseFieldSolver<dim>::template local_residual<c_degree, u_degree>, \
    this,                                                                \
    this->system_rhs,                                                    \
    this->solution)

  EXPAND_OPERATIONS(OPERATION);

#undef OPERATION

  apply_contact_bc<1>(this->system_rhs, this->solution);
  this->system_rhs.compress(VectorOperation::add);
  const double residual_norm = this->system_rhs.l2_norm();
  this->timer->leave_subsection();
  return residual_norm;
}



template <int dim>
void
PhaseFieldSolver<dim>::vmult(
  LinearAlgebra::distributed::BlockVector<double> &      dst,
  const LinearAlgebra::distributed::BlockVector<double> &src) const
{
  const unsigned int ls_degree = this->fe->degree;

  dst = 0.;

#define OPERATION(c_degree)                                                           \
  this->matrix_free.cell_loop(&PhaseFieldSolver<dim>::template local_vmult<c_degree>, \
                              this,                                                   \
                              dst,                                                    \
                              src)

  if (ls_degree == 1)
    OPERATION(1);
  else if (ls_degree == 2)
    OPERATION(2);
  else if (ls_degree == 3)
    OPERATION(3);
  else if (ls_degree == 4)
    OPERATION(4);

  apply_contact_bc<0>(dst, src);
#undef OPERATION

  dst.compress(VectorOperation::add);

  for (unsigned int i = 0; i < this->matrix_free.get_constrained_dofs(2).size(); ++i)
    dst.block(0).local_element(this->matrix_free.get_constrained_dofs(2)[i]) =
      src.block(0).local_element(this->matrix_free.get_constrained_dofs(2)[i]);
  for (unsigned int i = 0; i < this->matrix_free.get_constrained_dofs(3).size(); ++i)
    dst.block(1).local_element(this->matrix_free.get_constrained_dofs(3)[i]) =
      src.block(1).local_element(this->matrix_free.get_constrained_dofs(3)[i]);
}



template <int dim>
void
PhaseFieldSolver<dim>::mass_vmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src) const
{
  dst = 0.;

  const unsigned int ls_degree = this->fe->degree;
  AssertThrow(ls_degree >= 1 && ls_degree <= 4, ExcNotImplemented());

#define OPERATION(c_degree)                                                          \
  this->matrix_free.cell_loop(&PhaseFieldSolver<dim>::template local_mass<c_degree>, \
                              this,                                                  \
                              dst,                                                   \
                              src)

  if (ls_degree == 1)
    OPERATION(1);
  else if (ls_degree == 2)
    OPERATION(2);
  else if (ls_degree == 3)
    OPERATION(3);
  else if (ls_degree == 4)
    OPERATION(4);

#undef OPERATION

  for (unsigned int i = 0; i < this->matrix_free.get_constrained_dofs(2).size(); ++i)
    dst.local_element(this->matrix_free.get_constrained_dofs(2)[i]) =
      src.local_element(this->matrix_free.get_constrained_dofs(2)[i]);
}



// explicit instantiations

template class PhaseFieldSolver<2>;
template class PhaseFieldSolver<3>;
