// --------------------------------------------------------------------------
//
// Copyright (C) 2020 by the adaflo authors
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

#include <deal.II/base/timer.h>

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/vector_tools.h>

#include <adaflo/level_set_okz_advance_concentration.h>

namespace
{
  /**
   * Compute maximal velocity for a given vector and the corresponding
   * dof-handler object.
   */
  template <int dim, typename VectorType>
  double
  get_maximal_velocity(const DoFHandler<dim> &dof_handler,
                       const VectorType &     solution,
                       const Quadrature<dim> &quad_in)
  {
    // [PM] We use QIterated in the case of hex mesh for backwards compatibility.
    const Quadrature<dim> quadrature_formula =
      dof_handler.get_fe().reference_cell() == ReferenceCells::get_hypercube<dim>() ?
        Quadrature<dim>(
          QIterated<dim>(QTrapez<1>(), dof_handler.get_fe().tensor_degree() + 1)) :
        quad_in;

    FEValues<dim> fe_values(dof_handler.get_fe(), quadrature_formula, update_values);
    std::vector<Tensor<1, dim>> velocity_values(quadrature_formula.size());

    const FEValuesExtractors::Vector velocities(0);

    double max_velocity = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[velocities].get_function_values(solution, velocity_values);

          for (const auto q : fe_values.quadrature_point_indices())
            max_velocity = std::max(max_velocity, velocity_values[q].norm());
        }

    return Utilities::MPI::max(max_velocity, get_communicator(dof_handler));
  }

  template <int dim, int spacedim>
  double
  diameter_on_coarse_grid(const Triangulation<dim, spacedim> &tria)
  {
    const std::vector<Point<spacedim>> &vertices = tria.get_vertices();
    std::vector<bool>                   boundary_vertices(vertices.size(), false);

    for (const auto &cell : tria.cell_iterators_on_level(0))
      for (const unsigned int face : cell->face_indices())
        if (cell->face(face)->at_boundary())
          for (unsigned int i = 0; i < cell->face(face)->n_vertices(); ++i)
            boundary_vertices[cell->face(face)->vertex_index(i)] = true;

    // now traverse the list of boundary vertices and check distances.
    // since distances are symmetric, we only have to check one half
    double                            max_distance_sqr = 0;
    std::vector<bool>::const_iterator pi               = boundary_vertices.begin();
    const unsigned int                N                = boundary_vertices.size();
    for (unsigned int i = 0; i < N; ++i, ++pi)
      {
        std::vector<bool>::const_iterator pj = pi + 1;
        for (unsigned int j = i + 1; j < N; ++j, ++pj)
          if ((*pi == true) && (*pj == true) &&
              ((vertices[i] - vertices[j]).norm_square() > max_distance_sqr))
            max_distance_sqr = (vertices[i] - vertices[j]).norm_square();
      }

    return std::sqrt(max_distance_sqr);
  }
} // namespace



#define EXPAND_OPERATIONS(OPERATION)                                      \
  if (this->matrix_free.get_dof_handler(parameters.dof_index_vel)         \
        .get_fe()                                                         \
        .reference_cell() != ReferenceCells::get_hypercube<dim>())        \
    {                                                                     \
      OPERATION(-1, -1);                                                  \
    }                                                                     \
  else                                                                    \
    {                                                                     \
      const unsigned int degree_u =                                       \
        this->matrix_free.get_dof_handler(parameters.dof_index_vel)       \
          .get_fe()                                                       \
          .tensor_degree();                                               \
      const unsigned int ls_degree =                                      \
        this->matrix_free.get_dof_handler(parameters.dof_index_ls)        \
          .get_fe()                                                       \
          .tensor_degree();                                               \
                                                                          \
      AssertThrow(degree_u >= 1 && degree_u <= 5, ExcNotImplemented());   \
      AssertThrow(ls_degree >= 1 && ls_degree <= 4, ExcNotImplemented()); \
      if (ls_degree == 1)                                                 \
        {                                                                 \
          if (degree_u == 1)                                              \
            OPERATION(1, 1);                                              \
          else if (degree_u == 2)                                         \
            OPERATION(1, 2);                                              \
          else if (degree_u == 3)                                         \
            OPERATION(1, 3);                                              \
          else if (degree_u == 4)                                         \
            OPERATION(1, 4);                                              \
          else if (degree_u == 5)                                         \
            OPERATION(1, 5);                                              \
        }                                                                 \
      else if (ls_degree == 2)                                            \
        {                                                                 \
          if (degree_u == 1)                                              \
            OPERATION(2, 1);                                              \
          else if (degree_u == 2)                                         \
            OPERATION(2, 2);                                              \
          else if (degree_u == 3)                                         \
            OPERATION(2, 3);                                              \
          else if (degree_u == 4)                                         \
            OPERATION(2, 4);                                              \
          else if (degree_u == 5)                                         \
            OPERATION(2, 5);                                              \
        }                                                                 \
      else if (ls_degree == 3)                                            \
        {                                                                 \
          if (degree_u == 1)                                              \
            OPERATION(3, 1);                                              \
          else if (degree_u == 2)                                         \
            OPERATION(3, 2);                                              \
          else if (degree_u == 3)                                         \
            OPERATION(3, 3);                                              \
          else if (degree_u == 4)                                         \
            OPERATION(3, 4);                                              \
          else if (degree_u == 5)                                         \
            OPERATION(3, 5);                                              \
        }                                                                 \
      else if (ls_degree == 4)                                            \
        {                                                                 \
          if (degree_u == 1)                                              \
            OPERATION(4, 1);                                              \
          else if (degree_u == 2)                                         \
            OPERATION(4, 2);                                              \
          else if (degree_u == 3)                                         \
            OPERATION(4, 3);                                              \
          else if (degree_u == 4)                                         \
            OPERATION(4, 4);                                              \
          else if (degree_u == 5)                                         \
            OPERATION(4, 5);                                              \
        }                                                                 \
    }



template <int dim>
LevelSetOKZSolverAdvanceConcentration<dim>::LevelSetOKZSolverAdvanceConcentration(
  VectorType &                                  solution,
  const VectorType &                            solution_old,
  const VectorType &                            solution_old_old,
  VectorType &                                  increment,
  VectorType &                                  rhs,
  const VectorType &                            vel_solution,
  const VectorType &                            vel_solution_old,
  const VectorType &                            vel_solution_old_old,
  const AlignedVector<VectorizedArray<double>> &cell_diameters,
  const AffineConstraints<double> &             constraints,
  const ConditionalOStream &                    pcout,
  const LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor<dim> &boundary,
  const MatrixFree<dim> &                                             matrix_free,
  const LevelSetOKZSolverAdvanceConcentrationParameter &              parameters,
  const DiagonalPreconditioner<double> &                              preconditioner)
  : parameters(parameters)
  , solution(solution)
  , solution_old(solution_old)
  , solution_old_old(solution_old_old)
  , increment(increment)
  , rhs(rhs)
  , vel_solution(vel_solution)
  , vel_solution_old(vel_solution_old)
  , vel_solution_old_old(vel_solution_old_old)
  , matrix_free(matrix_free)
  , constraints(constraints)
  , pcout(pcout)
  , time_stepping(parameters.time)
  , global_omega_diameter(0.0)
  , cell_diameters(cell_diameters)
  , boundary(boundary)
  , preconditioner(preconditioner)
{}



template <int dim>
template <int ls_degree, int velocity_degree>
void
LevelSetOKZSolverAdvanceConcentration<dim>::local_advance_concentration(
  const MatrixFree<dim, double> &              data,
  VectorType &                                 dst,
  const VectorType &                           src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function)
  const unsigned int n_q_points = ls_degree == -1 ? 0 : 2 * ls_degree;

  FEEvaluation<dim, ls_degree, n_q_points, 1> ls_values(data,
                                                        parameters.dof_index_ls,
                                                        parameters.quad_index);
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      const Tensor<1, dim, VectorizedArray<double>> *velocities =
        &evaluated_vel[cell * ls_values.n_q_points];
      ls_values.reinit(cell);

      ls_values.gather_evaluate(src, true, true);

      for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
        {
          const auto ls_val  = ls_values.get_value(q);
          const auto ls_grad = ls_values.get_gradient(q);
          ls_values.submit_value(ls_val * this->time_stepping.weight() +
                                   ls_grad * velocities[q],
                                 q);
          if (this->parameters.convection_stabilization)
            ls_values.submit_gradient(artificial_viscosities[cell] * ls_grad, q);
        }
      ls_values.integrate_scatter(true, this->parameters.convection_stabilization, dst);
    }
}


namespace
{
  template <typename Number, int dim>
  Number
  dot(const Tensor<1, dim, Number> &t1, const Tensor<1, dim, Number> &t2)
  {
    return t1 * t2;
  }

  template <typename Number>
  Number
  dot(const Tensor<1, 1, Number> &t1, const Number &t2)
  {
    return t1[0] * t2;
  }

  template <typename Number>
  Number
  dot(const Number &t1, const Tensor<1, 1, Number> &t2)
  {
    return t1 * t2[0];
  }

} // namespace



template <int dim>
template <int ls_degree, int velocity_degree>
void
LevelSetOKZSolverAdvanceConcentration<dim>::local_advance_concentration_rhs(
  const MatrixFree<dim, double> &data,
  VectorType &                   dst,
  const VectorType &,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function) and 0 means
  // &navier_stokes.get_constraints_u()
  const unsigned int n_q_points = ls_degree == -1 ? 0 : 2 * ls_degree;

  FEEvaluation<dim, ls_degree, n_q_points, 1>         ls_values(data,
                                                        parameters.dof_index_ls,
                                                        parameters.quad_index);
  FEEvaluation<dim, ls_degree, n_q_points, 1>         ls_values_old(data,
                                                            parameters.dof_index_ls,
                                                            parameters.quad_index);
  FEEvaluation<dim, ls_degree, n_q_points, 1>         ls_values_old_old(data,
                                                                parameters.dof_index_ls,
                                                                parameters.quad_index);
  FEEvaluation<dim, velocity_degree, n_q_points, dim> vel_values(data,
                                                                 parameters.dof_index_vel,
                                                                 parameters.quad_index);
  FEEvaluation<dim, velocity_degree, n_q_points, dim> vel_values_old(
    data, parameters.dof_index_vel, parameters.quad_index);
  FEEvaluation<dim, velocity_degree, n_q_points, dim> vel_values_old_old(
    data, parameters.dof_index_vel, parameters.quad_index);

  typedef VectorizedArray<double> vector_t;

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      if (velocity_at_quadrature_points_given == false)
        {
          vel_values.reinit(cell);
          vel_values_old.reinit(cell);
          vel_values_old_old.reinit(cell);
          vel_values.read_dof_values_plain(vel_solution);
          vel_values_old.read_dof_values_plain(vel_solution_old);
          vel_values_old_old.read_dof_values_plain(vel_solution_old_old);
          vel_values.evaluate(true, false);
          vel_values_old.evaluate(true, false);
          vel_values_old_old.evaluate(true, false);
        }

      ls_values.reinit(cell);
      ls_values_old.reinit(cell);
      ls_values_old_old.reinit(cell);
      ls_values.read_dof_values_plain(this->solution);
      ls_values_old.read_dof_values_plain(this->solution_old);
      ls_values_old_old.read_dof_values_plain(this->solution_old_old);
      ls_values.evaluate(true, true);
      ls_values_old.evaluate(true, true);
      ls_values_old_old.evaluate(true, true);

      if (this->parameters.convection_stabilization)
        {
          vector_t max_residual = vector_t(), max_velocity = vector_t();
          for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
            {
              // compute residual of concentration equation
              Tensor<1, dim, VectorizedArray<double>> u;

              if (velocity_at_quadrature_points_given)
                u = evaluated_vel_old[cell * ls_values.n_q_points + q] +
                    evaluated_vel_old_old[cell * ls_values.n_q_points + q];
              else
                u = vel_values_old.get_value(q) + vel_values_old_old.get_value(q);

              vector_t dc_dt =
                (ls_values_old.get_value(q) - ls_values_old_old.get_value(q)) /
                this->time_stepping.old_step_size();
              vector_t residual = std::abs(
                dc_dt +
                dot(u,
                    (ls_values_old.get_gradient(q) + ls_values_old_old.get_gradient(q))) *
                  0.25);
              max_residual = std::max(residual, max_residual);
              max_velocity = std::max(std::sqrt(u * u), max_velocity);
            }
          double global_scaling = global_max_velocity * 2 * this->global_omega_diameter;
          const vector_t cell_diameter = this->cell_diameters[cell];

          artificial_viscosities[cell] =
            0.03 * max_velocity * cell_diameter *
            std::min(make_vectorized_array(1.), 1. * max_residual / global_scaling);
        }

      Tensor<1, dim, VectorizedArray<double>> *velocities =
        &evaluated_vel[cell * ls_values.n_q_points];

      for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
        {
          if (velocity_at_quadrature_points_given == false)
            velocities[q] = vel_values.get_value(q);

          // compute right hand side
          auto old_value = this->time_stepping.weight_old() * ls_values_old.get_value(q);
          if (this->time_stepping.scheme() == TimeSteppingParameters::Scheme::bdf_2 &&
              this->time_stepping.step_no() > 1)
            old_value +=
              this->time_stepping.weight_old_old() * ls_values_old_old.get_value(q);
          const auto ls_val   = ls_values.get_value(q);
          const auto ls_grad  = ls_values.get_gradient(q);
          const auto residual = -(ls_val * this->time_stepping.weight() +
                                  dot(velocities[q], ls_grad) + old_value);
          ls_values.submit_value(residual, q);
          if (this->parameters.convection_stabilization)
            ls_values.submit_gradient(-artificial_viscosities[cell] * ls_grad, q);
        }
      ls_values.integrate_scatter(true, this->parameters.convection_stabilization, dst);
    }
}



template <int dim>
void
LevelSetOKZSolverAdvanceConcentration<dim>::advance_concentration_vmult(
  VectorType &      dst,
  const VectorType &src) const
{
  dst = 0.;
#define OPERATION(c_degree, u_degree)                                 \
  this->matrix_free.cell_loop(                                        \
    &LevelSetOKZSolverAdvanceConcentration<                           \
      dim>::template local_advance_concentration<c_degree, u_degree>, \
    this,                                                             \
    dst,                                                              \
    src)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION


  if (this->parameters.convection_stabilization)
    {
      const auto &dof_handler =
        this->matrix_free.get_dof_handler(parameters.dof_index_ls);
      const auto &fe = dof_handler.get_fe();

      // Boundary part of stabilization-term:
      FEFaceValues<dim> fe_face_values(
        fe,
        this->matrix_free.get_face_quadrature(parameters.quad_index),
        update_values | update_gradients | update_JxW_values | update_normal_vectors);
      Vector<double>                       cell_rhs(fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
      std::vector<Tensor<1, dim>> local_gradients(fe_face_values.get_quadrature().size());
      src.update_ghost_values();

      for (unsigned int mcell = 0; mcell < this->matrix_free.n_cell_batches(); ++mcell)
        for (unsigned int v = 0;
             v < this->matrix_free.n_active_entries_per_cell_batch(mcell);
             ++v)
          {
            typename DoFHandler<dim>::active_cell_iterator cell =
              this->matrix_free.get_cell_iterator(mcell,
                                                  v,
                                                  this->parameters.dof_index_ls);
            cell_rhs = 0;

            for (const auto &face : cell->face_iterators())
              {
                if (face->at_boundary() == false)
                  continue;

                if (this->boundary.symmetry.find(face->boundary_id()) !=
                    this->boundary.symmetry.end())
                  continue;

                fe_face_values.reinit(cell, face);
                fe_face_values.get_function_gradients(src, local_gradients);
                for (const auto i : fe_face_values.dof_indices())
                  for (const auto q : fe_face_values.quadrature_point_indices())
                    cell_rhs(i) +=
                      -((fe_face_values.shape_value(i, q) *
                         fe_face_values.normal_vector(q) *
                         artificial_viscosities[mcell][v] * local_gradients[q]) *
                        fe_face_values.JxW(q));
              }

            cell->get_dof_indices(local_dof_indices);
            this->constraints.distribute_local_to_global(cell_rhs,
                                                         local_dof_indices,
                                                         dst);
          }

      dst.compress(VectorOperation::add);
    }

  for (unsigned int i = 0;
       i < this->matrix_free.get_constrained_dofs(this->parameters.dof_index_ls).size();
       ++i)
    dst.local_element(
      this->matrix_free.get_constrained_dofs(this->parameters.dof_index_ls)[i]) =
      preconditioner.get_vector().local_element(
        this->matrix_free.get_constrained_dofs(this->parameters.dof_index_ls)[i]) *
      src.local_element(
        this->matrix_free.get_constrained_dofs(this->parameters.dof_index_ls)[i]);
}



template <int dim, typename VectorType>
struct AdvanceConcentrationMatrix
{
  AdvanceConcentrationMatrix(const LevelSetOKZSolverAdvanceConcentration<dim> &problem)
    : problem(problem)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    problem.advance_concentration_vmult(dst, src);
  }

  const LevelSetOKZSolverAdvanceConcentration<dim> &problem;
};



// @sect4{LevelSetOKZSolverAdvanceConcentration::advance_concentration}
template <int dim>
void
LevelSetOKZSolverAdvanceConcentration<dim>::advance_concentration(const double dt)
{
  this->time_stepping.set_time_step(dt);
  this->time_stepping.next();

  if (global_omega_diameter == 0.0)
    global_omega_diameter =
      diameter_on_coarse_grid(matrix_free.get_dof_handler().get_triangulation());

  if (evaluated_vel.size() != this->matrix_free.n_cell_batches() *
                                this->matrix_free.get_n_q_points(parameters.quad_index))
    evaluated_vel.resize(this->matrix_free.n_cell_batches() *
                         this->matrix_free.get_n_q_points(parameters.quad_index));

  const auto &mapping     = *this->matrix_free.get_mapping_info().mapping;
  const auto &dof_handler = this->matrix_free.get_dof_handler(parameters.dof_index_ls);
  const auto &fe          = dof_handler.get_fe();

  if (artificial_viscosities.size() != this->matrix_free.n_cell_batches())
    artificial_viscosities.resize(this->matrix_free.n_cell_batches());

  // apply boundary values
  {
    std::map<types::boundary_id, const Function<dim> *> dirichlet;

    for (const auto &b : this->boundary.dirichlet)
      dirichlet[b.first] = b.second.get();

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             dirichlet,
                                             boundary_values);

    for (const auto &it : boundary_values)
      if (this->solution.locally_owned_elements().is_element(it.first))
        this->solution(it.first) = it.second;
    this->solution.update_ghost_values();
  }

  // compute right hand side
  if (velocity_at_quadrature_points_given == false)
    global_max_velocity =
      get_maximal_velocity(matrix_free.get_dof_handler(parameters.dof_index_vel),
                           vel_solution,
                           matrix_free.get_quadrature(parameters.quad_index));
  else
    {
      global_max_velocity = 0;

      for (const auto &i : evaluated_vel)
        {
          const auto ii = i.norm();

          for (const auto &iii : ii)
            global_max_velocity = std::max(iii, global_max_velocity);
        }
    }

  rhs = 0;

#define OPERATION(c_degree, u_degree)                                     \
  this->matrix_free.cell_loop(                                            \
    &LevelSetOKZSolverAdvanceConcentration<                               \
      dim>::template local_advance_concentration_rhs<c_degree, u_degree>, \
    this,                                                                 \
    rhs,                                                                  \
    this->solution)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  AdvanceConcentrationMatrix<dim, VectorType> matrix(*this);



  if (this->parameters.convection_stabilization)
    {
      // Boundary part of stabilization-term:
      FEFaceValues<dim> fe_face_values(
        mapping,
        fe,
        this->matrix_free.get_face_quadrature(parameters.quad_index),
        update_values | update_gradients | update_JxW_values | update_normal_vectors);

      Vector<double>                       cell_rhs(fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
      std::vector<Tensor<1, dim>> local_gradients(fe_face_values.get_quadrature().size());

      for (unsigned int mcell = 0; mcell < this->matrix_free.n_cell_batches(); ++mcell)
        for (unsigned int v = 0;
             v < this->matrix_free.n_active_entries_per_cell_batch(mcell);
             ++v)
          {
            typename DoFHandler<dim>::active_cell_iterator cell =
              this->matrix_free.get_cell_iterator(mcell,
                                                  v,
                                                  this->parameters.dof_index_ls);
            cell_rhs = 0;

            for (const auto face : cell->face_iterators())
              {
                if (face->at_boundary() == false)
                  continue;

                if (this->boundary.symmetry.find(face->boundary_id()) !=
                    this->boundary.symmetry.end())
                  continue;

                fe_face_values.reinit(cell, face);
                fe_face_values.get_function_gradients(this->solution, local_gradients);

                for (const auto i : fe_face_values.dof_indices())
                  for (const auto q : fe_face_values.quadrature_point_indices())
                    cell_rhs(i) +=
                      ((fe_face_values.shape_value(i, q) *
                        fe_face_values.normal_vector(q) *
                        artificial_viscosities[mcell][v] * local_gradients[q]) *
                       fe_face_values.JxW(q));
              }

            cell->get_dof_indices(local_dof_indices);
            this->constraints.distribute_local_to_global(cell_rhs,
                                                         local_dof_indices,
                                                         this->rhs);
          }
      this->rhs.compress(VectorOperation::add);
    }


  // solve linear system with Bicgstab (non-symmetric system!)
  unsigned int n_iterations     = 0;
  double       initial_residual = 0.;
  try
    {
      ReductionControl control(30, 0.05 * this->parameters.tol_nl_iteration, 1e-8);
      SolverBicgstab<VectorType>::AdditionalData bicg_data;
      bicg_data.exact_residual = false;
      SolverBicgstab<VectorType> solver(control, bicg_data);
      increment = 0;
      solver.solve(matrix, increment, rhs, preconditioner);
      n_iterations     = control.last_step();
      initial_residual = control.initial_value();
    }
  catch (const SolverControl::NoConvergence &)
    {
      // GMRES is typically slower but much more robust
      ReductionControl control(3000, 0.05 * this->parameters.tol_nl_iteration, 1e-8);
      SolverGMRES<VectorType> solver(control);
      solver.solve(matrix, increment, rhs, preconditioner);
      n_iterations = 30 + control.last_step();
    }
  if (!this->parameters.do_iteration)
    this->pcout << "  Concentration advance: advect [" << initial_residual << "/"
                << n_iterations << "]";

  this->constraints.distribute(increment);
  this->solution += increment;
  this->solution.update_ghost_values();
}


template class LevelSetOKZSolverAdvanceConcentration<1>;
template class LevelSetOKZSolverAdvanceConcentration<2>;
template class LevelSetOKZSolverAdvanceConcentration<3>;
