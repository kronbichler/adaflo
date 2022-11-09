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

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <adaflo/level_set_okz_reinitialization.h>

#define EXPAND_OPERATIONS(OPERATION)                                      \
  if (this->matrix_free.get_dof_handler(parameters.dof_index_ls)          \
        .get_fe()                                                         \
        .reference_cell() != ReferenceCells::get_hypercube<dim>())        \
    {                                                                     \
      OPERATION(-1, 0);                                                   \
    }                                                                     \
  else                                                                    \
    {                                                                     \
      const unsigned int ls_degree =                                      \
        this->matrix_free.get_dof_handler(parameters.dof_index_ls)        \
          .get_fe()                                                       \
          .tensor_degree();                                               \
                                                                          \
      AssertThrow(ls_degree >= 1 && ls_degree <= 4, ExcNotImplemented()); \
      if (ls_degree == 1)                                                 \
        OPERATION(1, 0);                                                  \
      else if (ls_degree == 2)                                            \
        OPERATION(2, 0);                                                  \
      else if (ls_degree == 3)                                            \
        OPERATION(3, 0);                                                  \
      else if (ls_degree == 4)                                            \
        OPERATION(4, 0);                                                  \
    }

using namespace dealii;

template <int dim>
template <int ls_degree, bool diffuse_only>
void
LevelSetOKZSolverReinitialization<dim>::local_reinitialize(
  const MatrixFree<dim, double> &              data,
  VectorType &                                 dst,
  const VectorType &                           src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  const unsigned int concentration_subdivisions =
    this->matrix_free.get_dof_handler(parameters.dof_index_ls).get_fe().tensor_degree();

  const double dtau_inv = std::max(0.95 / (1. / (dim * dim) * this->minimal_edge_length /
                                           concentration_subdivisions),
                                   1. / (5. * this->time_stepping.step_size()));

  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function)
  const unsigned int n_q_points = ls_degree == -1 ? 0 : 2 * ls_degree;
  FEEvaluation<dim, ls_degree, n_q_points> phi(data,
                                               parameters.dof_index_ls,
                                               parameters.quad_index);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(src);
      phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      VectorizedArray<double> cell_diameter = this->cell_diameters[cell];
      VectorizedArray<double> diffusion =
        std::max(make_vectorized_array(this->epsilon_used),
                 cell_diameter / static_cast<double>(ls_degree));

      const Tensor<1, dim, VectorizedArray<double>> *normal =
        &evaluated_normal[cell * phi.n_q_points];
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        if (!diffuse_only)
          {
            phi.submit_value(dtau_inv * phi.get_value(q), q);
            phi.submit_gradient((diffusion * (normal[q] * phi.get_gradient(q))) *
                                  normal[q],
                                q);
          }
        else
          {
            phi.submit_value(dtau_inv * phi.get_value(q), q);
            phi.submit_gradient(phi.get_gradient(q) * diffusion, q);
          }

      phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi.distribute_local_to_global(dst);
    }
}


namespace
{
  template <typename number>
  VectorizedArray<number>
  normalize(const VectorizedArray<number> &in)
  {
    return std::abs(in);
  }

  template <int dim, typename number>
  static VectorizedArray<number>
  normalize(const Tensor<1, dim, VectorizedArray<number>> &in)
  {
    return in.norm();
  }
} // namespace



template <int dim>
template <int ls_degree, bool diffuse_only>
void
LevelSetOKZSolverReinitialization<dim>::local_reinitialize_rhs(
  const MatrixFree<dim, double> &data,
  VectorType &                   dst,
  const VectorType &,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function) and 4 means constraints_normals
  const unsigned int n_q_points = ls_degree == -1 ? 0 : 2 * ls_degree;

  FEEvaluation<dim, ls_degree, n_q_points>      phi(data,
                                               parameters.dof_index_ls,
                                               parameters.quad_index);
  FEEvaluation<dim, ls_degree, n_q_points, dim> normals(data,
                                                        parameters.dof_index_normal,
                                                        parameters.quad_index);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values_plain(this->solution);
      phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      normals.reinit(cell);
      normals.read_dof_values_plain(this->normal_vector_field);
      normals.evaluate(EvaluationFlags::values);

      VectorizedArray<double> cell_diameter = this->cell_diameters[cell];
      VectorizedArray<double> diffusion =
        std::max(make_vectorized_array(this->epsilon_used),
                 cell_diameter / static_cast<double>(ls_degree));

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        if (!diffuse_only)
          {
            Tensor<1, dim, VectorizedArray<double>> grad = phi.get_gradient(q);
            if (first_reinit_step)
              {
                auto normal = normals.get_value(q);
                normal /= std::max(make_vectorized_array(1e-4), normalize(normal));
                evaluated_normal[cell * phi.n_q_points + q] = normal;
              }
            // take normal as it was for the first reinit step
            Tensor<1, dim, VectorizedArray<double>> normal =
              evaluated_normal[cell * phi.n_q_points + q];
            phi.submit_gradient(normal *
                                  (0.5 * (1. - phi.get_value(q) * phi.get_value(q)) -
                                   (normal * grad * diffusion)),
                                q);
          }
        else
          {
            phi.submit_gradient(-diffusion * phi.get_gradient(q), q);
          }

      phi.integrate(EvaluationFlags::gradients);
      phi.distribute_local_to_global(dst);
    }
}



template <int dim>
void
LevelSetOKZSolverReinitialization<dim>::reinitialization_vmult(
  VectorType &      dst,
  const VectorType &src,
  const bool        diffuse_only) const
{
  dst = 0.;
  if (diffuse_only)
    {
#define OPERATION(c_degree, u_degree)                                              \
  this->matrix_free.cell_loop(&LevelSetOKZSolverReinitialization<                  \
                                dim>::template local_reinitialize<c_degree, true>, \
                              this,                                                \
                              dst,                                                 \
                              src)

      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }
  else
    {
#define OPERATION(c_degree, u_degree)                                               \
  this->matrix_free.cell_loop(&LevelSetOKZSolverReinitialization<                   \
                                dim>::template local_reinitialize<c_degree, false>, \
                              this,                                                 \
                              dst,                                                  \
                              src)

      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }


  for (const unsigned int entry :
       this->matrix_free.get_constrained_dofs(parameters.dof_index_ls))
    dst.local_element(entry) =
      preconditioner.get_vector().local_element(entry) * src.local_element(entry);
}



template <int dim, typename VectorType>
struct ReinitializationMatrix
{
  ReinitializationMatrix(const LevelSetOKZSolverReinitialization<dim> &problem,
                         const bool                                    diffuse_only)
    : problem(problem)
    , diffuse_only(diffuse_only)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    problem.reinitialization_vmult(dst, src, diffuse_only);
  }

  const LevelSetOKZSolverReinitialization<dim> &problem;
  const bool                                    diffuse_only;
};



template <int dim>
void
LevelSetOKZSolverReinitialization<dim>::reinitialize(
  const double                     dt,
  const unsigned int               stab_steps,
  const unsigned int               diff_steps,
  const std::function<void(bool)> &compute_normal)
{
  this->time_stepping.set_desired_time_step(dt);

  // This function assembles and solves for a given profile using the approach
  // described in the paper by Olsson, Kreiss, and Zahedi.

  if (evaluated_normal.size() !=
      this->matrix_free.n_cell_batches() *
        this->matrix_free.get_n_q_points(parameters.quad_index))
    evaluated_normal.resize(this->matrix_free.n_cell_batches() *
                            this->matrix_free.get_n_q_points(parameters.quad_index));

  std::cout.precision(3);

  // perform several reinitialization steps until we reach the maximum number
  // of steps.
  //
  // TODO: make an adaptive choice of the number of iterations
  unsigned actual_diff_steps = diff_steps;
  if (this->last_concentration_range.first < -1.02 ||
      this->last_concentration_range.second > 1.02)
    actual_diff_steps += 3;
  if (!this->parameters.do_iteration)
    this->pcout << (this->time_stepping.now() == this->time_stepping.start() ? "  " :
                                                                               " and ")
                << "reinitialize (";
  for (unsigned int tau = 0; tau < actual_diff_steps + stab_steps; tau++)
    {
      first_reinit_step = (tau == actual_diff_steps);
      if (first_reinit_step)
        compute_normal(true);

      // compute right hand side
      VectorType &rhs       = this->system_rhs;
      VectorType &increment = this->solution_update;
      rhs                   = 0;

      if (tau < actual_diff_steps)
        {
#define OPERATION(c_degree, u_degree)                                                  \
  this->matrix_free.cell_loop(&LevelSetOKZSolverReinitialization<                      \
                                dim>::template local_reinitialize_rhs<c_degree, true>, \
                              this,                                                    \
                              rhs,                                                     \
                              this->solution)

          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
        }
      else
        {
#define OPERATION(c_degree, u_degree)                                                   \
  this->matrix_free.cell_loop(&LevelSetOKZSolverReinitialization<                       \
                                dim>::template local_reinitialize_rhs<c_degree, false>, \
                              this,                                                     \
                              rhs,                                                      \
                              this->solution)

          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
        }

      // solve linear system
      {
        ReinitializationMatrix<dim, VectorType> matrix(*this, tau < actual_diff_steps);
        increment = 0;

        // reduce residual by 1e-6. To obtain good interface shapes, it is
        // essential that this tolerance is relative to the rhs
        // (ReductionControl steered solver, last argument determines the
        // solver)
        ReductionControl     solver_control(2000, 1e-50, 1e-6);
        SolverCG<VectorType> cg(solver_control);
        cg.solve(matrix, increment, rhs, preconditioner);
        this->constraints.distribute(increment);
        if (!this->parameters.do_iteration)
          {
            if (tau < actual_diff_steps)
              this->pcout << "d" << solver_control.last_step();
            else
              this->pcout << solver_control.last_step();
          }
      }

      this->solution += increment;
      this->solution.update_ghost_values();

      // check residual
      const double update_norm = increment.l2_norm();
      if (update_norm < 1e-6)
        break;

      if (!this->parameters.do_iteration && tau < actual_diff_steps + stab_steps - 1)
        this->pcout << " + ";
    }

  if (!this->parameters.do_iteration)
    this->pcout << ")" << std::endl << std::flush;

  this->time_stepping.next();
}

template class LevelSetOKZSolverReinitialization<1>;
template class LevelSetOKZSolverReinitialization<2>;
template class LevelSetOKZSolverReinitialization<3>;
