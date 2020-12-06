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


#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <adaflo/level_set_okz_compute_curvature.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/level_set_okz_template_instantations.h>


template <int dim>
template <int ls_degree, typename Number>
void
LevelSetOKZSolverComputeNormal<dim>::local_compute_normal(
  const MatrixFree<dim, Number> &                        data,
  LinearAlgebra::distributed::BlockVector<Number> &      dst,
  const LinearAlgebra::distributed::BlockVector<Number> &src,
  const std::pair<unsigned int, unsigned int> &          cell_range) const
{
  bool do_float = std::is_same<Number, float>::value;
  // The second input argument below refers to which constrains should be used,
  // 4 means constraints_normals
  FEEvaluation<dim, ls_degree, 2 * ls_degree, dim, Number> phi(data,
                                                               do_float ? 0 : 4,
                                                               do_float ? 0 : 2);
  const VectorizedArray<Number>                            min_diameter =
    make_vectorized_array<Number>(this->epsilon_used / this->parameters.epsilon);
  // cast avoids compile errors, but we always use the path without casting
  const VectorizedArray<Number> *cell_diameters =
    do_float ?
      reinterpret_cast<const VectorizedArray<Number> *>(cell_diameters_float.begin()) :
      reinterpret_cast<const VectorizedArray<Number> *>(this->cell_diameters.begin());

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(src);
      phi.evaluate(true, true);
      const VectorizedArray<Number> damping =
        Number(4.) *
        Utilities::fixed_power<2>(
          std::max(min_diameter, cell_diameters[cell] / static_cast<Number>(ls_degree)));
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient(phi.get_gradient(q) * damping, q);
        }
      phi.integrate(true, true);
      phi.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int ls_degree>
void
LevelSetOKZSolverComputeNormal<dim>::local_compute_normal_rhs(
  const MatrixFree<dim, double> &                  data,
  LinearAlgebra::distributed::BlockVector<double> &dst,
  const LinearAlgebra::distributed::Vector<double> &,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // The second input argument below refers to which constrains should be used,
  // 4 means constraints_normals and 2 means constraints (for LS-function)
  FEEvaluation<dim, ls_degree, 2 * ls_degree, dim> normal_values(data, 4, 2);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1>   ls_values(data, 2, 2);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      normal_values.reinit(cell);
      ls_values.reinit(cell);

      ls_values.read_dof_values_plain(this->solution.block(0));
      ls_values.evaluate(false, true, false);

      for (unsigned int q = 0; q < normal_values.n_q_points; ++q)
        normal_values.submit_value(ls_values.get_gradient(q), q);

      normal_values.integrate(true, false);
      normal_values.distribute_local_to_global(dst);
    }
}



template <int dim>
void
LevelSetOKZSolverComputeNormal<dim>::compute_normal_vmult(
  LinearAlgebra::distributed::BlockVector<double> &      dst,
  const LinearAlgebra::distributed::BlockVector<double> &src) const
{
  dst = 0.;
#define OPERATION(c_degree, u_degree)                                                  \
  this->matrix_free.cell_loop(&LevelSetOKZSolverComputeNormal<                         \
                                dim>::template local_compute_normal<c_degree, double>, \
                              this,                                                    \
                              dst,                                                     \
                              src)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  // The number "4" below is so that constraints_normals is used
  for (unsigned int i = 0; i < this->matrix_free.get_constrained_dofs(4).size(); ++i)
    for (unsigned int d = 0; d < dim; ++d)
      dst.block(d).local_element(this->matrix_free.get_constrained_dofs(4)[i]) =
        preconditioner.get_vector().local_element(
          this->matrix_free.get_constrained_dofs(4)[i]) *
        src.block(d).local_element(this->matrix_free.get_constrained_dofs(4)[i]);
}



template <int dim>
void
LevelSetOKZSolverComputeNormal<dim>::compute_normal_vmult(
  LinearAlgebra::distributed::BlockVector<float> &      dst,
  const LinearAlgebra::distributed::BlockVector<float> &src) const
{
  dst = 0.;
#define OPERATION(c_degree, u_degree)                                                 \
  matrix_free_float.cell_loop(&LevelSetOKZSolverComputeNormal<                        \
                                dim>::template local_compute_normal<c_degree, float>, \
                              this,                                                   \
                              dst,                                                    \
                              src)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  // The number "4" below is so that constraints_normals is used
  for (unsigned int i = 0; i < this->matrix_free.get_constrained_dofs(4).size(); ++i)
    for (unsigned int d = 0; d < dim; ++d)
      dst.block(d).local_element(this->matrix_free.get_constrained_dofs(4)[i]) =
        preconditioner.get_vector().local_element(
          this->matrix_free.get_constrained_dofs(4)[i]) *
        src.block(d).local_element(this->matrix_free.get_constrained_dofs(4)[i]);
}



template <int dim>
struct ComputeNormalMatrix
{
  ComputeNormalMatrix(const LevelSetOKZSolverComputeNormal<dim> &problem)
    : problem(problem)
  {}

  template <typename Number>
  void
  vmult(LinearAlgebra::distributed::BlockVector<Number> &      dst,
        const LinearAlgebra::distributed::BlockVector<Number> &src) const
  {
    problem.compute_normal_vmult(dst, src);
  }

  const LevelSetOKZSolverComputeNormal<dim> &problem;
};



template <int dim>
class InverseNormalMatrix
{
public:
  InverseNormalMatrix(
    GrowingVectorMemory<LinearAlgebra::distributed::BlockVector<float>> &mem,
    const ComputeNormalMatrix<dim> &                                     matrix,
    const DiagonalPreconditioner<float> &                                preconditioner)
    : memory(mem)
    , matrix(matrix)
    , preconditioner(preconditioner)
  {}

  void
  vmult(LinearAlgebra::distributed::BlockVector<double> &      dst,
        const LinearAlgebra::distributed::BlockVector<double> &src) const
  {
    LinearAlgebra::distributed::BlockVector<float> *src_f = memory.alloc();
    LinearAlgebra::distributed::BlockVector<float> *dst_f = memory.alloc();

    src_f->reinit(src);
    dst_f->reinit(dst);

    *dst_f = 0;
    *src_f = src;
    ReductionControl                                         control(10000, 1e-30, 1e-1);
    SolverCG<LinearAlgebra::distributed::BlockVector<float>> solver(control, memory);
    try
      {
        solver.solve(matrix, *dst_f, *src_f, preconditioner);
      }
    catch (...)
      {
        std::cout << "Error, normal solver did not converge!" << std::endl;
      }
    dst = *dst_f;

    memory.free(src_f);
    memory.free(dst_f);
  }

private:
  GrowingVectorMemory<LinearAlgebra::distributed::BlockVector<float>> &memory;
  const ComputeNormalMatrix<dim> &                                     matrix;
  const DiagonalPreconditioner<float> &                                preconditioner;
};



// @sect4{LevelSetOKZSolverComputeNormal::compute_normal}
template <int dim>
void
LevelSetOKZSolverComputeNormal<dim>::compute_normal(const bool fast_computation)
{
  // This function computes the normal from a projection of $\nabla C$ onto
  // the space of linear finite elements (with some small damping)

  TimerOutput::Scope timer(*this->timer, "LS compute normal.");

  // compute right hand side
  this->normal_vector_rhs = 0;
#define OPERATION(c_degree, u_degree)                                                  \
  this->matrix_free.cell_loop(                                                         \
    &LevelSetOKZSolverComputeNormal<dim>::template local_compute_normal_rhs<c_degree>, \
    this,                                                                              \
    this->normal_vector_rhs,                                                           \
    this->solution.block(0))

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  if (this->parameters.approximate_projections == true)
    {
      // apply damping to avoid oscillations. this corresponds to one time
      // step of exlicit Euler for a diffusion problem (need to avoid too
      // large diffusions!)
      const unsigned int n_steps = 3;
      for (unsigned int i = 0; i < n_steps; ++i)
        {
          for (unsigned int block = 0; block < dim; ++block)
            {
              this->constraints_normals.distribute(
                this->normal_vector_field.block(block));
              curvatur_operator.compute_curvature_vmult(
                this->normal_vector_rhs.block(block),
                this->normal_vector_field.block(block),
                2);
            }
          preconditioner.vmult(this->normal_vector_rhs, this->normal_vector_rhs);
          this->normal_vector_field.add(-0.05, this->normal_vector_rhs);
        }
    }
  else
    {
      ComputeNormalMatrix<dim> matrix(*this);

      // solve linear system, reduce residual by 3e-7 in standard case and
      // 1e-3 in fast case. Need a quite strict tolerance for the normal
      // computation, otherwise the profile gets very bad when high curvatures
      // appear and the solver fails
      ReductionControl solver_control(4000, 1e-50, fast_computation ? 1e-5 : 1e-7);

      // ... in case we can somehow come up with a better combination of
      // float/double solvers
      // InverseNormalMatrix<dim> inverse(vectors_normal, matrix,
      // preconditioner_float);
      SolverCG<LinearAlgebra::distributed::BlockVector<double>> solver(solver_control);
      // solver.solve (matrix, this->normal_vector_field,
      // this->normal_vector_rhs,
      //              preconditioner);
      solver.solve(*projection_matrix,
                   this->normal_vector_field,
                   this->normal_vector_rhs,
                   *ilu_projection_matrix);
      // this->pcout << "N its normal: " << solver_control.last_step() <<
      // std::endl;
    }


  for (unsigned int d = 0; d < dim; ++d)
    {
      this->constraints_normals.distribute(this->normal_vector_field.block(d));
      this->normal_vector_field.block(d).update_ghost_values();
    }
}

template class LevelSetOKZSolverComputeNormal<2>;
template class LevelSetOKZSolverComputeNormal<3>;
