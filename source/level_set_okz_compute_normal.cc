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

#include <adaflo/level_set_okz_compute_normal.h>


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
adaflo::LevelSetOKZSolverComputeNormal<dim>::LevelSetOKZSolverComputeNormal(
  BlockVectorType                               &normal_vector_field,
  BlockVectorType                               &normal_vector_rhs,
  const VectorType                              &level_set_field,
  const AlignedVector<VectorizedArray<double>>  &cell_diameters,
  const double                                  &epsilon_used,
  const double                                  &minimal_edge_length,
  const AffineConstraints<double>               &constraints_normals,
  const LevelSetOKZSolverComputeNormalParameter &parameters,
  const MatrixFree<dim>                         &matrix_free,
  const DiagonalPreconditioner<double>          &preconditioner,
  const std::shared_ptr<BlockMatrixExtension>   &projection_matrix,
  const std::shared_ptr<BlockILUExtension>      &ilu_projection_matrix)
  : parameters(parameters)
  , normal_vector_field(normal_vector_field)
  , normal_vector_rhs(normal_vector_rhs)
  , level_set_solution(level_set_field)
  , matrix_free(matrix_free)
  , constraints_normals(constraints_normals)
  , cell_diameters(cell_diameters)
  , epsilon_used(epsilon_used)
  , minimal_edge_length(minimal_edge_length)
  , preconditioner(preconditioner)
  , projection_matrix(projection_matrix)
  , ilu_projection_matrix(ilu_projection_matrix)
{}


template <int dim>
template <int ls_degree, typename Number>
void
adaflo::LevelSetOKZSolverComputeNormal<dim>::local_compute_normal(
  const MatrixFree<dim, Number>                         &data,
  LinearAlgebra::distributed::BlockVector<Number>       &dst,
  const LinearAlgebra::distributed::BlockVector<Number> &src,
  const std::pair<unsigned int, unsigned int>           &cell_range) const
{
  // The second input argument below refers to which constrains should be used,
  // 4 means constraints_normals
  const unsigned int n_q_points = ls_degree == -1 ? 0 : 2 * ls_degree;
  FEEvaluation<dim, ls_degree, n_q_points, dim, Number> phi(data,
                                                            parameters.dof_index_normal,
                                                            parameters.quad_index);
  const VectorizedArray<Number>                         min_diameter =
    make_vectorized_array<Number>(this->epsilon_used / this->parameters.epsilon);
  // cast avoids compile errors, but we always use the path without casting
  const VectorizedArray<Number> *cell_diameters = this->cell_diameters.begin();

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(src);
      phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      const VectorizedArray<Number> damping =
        Number(parameters.damping_scale_factor) *
        Utilities::fixed_power<2>(
          std::max(min_diameter, cell_diameters[cell] / static_cast<Number>(ls_degree)));
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient(phi.get_gradient(q) * damping, q);
        }
      phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int ls_degree>
void
adaflo::LevelSetOKZSolverComputeNormal<dim>::local_compute_normal_rhs(
  const MatrixFree<dim, double> &data,
  BlockVectorType               &dst,
  const VectorType &,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // The second input argument below refers to which constrains should be used,
  // 4 means constraints_normals and 2 means constraints (for LS-function)
  const unsigned int n_q_points = ls_degree == -1 ? 0 : 2 * ls_degree;
  FEEvaluation<dim, ls_degree, n_q_points, dim> normal_values(data,
                                                              parameters.dof_index_normal,
                                                              parameters.quad_index);
  FEEvaluation<dim, ls_degree, n_q_points, 1>   ls_values(data,
                                                        parameters.dof_index_ls,
                                                        parameters.quad_index);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      normal_values.reinit(cell);
      ls_values.reinit(cell);

      ls_values.read_dof_values_plain(this->level_set_solution);
      ls_values.evaluate(EvaluationFlags::gradients);

      for (unsigned int q = 0; q < normal_values.n_q_points; ++q)
        normal_values.submit_value(ls_values.get_gradient(q), q);

      normal_values.integrate(EvaluationFlags::values);
      normal_values.distribute_local_to_global(dst);
    }
}



template <int dim>
void
adaflo::LevelSetOKZSolverComputeNormal<dim>::compute_normal_vmult(
  BlockVectorType       &dst,
  const BlockVectorType &src) const
{
  dst = 0.;
#define OPERATION(c_degree, dummy)                                                     \
  this->matrix_free.cell_loop(&LevelSetOKZSolverComputeNormal<                         \
                                dim>::template local_compute_normal<c_degree, double>, \
                              this,                                                    \
                              dst,                                                     \
                              src)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  for (const unsigned int entry :
       this->matrix_free.get_constrained_dofs(parameters.dof_index_normal))
    for (unsigned int d = 0; d < dim; ++d)
      dst.block(d).local_element(entry) =
        preconditioner.get_vector().local_element(entry) *
        src.block(d).local_element(entry);
}



template <int dim>
struct ComputeNormalMatrix
{
  ComputeNormalMatrix(const adaflo::LevelSetOKZSolverComputeNormal<dim> &problem)
    : problem(problem)
  {}

  template <typename Number>
  void
  vmult(LinearAlgebra::distributed::BlockVector<Number>       &dst,
        const LinearAlgebra::distributed::BlockVector<Number> &src) const
  {
    problem.compute_normal_vmult(dst, src);
  }

  const adaflo::LevelSetOKZSolverComputeNormal<dim> &problem;
};



// @sect4{LevelSetOKZSolverComputeNormal::compute_normal}
template <int dim>
void
adaflo::LevelSetOKZSolverComputeNormal<dim>::compute_normal(const bool fast_computation)
{
  // This function computes the normal from a projection of $\nabla C$ onto
  // the space of linear finite elements (with some small damping)

  // compute right hand side
  this->normal_vector_rhs = 0;
#define OPERATION(c_degree, u_degree)                                                  \
  this->matrix_free.cell_loop(                                                         \
    &LevelSetOKZSolverComputeNormal<dim>::template local_compute_normal_rhs<c_degree>, \
    this,                                                                              \
    this->normal_vector_rhs,                                                           \
    this->level_set_solution)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  if (this->parameters.approximate_projections == true)
    {
      AssertThrow(false, ExcNotImplemented());
      // [PM]: The following code has been removed since it was not used in any test
      // and it introduced annoying cyclic dependencies.

      /*
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
       */
    }
  else
    {
      ComputeNormalMatrix<dim> matrix(*this);

      // solve linear system, reduce residual by 3e-7 in standard case and
      // 1e-3 in fast case. Need a quite strict tolerance for the normal
      // computation, otherwise the profile gets very bad when high curvatures
      // appear and the solver fails
      ReductionControl solver_control(4000, 1e-50, fast_computation ? 1e-5 : 1e-7);

      SolverCG<BlockVectorType> solver(solver_control);
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

template class adaflo::LevelSetOKZSolverComputeNormal<1>;
template class adaflo::LevelSetOKZSolverComputeNormal<2>;
template class adaflo::LevelSetOKZSolverComputeNormal<3>;
