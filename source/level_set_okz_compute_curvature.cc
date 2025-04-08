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
adaflo::LevelSetOKZSolverComputeCurvature<dim>::LevelSetOKZSolverComputeCurvature(
  const AlignedVector<VectorizedArray<double>>          &cell_diameters,
  const LinearAlgebra::distributed::BlockVector<double> &normal_vector_field,
  const AffineConstraints<double>                       &constraints_curvature,
  const AffineConstraints<double>                       &constraints,
  const double                                          &epsilon_used,
  LinearAlgebra::distributed::Vector<double>            &system_rhs,
  const LevelSetOKZSolverComputeCurvatureParameter      &parameters,
  LinearAlgebra::distributed::Vector<double>            &solution_curvature,
  const LinearAlgebra::distributed::Vector<double>      &solution_ls,
  const MatrixFree<dim>                                 &matrix_free,
  const DiagonalPreconditioner<double>                  &preconditioner,
  std::shared_ptr<BlockMatrixExtension>                 &projection_matrix,
  std::shared_ptr<BlockILUExtension>                    &ilu_projection_matrix)
  : parameters(parameters)
  , solution_curvature(solution_curvature)
  , rhs(system_rhs)
  , solution_ls(solution_ls)
  , normal_vector_field(normal_vector_field)
  , matrix_free(matrix_free)
  , constraints_curvature(constraints_curvature)
  , constraints(constraints)
  , cell_diameters(cell_diameters)
  , epsilon_used(epsilon_used)
  , preconditioner(preconditioner)
  , projection_matrix(projection_matrix)
  , ilu_projection_matrix(ilu_projection_matrix)
{}


template <int dim>
template <int ls_degree, int diffusion_setting>
void
adaflo::LevelSetOKZSolverComputeCurvature<dim>::local_compute_curvature(
  const MatrixFree<dim, double>                    &data,
  LinearAlgebra::distributed::Vector<double>       &dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int>      &cell_range) const
{
  // The second input argument below refers to which constrains should be used,
  // 3 means constraints_curvature
  const unsigned int n_q_points = ls_degree == -1 ? 0 : 2 * ls_degree;
  FEEvaluation<dim, ls_degree, n_q_points, 1> phi(data,
                                                  parameters.dof_index_curvature,
                                                  parameters.quad_index);
  const VectorizedArray<double>               min_diameter =
    make_vectorized_array(this->epsilon_used / this->parameters.epsilon);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(src);
      // If diffusion_setting is true a damping term is added to the weak form
      //  i.e. diffusion_setting=1 => diffusion_setting%2 == 1 is true.
      phi.evaluate(
        ((diffusion_setting < 2) ? EvaluationFlags::values : EvaluationFlags::nothing) |
        ((diffusion_setting % 2 == 1) ? EvaluationFlags::gradients :
                                        EvaluationFlags::nothing));
      const VectorizedArray<double> damping =
        diffusion_setting % 2 == 1 ?
          Utilities::fixed_power<2>(
            std::max(min_diameter,
                     this->cell_diameters[cell] / static_cast<double>(ls_degree))) :
          VectorizedArray<double>();
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          if (diffusion_setting < 2)
            phi.submit_value(phi.get_value(q), q);
          if (diffusion_setting % 2 == 1)
            phi.submit_gradient(phi.get_gradient(q) * damping, q);
        }
      phi.integrate(
        ((diffusion_setting < 2) ? EvaluationFlags::values : EvaluationFlags::nothing) |
        ((diffusion_setting % 2 == 1) ? EvaluationFlags::gradients :
                                        EvaluationFlags::nothing));
      phi.distribute_local_to_global(dst);
    }
}

namespace
{
  template <typename number>
  VectorizedArray<number>
  normalize(const VectorizedArray<number> &in, bool &all_zero)
  {
    VectorizedArray<number> vec;

    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      vec[v] = in[v] >= 1e-2 ? 1.0 : (in[v] <= -1e-2 ? -1.0 : 0.0);

    all_zero = false; // TODO?

    return vec;
  }

  template <int dim, typename number>
  static Tensor<1, dim, VectorizedArray<number>>
  normalize(const Tensor<1, dim, VectorizedArray<number>> &normal, bool &all_zero)
  {
    Tensor<1, dim, VectorizedArray<number>> vec = normal;

    const VectorizedArray<double> normal_norm = normal.norm();
    for (unsigned int d = 0; d < VectorizedArray<double>::size(); ++d)
      if (normal_norm[d] > 1e-2)
        {
          all_zero = false;
          for (unsigned int e = 0; e < dim; ++e)
            vec[e][d] /= normal_norm[d];
        }
      else
        for (unsigned int e = 0; e < dim; ++e)
          vec[e][d] = 0;

    return vec;
  }

  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number,
            typename VectorizedArrayType>
  static VectorizedArray<Number>
  get_divergence(const FEEvaluation<dim,
                                    fe_degree,
                                    n_q_points_1d,
                                    n_components_,
                                    Number,
                                    VectorizedArrayType> &phi,
                 const unsigned int                       q)
  {
    return phi.get_divergence(q);
  }

  template <int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number,
            typename VectorizedArrayType>
  static VectorizedArray<Number>
  get_divergence(const FEEvaluation<1,
                                    fe_degree,
                                    n_q_points_1d,
                                    n_components_,
                                    Number,
                                    VectorizedArrayType> &phi,
                 const unsigned int                       q)
  {
    (void)q;
    (void)phi;
    return 0.0;
  }
} // namespace



template <int dim>
template <int ls_degree>
void
adaflo::LevelSetOKZSolverComputeCurvature<dim>::local_compute_curvature_rhs(
  const MatrixFree<dim, double>              &data,
  LinearAlgebra::distributed::Vector<double> &dst,
  const LinearAlgebra::distributed::Vector<double> &,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // The second input argument below refers to which constrains should be used,
  // 4 means constraints_normals and 3 constraints_curvature
  const unsigned int n_q_points = ls_degree == -1 ? 0 : 2 * ls_degree;
  FEEvaluation<dim, ls_degree, n_q_points, dim> normal_values(data,
                                                              parameters.dof_index_normal,
                                                              parameters.quad_index);
  FEEvaluation<dim, ls_degree, n_q_points, 1>   curv_values(data,
                                                          parameters.dof_index_curvature,
                                                          parameters.quad_index);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      normal_values.reinit(cell);
      curv_values.reinit(cell);

      normal_values.read_dof_values_plain(this->normal_vector_field);

      // This computes (v, \nabla \cdot (n/|n|)). Since we store only \nabla
      // \phi in the vector normal_values, the normalization must be done here
      // and be done before we apply the derivatives, which is done in the
      // code below.
      bool all_zero = true;
      for (unsigned int i = 0; i < normal_values.dofs_per_component; ++i)
        {
          normal_values.submit_dof_value(normalize(normal_values.get_dof_value(i),
                                                   all_zero),
                                         i);
        }

      if (all_zero == false)
        {
          normal_values.evaluate(EvaluationFlags::gradients);
          for (unsigned int q = 0; q < normal_values.n_q_points; ++q)
            curv_values.submit_value(-get_divergence(normal_values, q), q);
          curv_values.integrate(EvaluationFlags::values);
          curv_values.distribute_local_to_global(dst);
        }
    }
}



template <int dim>
void
adaflo::LevelSetOKZSolverComputeCurvature<dim>::compute_curvature_vmult(
  LinearAlgebra::distributed::Vector<double>       &dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const bool                                        apply_diffusion) const
{
  dst = 0.;
  if (apply_diffusion)
    {
      // diffusion_setting will be 1 (true) in local_compute_curvature so that
      // damping will be added
#define OPERATION(c_degree, u_degree)                                                \
  this->matrix_free.cell_loop(&LevelSetOKZSolverComputeCurvature<                    \
                                dim>::template local_compute_curvature<c_degree, 1>, \
                              this,                                                  \
                              dst,                                                   \
                              src)

      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }
  else
    {
      // diffusion_setting will be 0 (fals) in local_compute_curvature so that
      // NO damping will be added
#define OPERATION(c_degree, u_degree)                                                \
  this->matrix_free.cell_loop(&LevelSetOKZSolverComputeCurvature<                    \
                                dim>::template local_compute_curvature<c_degree, 0>, \
                              this,                                                  \
                              dst,                                                   \
                              src)

      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

  for (const unsigned int entry :
       this->matrix_free.get_constrained_dofs(parameters.dof_index_curvature))
    dst.local_element(entry) =
      preconditioner.get_vector().local_element(entry) * src.local_element(entry);
}



template <int dim>
struct ComputeCurvatureMatrix
{
  ComputeCurvatureMatrix(const adaflo::LevelSetOKZSolverComputeCurvature<dim> &problem)
    : problem(problem)
  {}

  void
  vmult(LinearAlgebra::distributed::Vector<double>       &dst,
        const LinearAlgebra::distributed::Vector<double> &src) const
  {
    problem.compute_curvature_vmult(dst, src, true);
  }

  const adaflo::LevelSetOKZSolverComputeCurvature<dim> &problem;
};



// @sect4{LevelSetOKZSolverComputeCurvature::compute_normal}
template <int dim>
void
adaflo::LevelSetOKZSolverComputeCurvature<dim>::compute_curvature(const bool)
{
  // compute right hand side
  rhs = 0;

#define OPERATION(c_degree, u_degree)                                                 \
  this->matrix_free.cell_loop(&LevelSetOKZSolverComputeCurvature<                     \
                                dim>::template local_compute_curvature_rhs<c_degree>, \
                              this,                                                   \
                              rhs,                                                    \
                              this->solution_ls)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  // solve linear system for projection
  if (this->parameters.approximate_projections == true)
    preconditioner.vmult(this->solution_curvature, rhs);
  else
    {
      ComputeCurvatureMatrix<dim> matrix(*this);

      ReductionControl solver_control(2000, 1e-50, 1e-8);
      SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
      // cg.solve (matrix, this->solution_curvature, rhs, preconditioner);
      cg.solve(*projection_matrix, this->solution_curvature, rhs, *ilu_projection_matrix);
      // this->pcout << "N its curv: " << solver_control.last_step() <<
      // std::endl;
    }

  // correct curvature away from the zero level set by computing the distance
  // and correcting the value, if so requested
  if (this->parameters.curvature_correction == true)
    {
      for (unsigned int i = 0; i < this->solution_curvature.locally_owned_size(); ++i)
        if (this->solution_curvature.local_element(i) > 1e-4)
          {
            const double c_val = this->solution_ls.local_element(i);
            const double distance =
              (1 - c_val * c_val) > 1e-2 ?
                this->epsilon_used * std::log((1. + c_val) / (1. - c_val)) :
                0;

            this->solution_curvature.local_element(i) =
              1. /
              (1. / this->solution_curvature.local_element(i) + distance / (dim - 1));
          }
    }

  this->constraints_curvature.distribute(this->solution_curvature);

  // apply damping to avoid oscillations. this corresponds to one time
  // step of explicit Euler for a diffusion problem (need to avoid too
  // large diffusions!)
  if (this->parameters.approximate_projections == true)
    for (unsigned int i = 0; i < 8; ++i)
      {
        compute_curvature_vmult(rhs, this->solution_curvature, 2);
        preconditioner.vmult(rhs, rhs);
        this->solution_curvature.add(-0.05, rhs);
        this->constraints.distribute(this->solution_curvature);
      }
  this->solution_curvature.update_ghost_values();
}



template class adaflo::LevelSetOKZSolverComputeCurvature<1>;
template class adaflo::LevelSetOKZSolverComputeCurvature<2>;
template class adaflo::LevelSetOKZSolverComputeCurvature<3>;
