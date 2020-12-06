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
template <int ls_degree, int diffusion_setting>
void
LevelSetOKZSolverComputeCurvature<dim>::local_compute_curvature(
  const MatrixFree<dim, double> &                   data,
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  // The second input argument below refers to which constrains should be used,
  // 3 means constraints_curvature
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1> phi(data, 3, 2);
  const VectorizedArray<double>                  min_diameter =
    make_vectorized_array(this->epsilon_used / this->parameters.epsilon);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(src);
      // If diffusion_setting is true a damping term is added to the weak form
      //  i.e. diffusion_setting=1 => diffusion_setting%2 == 1 is true.
      phi.evaluate(diffusion_setting < 2, diffusion_setting % 2 == 1);
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
      phi.integrate(diffusion_setting < 2, diffusion_setting % 2 == 1);
      phi.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int ls_degree>
void
LevelSetOKZSolverComputeCurvature<dim>::local_compute_curvature_rhs(
  const MatrixFree<dim, double> &             data,
  LinearAlgebra::distributed::Vector<double> &dst,
  const LinearAlgebra::distributed::Vector<double> &,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // The second input argument below refers to which constrains should be used,
  // 4 means constraints_normals and 3 constraints_curvature
  FEEvaluation<dim, ls_degree, 2 * ls_degree, dim> normal_values(data, 4, 2);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1>   curv_values(data, 3, 2);

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
          Tensor<1, dim, VectorizedArray<double>> normal = normal_values.get_dof_value(i);
          const VectorizedArray<double>           normal_norm = normal.norm();
          for (unsigned int d = 0; d < VectorizedArray<double>::size(); ++d)
            if (normal_norm[d] > 1e-2)
              {
                all_zero = false;
                for (unsigned int e = 0; e < dim; ++e)
                  normal[e][d] /= normal_norm[d];
              }
            else
              for (unsigned int e = 0; e < dim; ++e)
                normal[e][d] = 0;
          normal_values.submit_dof_value(normal, i);
        }

      if (all_zero == false)
        {
          normal_values.evaluate(false, true, false);
          for (unsigned int q = 0; q < normal_values.n_q_points; ++q)
            curv_values.submit_value(-normal_values.get_divergence(q), q);
          curv_values.integrate(true, false);
          curv_values.distribute_local_to_global(dst);
        }
    }
}



template <int dim>
void
LevelSetOKZSolverComputeCurvature<dim>::compute_curvature_vmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
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

  // The numer "3" below is so that constraints_curvature is used
  for (unsigned int i = 0; i < this->matrix_free.get_constrained_dofs(3).size(); ++i)
    dst.local_element(this->matrix_free.get_constrained_dofs(3)[i]) =
      preconditioner.get_vector().local_element(
        this->matrix_free.get_constrained_dofs(3)[i]) *
      src.local_element(this->matrix_free.get_constrained_dofs(3)[i]);
}



template <int dim>
struct ComputeCurvatureMatrix
{
  ComputeCurvatureMatrix(const LevelSetOKZSolverComputeCurvature<dim> &problem)
    : problem(problem)
  {}

  void
  vmult(LinearAlgebra::distributed::Vector<double> &      dst,
        const LinearAlgebra::distributed::Vector<double> &src) const
  {
    problem.compute_curvature_vmult(dst, src, true);
  }

  const LevelSetOKZSolverComputeCurvature<dim> &problem;
};



// @sect4{LevelSetOKZSolverComputeCurvature::compute_normal}
template <int dim>
void
LevelSetOKZSolverComputeCurvature<dim>::compute_curvature(const bool)
{
  // This function computes the curvature from the normal field. Could also
  // compute the curvature directly from C, but that is less accurate. TODO:
  // include that variant by a parameter
  normal_operator.compute_normal(false);

  TimerOutput::Scope timer(*this->timer, "LS compute curvature.");

  // compute right hand side
  LinearAlgebra::distributed::Vector<double> &rhs = this->system_rhs.block(0);
  rhs                                             = 0;

#define OPERATION(c_degree, u_degree)                                                 \
  this->matrix_free.cell_loop(&LevelSetOKZSolverComputeCurvature<                     \
                                dim>::template local_compute_curvature_rhs<c_degree>, \
                              this,                                                   \
                              rhs,                                                    \
                              this->solution.block(0))

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  // solve linear system for projection
  if (this->parameters.approximate_projections == true)
    preconditioner.vmult(this->solution.block(1), rhs);
  else
    {
      ComputeCurvatureMatrix<dim> matrix(*this);

      ReductionControl solver_control(2000, 1e-50, 1e-8);
      SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
      // cg.solve (matrix, this->solution.block(1), rhs, preconditioner);
      cg.solve(*projection_matrix, this->solution.block(1), rhs, *ilu_projection_matrix);
      // this->pcout << "N its curv: " << solver_control.last_step() <<
      // std::endl;
    }

  // correct curvature away from the zero level set by computing the distance
  // and correcting the value, if so requested
  if (this->parameters.curvature_correction == true)
    {
      for (unsigned int i = 0; i < this->solution.block(1).local_size(); ++i)
        if (this->solution.block(1).local_element(i) > 1e-4)
          {
            const double c_val = this->solution.block(0).local_element(i);
            const double distance =
              (1 - c_val * c_val) > 1e-2 ?
                this->epsilon_used * std::log((1. + c_val) / (1. - c_val)) :
                0;

            this->solution.block(1).local_element(i) =
              1. / (1. / this->solution.block(1).local_element(i) + distance / (dim - 1));
          }
    }

  this->constraints_curvature.distribute(this->solution.block(1));

  // apply damping to avoid oscillations. this corresponds to one time
  // step of explicit Euler for a diffusion problem (need to avoid too
  // large diffusions!)
  if (this->parameters.approximate_projections == true)
    for (unsigned int i = 0; i < 8; ++i)
      {
        compute_curvature_vmult(rhs, this->solution.block(1), 2);
        preconditioner.vmult(rhs, rhs);
        this->solution.block(1).add(-0.05, rhs);
        this->constraints.distribute(this->solution.block(1));
      }
  this->solution.block(1).update_ghost_values();
}



template class LevelSetOKZSolverComputeCurvature<2>;
template class LevelSetOKZSolverComputeCurvature<3>;
