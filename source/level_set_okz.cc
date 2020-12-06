// --------------------------------------------------------------------------
//
// Copyright (C) 2008 - 2016 by the adaflo authors
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
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include <adaflo/block_matrix_extension.h>
#include <adaflo/level_set_okz.h>
#include <adaflo/level_set_okz_advance_concentration.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/level_set_okz_reinitialization.h>
#include <adaflo/level_set_okz_template_instantations.h>
#include <adaflo/util.h>

#include <fstream>
#include <iostream>


using namespace dealii;



template <int dim>
LevelSetOKZSolver<dim>::LevelSetOKZSolver(const FlowParameters &parameters_in,
                                          Triangulation<dim> &  tria_in)
  : LevelSetBaseAlgorithm<dim>(parameters_in, tria_in)
  , first_reinit_step(true)
{}



template <int dim>
void
LevelSetOKZSolver<dim>::transform_distance_function(
  LinearAlgebra::distributed::Vector<double> &vector) const
{
  Assert(this->epsilon_used > 0, ExcInternalError());
  for (unsigned int i = 0; i < vector.local_size(); i++)
    vector.local_element(i) =
      -std::tanh(vector.local_element(i) / (2. * this->epsilon_used));
}



namespace AssemblyData
{
  struct Data
  {
    Data()
    {
      AssertThrow(false, ExcNotImplemented());
    }

    Data(const unsigned int size)
      : matrices(VectorizedArray<double>::size(), FullMatrix<double>(size, size))
      , dof_indices(size)
    {}

    Data(const Data &other)
      : matrices(other.matrices)
      , dof_indices(other.dof_indices)
    {}

    std::vector<FullMatrix<double>>      matrices;
    std::vector<types::global_dof_index> dof_indices;
  };
} // namespace AssemblyData


// @sect4{LevelSetOKZSolver::make_grid_and_dofs}
template <int dim>
void
LevelSetOKZSolver<dim>::initialize_data_structures()
{
  cell_diameters_float.clear();
  matrix_free_float.clear();
  this->LevelSetBaseAlgorithm<dim>::initialize_data_structures();

  artificial_viscosities.resize(this->matrix_free.n_cell_batches());
  evaluated_convection.resize(this->matrix_free.n_cell_batches() *
                              this->matrix_free.get_n_q_points(2));

  // create diagonal preconditioner vector by assembly of mass matrix diagonal
  LinearAlgebra::distributed::Vector<double> diagonal(this->solution_update.block(0));
  {
    diagonal = 0;
    QIterated<dim> quadrature(QGauss<1>(2), this->parameters.concentration_subdivisions);
    FEValues<dim>  fe_values(this->mapping,
                            *this->fe,
                            quadrature,
                            update_values | update_JxW_values);
    Vector<double> local_rhs(this->fe->dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(this->fe->dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     this->dof_handler.begin_active(),
                                                   end = this->dof_handler.end();
    for (; cell != end; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          for (unsigned int i = 0; i < this->fe->dofs_per_cell; ++i)
            {
              double value = 0;
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                value += fe_values.shape_value(i, q) * fe_values.shape_value(i, q) *
                         fe_values.JxW(q);
              local_rhs(i) = value;
            }
          cell->get_dof_indices(local_dof_indices);
          this->hanging_node_constraints.distribute_local_to_global(local_rhs,
                                                                    local_dof_indices,
                                                                    diagonal);
        }
    diagonal.compress(VectorOperation::add);
    preconditioner.reinit(diagonal);
  }

  // Initialize float matrix-free object for normal computation in case we
  // want to do that at some point...

  // vectors_normal.release_unused_memory();
  // typename MatrixFree<dim,float>::AdditionalData data;
  // data.tasks_parallel_scheme =
  //   MatrixFree<dim,float>::AdditionalData::partition_partition;
  // data.mapping_update_flags = update_JxW_values | update_gradients;
  // data.store_plain_indices = false;
  // data.mpi_communicator = this->triangulation.get_communicator();
  // matrix_free_float.reinit(this->mapping, this->dof_handler,
  //                          this->constraints,
  //                          QIterated<1> (QGauss<1>(2),
  //                          this->parameters.concentration_subdivisions),
  //                          data);
  // cell_diameters_float.resize(matrix_free_float.n_cell_batches());
  // std::map<std::pair<unsigned int,unsigned int>,double> diameters;
  // for (unsigned int c=0; c<this->matrix_free.n_cell_batches(); ++c)
  //   for (unsigned int v=0; v<this->matrix_free.n_active_entries_per_cell_batch(c); ++v)
  //     diameters[std::make_pair(this->matrix_free.get_cell_iterator(c,v)->level(),
  //                              this->matrix_free.get_cell_iterator(c,v)->index())]
  //       = this->cell_diameters[c][v];
  // for (unsigned int c=0; c<matrix_free_float.n_cell_batches(); ++c)
  //   for (unsigned int v=0; v<matrix_free_float.n_active_entries_per_cell_batch(c); ++v)
  //     cell_diameters_float[c][v] =
  //       diameters[std::make_pair(matrix_free_float.get_cell_iterator(c,v)->level(),
  //                                matrix_free_float.get_cell_iterator(c,v)->index())];

  // LinearAlgebra::distributed::Vector<float> diagonal_f;
  // diagonal_f = this->solution_update.block(0);
  // preconditioner_float.reinit(diagonal_f);


  // create sparse matrix for projection systems.
  //
  // First off is the creation of a mask that only adds those entries of
  // FE_Q_iso_Q0 that are going to have a non-zero matrix entry -> this
  // ensures as compact a matrix as for Q1 on the fine mesh. To find them,
  // check terms in a mass matrix.
  Table<2, bool> dof_mask(this->fe->dofs_per_cell, this->fe->dofs_per_cell);
  {
    QIterated<dim> quadrature(QGauss<1>(1), this->parameters.concentration_subdivisions);
    FEValues<dim>  fe_values(this->mapping, *this->fe, quadrature, update_values);
    fe_values.reinit(this->dof_handler.begin());
    for (unsigned int i = 0; i < this->fe->dofs_per_cell; ++i)
      for (unsigned int j = 0; j < this->fe->dofs_per_cell; ++j)
        {
          double sum = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            sum += fe_values.shape_value(i, q) * fe_values.shape_value(j, q);
          if (sum != 0)
            dof_mask(i, j) = true;
        }
  }
  {
    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(this->dof_handler, relevant_dofs);
    TrilinosWrappers::SparsityPattern csp;
    csp.reinit(this->dof_handler.locally_owned_dofs(),
               this->dof_handler.locally_owned_dofs(),
               relevant_dofs,
               get_communicator(this->triangulation));
    std::vector<types::global_dof_index> local_dof_indices(this->fe->dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     this->dof_handler.begin_active(),
                                                   endc = this->dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);
          this->constraints_normals.add_entries_local_to_global(local_dof_indices,
                                                                csp,
                                                                false,
                                                                dof_mask);
        }
    csp.compress();
    projection_matrix = std::make_shared<BlockMatrixExtension>();
    projection_matrix->reinit(csp);
  }
  {
    AssemblyData::Data scratch_data(this->fe->dofs_per_cell);
    auto               scratch_local =
      std::make_shared<Threads::ThreadLocalStorage<AssemblyData::Data>>(scratch_data);
    unsigned int dummy = 0;
    this->matrix_free.cell_loop(&LevelSetOKZSolver<dim>::local_projection_matrix,
                                this,
                                scratch_local,
                                dummy);
    projection_matrix->compress(VectorOperation::add);
    ilu_projection_matrix = std::make_shared<BlockILUExtension>();
    ilu_projection_matrix->initialize(*projection_matrix);
  }
}



template <int dim>
void
LevelSetOKZSolver<dim>::local_projection_matrix(
  const MatrixFree<dim> &                                           data,
  std::shared_ptr<Threads::ThreadLocalStorage<AssemblyData::Data>> &scratch,
  const unsigned int &,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  const unsigned int ls_degree = this->parameters.concentration_subdivisions;
  if (ls_degree == 1)
    local_projection_matrix<1>(data, scratch, cell_range);
  else if (ls_degree == 2)
    local_projection_matrix<2>(data, scratch, cell_range);
  else if (ls_degree == 3)
    local_projection_matrix<3>(data, scratch, cell_range);
  else if (ls_degree == 4)
    local_projection_matrix<4>(data, scratch, cell_range);
  else
    AssertThrow(false, ExcNotImplemented());
}



template <int dim>
template <int ls_degree>
void
LevelSetOKZSolver<dim>::local_projection_matrix(
  const MatrixFree<dim> &                                           data,
  std::shared_ptr<Threads::ThreadLocalStorage<AssemblyData::Data>> &scratch_data,
  const std::pair<unsigned int, unsigned int> &                     cell_range)
{
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1, double> phi(data, 4, 2);
  AssemblyData::Data &                                   scratch = scratch_data->get();

  const VectorizedArray<double> min_diameter =
    make_vectorized_array(this->epsilon_used / this->parameters.epsilon);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      const VectorizedArray<double> damping =
        4. * Utilities::fixed_power<2>(
               std::max(min_diameter,
                        this->cell_diameters[cell] / static_cast<double>(ls_degree)));

      for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
            phi.begin_dof_values()[j] = VectorizedArray<double>();
          phi.begin_dof_values()[i] = 1.;
          phi.evaluate(true, true);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              phi.submit_value(phi.get_value(q), q);
              phi.submit_gradient(phi.get_gradient(q) * damping, q);
            }
          phi.integrate(true, true);
          for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              scratch.matrices[v](phi.get_shape_info().lexicographic_numbering[j],
                                  phi.get_shape_info().lexicographic_numbering[i]) =
                phi.begin_dof_values()[j][v];
        }
      for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
        {
          typename DoFHandler<dim>::active_cell_iterator dcell =
            this->matrix_free.get_cell_iterator(cell, v, 2);
          dcell->get_dof_indices(scratch.dof_indices);
          this->constraints_normals.distribute_local_to_global(
            scratch.matrices[v],
            scratch.dof_indices,
            static_cast<TrilinosWrappers::SparseMatrix &>(*projection_matrix));
        }
    }
}



template <int dim>
template <int ls_degree, int velocity_degree>
void
LevelSetOKZSolver<dim>::local_compute_force(
  const MatrixFree<dim, double> &             data,
  LinearAlgebra::distributed::Vector<double> &dst,
  const LinearAlgebra::distributed::Vector<double> &,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function), 3 means constraints_curvature and so
  // an
  FEEvaluation<dim, ls_degree, velocity_degree + 1, 1>           ls_values(data, 2, 0);
  FEEvaluation<dim, ls_degree, velocity_degree + 1, 1>           curv_values(data, 3, 0);
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

      ls_values.read_dof_values_plain(this->heaviside);

      // set variable parameters
      if (use_variable_parameters)
        {
          vector_t *densities = this->navier_stokes.get_matrix().begin_densities(cell);
          vector_t *viscosities =
            this->navier_stokes.get_matrix().begin_viscosities(cell);
          ls_values.evaluate(true, false);
          for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
            {
              densities[q] = this->parameters.density +
                             this->parameters.density_diff * ls_values.get_value(q);
              viscosities[q] = this->parameters.viscosity +
                               this->parameters.viscosity_diff * ls_values.get_value(q);
            }
        }

      // interpolate ls values onto pressure
      if (this->parameters.interpolate_grad_onto_pressure)
        {
          for (unsigned int i = 0; i < pre_values.dofs_per_cell; ++i)
            {
              vector_t projected_value = vector_t();
              for (unsigned int j = 0; j < ls_values.dofs_per_cell; ++j)
                projected_value += this->interpolation_concentration_pressure(i, j) *
                                   ls_values.get_dof_value(j);
              pre_values.submit_dof_value(projected_value, i);
            }
          pre_values.evaluate(false, true);
        }
      else
        ls_values.evaluate(false, true);

      // evaluate curvature and level set gradient
      curv_values.read_dof_values_plain(this->solution.block(1));
      curv_values.evaluate(true, false);

      // evaluate surface tension force and gravity force
      for (unsigned int q = 0; q < curv_values.n_q_points; ++q)
        {
          // surface tension
          Tensor<1, dim, vector_t> force =
            (this->parameters.surface_tension * curv_values.get_value(q)) *
            (this->parameters.interpolate_grad_onto_pressure ?
               pre_values.get_gradient(q) :
               ls_values.get_gradient(q));

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
void
LevelSetOKZSolver<dim>::compute_force()
{
  compute_heaviside();
  compute_curvature();

  TimerOutput::Scope timer(*this->timer, "LS compute force.");

  this->compute_density_on_faces();

  this->navier_stokes.user_rhs = 0;
#define OPERATION(ls_degree, vel_degree)                                          \
  this->matrix_free.cell_loop(                                                    \
    &LevelSetOKZSolver<dim>::template local_compute_force<ls_degree, vel_degree>, \
    this,                                                                         \
    this->navier_stokes.user_rhs.block(0),                                        \
    this->solution.block(0))

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
}



// @sect4{LevelSetOKZSolver::advance_concentration}
template <int dim>
void
LevelSetOKZSolver<dim>::advance_concentration()
{
  LevelSetOKZSolverAdvanceConcentration<dim> op(this->solution_old,
                                                this->solution_old_old,
                                                this->triangulation,
                                                this->global_omega_diameter,
                                                this->cell_diameters,

                                                this->constraints,
                                                this->pcout,
                                                this->time_stepping,
                                                this->boundary,
                                                this->mapping,
                                                this->dof_handler,
                                                this->fe,
                                                this->matrix_free,
                                                this->timer,
                                                this->solution_update,
                                                this->solution,
                                                this->system_rhs,
                                                this->navier_stokes,
                                                this->parameters,
                                                this->artificial_viscosities,
                                                this->global_max_velocity,
                                                this->preconditioner,
                                                this->evaluated_convection);

  op.advance_concentration();
}



// @sect4{LevelSetOKZSolver::compute_normal}
template <int dim>
void
LevelSetOKZSolver<dim>::compute_normal(const bool fast_computation)
{
  LevelSetOKZSolverComputeNormal<dim> op(this->cell_diameters,
                                         this->epsilon_used,
                                         this->minimal_edge_length,
                                         this->constraints_normals,
                                         this->normal_vector_field,
                                         this->timer,
                                         this->navier_stokes,
                                         this->parameters,
                                         this->matrix_free,
                                         this->solution,
                                         this->normal_vector_rhs,
                                         this->matrix_free_float,
                                         this->cell_diameters_float,
                                         this->preconditioner,
                                         this->projection_matrix,
                                         this->ilu_projection_matrix);
  op.compute_normal(fast_computation);
}



template <int dim>
template <int ls_degree, int diffusion_setting>
void
LevelSetOKZSolver<dim>::local_compute_curvature(
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
LevelSetOKZSolver<dim>::local_compute_curvature_rhs(
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
LevelSetOKZSolver<dim>::compute_curvature_vmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const bool                                        apply_diffusion) const
{
  dst = 0.;
  if (apply_diffusion)
    {
      // diffusion_setting will be 1 (true) in local_compute_curvature so that
      // damping will be added
#define OPERATION(c_degree, u_degree)                                       \
  this->matrix_free.cell_loop(                                              \
    &LevelSetOKZSolver<dim>::template local_compute_curvature<c_degree, 1>, \
    this,                                                                   \
    dst,                                                                    \
    src)

      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }
  else
    {
      // diffusion_setting will be 0 (fals) in local_compute_curvature so that
      // NO damping will be added
#define OPERATION(c_degree, u_degree)                                       \
  this->matrix_free.cell_loop(                                              \
    &LevelSetOKZSolver<dim>::template local_compute_curvature<c_degree, 0>, \
    this,                                                                   \
    dst,                                                                    \
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
  ComputeCurvatureMatrix(const LevelSetOKZSolver<dim> &problem)
    : problem(problem)
  {}

  void
  vmult(LinearAlgebra::distributed::Vector<double> &      dst,
        const LinearAlgebra::distributed::Vector<double> &src) const
  {
    problem.compute_curvature_vmult(dst, src, true);
  }

  const LevelSetOKZSolver<dim> &problem;
};



// @sect4{LevelSetOKZSolver::compute_normal}
template <int dim>
void
LevelSetOKZSolver<dim>::compute_curvature(const bool)
{
  // This function computes the curvature from the normal field. Could also
  // compute the curvature directly from C, but that is less accurate. TODO:
  // include that variant by a parameter
  compute_normal(false);

  TimerOutput::Scope timer(*this->timer, "LS compute curvature.");

  // compute right hand side
  LinearAlgebra::distributed::Vector<double> &rhs = this->system_rhs.block(0);
  rhs                                             = 0;

#define OPERATION(c_degree, u_degree)                                        \
  this->matrix_free.cell_loop(                                               \
    &LevelSetOKZSolver<dim>::template local_compute_curvature_rhs<c_degree>, \
    this,                                                                    \
    rhs,                                                                     \
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



// force is only active in a few elements around the interface. find these
// elements now
template <int dim>
void
LevelSetOKZSolver<dim>::compute_heaviside()
{
  TimerOutput::Scope timer(*this->timer, "LS compute Heaviside.");
  const double       cutoff = std::tanh(2);

  const unsigned int                   dofs_per_cell = this->fe->dofs_per_cell;
  Vector<double>                       local_values(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler.begin_active(),
                                                 endc = this->dof_handler.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) // Check if the cell is owned by the local
                                  // processor
      {
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          local_values[i] = this->solution.block(0)[local_dof_indices[i]];
        bool consider_cell = false;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          if (std::fabs(local_values(i)) < cutoff)
            {
              consider_cell = true;
              break;
            }
        if (consider_cell == true)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const double c_val    = local_values(i);
                double       distance = 0;
                if (c_val < -cutoff)
                  distance = -3;
                else if (c_val > cutoff)
                  distance = 3;
                else
                  distance = std::log((1 + c_val) / (1 - c_val));

                // want to have force along width 2*h which is
                // what distance scaled by twice relative
                // epsilon but not mesh size is doing in
                // discrete_heaviside that is defined between
                // -2 and 2.
                distance *= this->parameters.epsilon * 2. /
                            this->parameters.concentration_subdivisions;
                this->heaviside(local_dof_indices[i]) = discrete_heaviside(distance);
              }
          }
        else
          {
            if (local_values(0) < 0)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                this->heaviside(local_dof_indices[i]) = 0.;
            else
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                this->heaviside(local_dof_indices[i]) = 1.;
          }
      }
  this->heaviside.update_ghost_values();
}



template <int dim>
void
LevelSetOKZSolver<dim>::reinitialize(const unsigned int stab_steps,
                                     const unsigned int diff_steps,
                                     const bool diffuse_cells_with_large_curvature_only)
{
  LevelSetOKZSolverReinitialization<dim> op(this->normal_vector_field,
                                            this->cell_diameters,
                                            this->epsilon_used,
                                            this->minimal_edge_length,
                                            this->navier_stokes,
                                            this->constraints,
                                            this->solution_update,
                                            this->solution,
                                            this->system_rhs,
                                            this->timer,
                                            this->pcout,
                                            this->preconditioner,
                                            this->last_concentration_range,
                                            this->parameters,
                                            this->time_stepping,
                                            this->first_reinit_step,
                                            this->matrix_free,
                                            this->evaluated_convection);

  op.reinitialize(stab_steps, diff_steps, diffuse_cells_with_large_curvature_only);
}



template class LevelSetOKZSolver<2>;
template class LevelSetOKZSolver<3>;
