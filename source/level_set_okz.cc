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

#include <fstream>
#include <iostream>


using namespace dealii;



template <int dim>
LevelSetOKZSolver<dim>::LevelSetOKZSolver(
  const FlowParameters &                     parameters_in,
  parallel::distributed::Triangulation<dim> &tria_in)
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

  artificial_viscosities.resize(this->matrix_free.n_macro_cells());
  evaluated_convection.resize(this->matrix_free.n_macro_cells() *
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
  // cell_diameters_float.resize(matrix_free_float.n_macro_cells());
  // std::map<std::pair<unsigned int,unsigned int>,double> diameters;
  // for (unsigned int c=0; c<this->matrix_free.n_macro_cells(); ++c)
  //   for (unsigned int v=0; v<this->matrix_free.n_components_filled(c); ++v)
  //     diameters[std::make_pair(this->matrix_free.get_cell_iterator(c,v)->level(),
  //                              this->matrix_free.get_cell_iterator(c,v)->index())]
  //       = this->cell_diameters[c][v];
  // for (unsigned int c=0; c<matrix_free_float.n_macro_cells(); ++c)
  //   for (unsigned int v=0; v<matrix_free_float.n_components_filled(c); ++v)
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
               this->triangulation.get_communicator());
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
          for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              scratch.matrices[v](phi.get_shape_info().lexicographic_numbering[j],
                                  phi.get_shape_info().lexicographic_numbering[i]) =
                phi.begin_dof_values()[j][v];
        }
      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
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
template <int ls_degree, int velocity_degree>
void
LevelSetOKZSolver<dim>::local_advance_concentration(
  const MatrixFree<dim, double> &                   data,
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function)
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1> ls_values(data, 2, 2);
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      const Tensor<1, dim, VectorizedArray<double>> *velocities =
        &evaluated_convection[cell * ls_values.n_q_points];
      ls_values.reinit(cell);

      ls_values.read_dof_values(src);
      ls_values.evaluate(true, true);

      for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
        {
          const VectorizedArray<double>                 ls_val = ls_values.get_value(q);
          const Tensor<1, dim, VectorizedArray<double>> ls_grad =
            ls_values.get_gradient(q);
          ls_values.submit_value(ls_val * this->time_stepping.weight() +
                                   ls_grad * velocities[q],
                                 q);
          if (this->parameters.convection_stabilization)
            ls_values.submit_gradient(artificial_viscosities[cell] * ls_grad, q);
        }
      ls_values.integrate(true, this->parameters.convection_stabilization);
      ls_values.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int ls_degree, int velocity_degree>
void
LevelSetOKZSolver<dim>::local_advance_concentration_rhs(
  const MatrixFree<dim, double> &             data,
  LinearAlgebra::distributed::Vector<double> &dst,
  const LinearAlgebra::distributed::Vector<double> &,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function) and 0 means
  // &navier_stokes.get_constraints_u()
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1>         ls_values(data, 2, 2);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1>         ls_values_old(data, 2, 2);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1>         ls_values_old_old(data, 2, 2);
  FEEvaluation<dim, velocity_degree, 2 * ls_degree, dim> vel_values(data, 0, 2);
  FEEvaluation<dim, velocity_degree, 2 * ls_degree, dim> vel_values_old(data, 0, 2);
  FEEvaluation<dim, velocity_degree, 2 * ls_degree, dim> vel_values_old_old(data, 0, 2);

  typedef VectorizedArray<double> vector_t;

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      ls_values.reinit(cell);
      ls_values_old.reinit(cell);
      ls_values_old_old.reinit(cell);
      vel_values.reinit(cell);
      vel_values_old.reinit(cell);
      vel_values_old_old.reinit(cell);

      vel_values.read_dof_values_plain(this->navier_stokes.solution.block(0));
      vel_values_old.read_dof_values_plain(this->navier_stokes.solution_old.block(0));
      vel_values_old_old.read_dof_values_plain(
        this->navier_stokes.solution_old_old.block(0));
      ls_values.read_dof_values_plain(this->solution.block(0));
      ls_values_old.read_dof_values_plain(this->solution_old.block(0));
      ls_values_old_old.read_dof_values_plain(this->solution_old_old.block(0));

      vel_values.evaluate(true, false);
      vel_values_old.evaluate(true, false);
      vel_values_old_old.evaluate(true, false);
      ls_values.evaluate(true, true);
      ls_values_old.evaluate(true, true);
      ls_values_old_old.evaluate(true, true);

      if (this->parameters.convection_stabilization)
        {
          vector_t max_residual = vector_t(), max_velocity = vector_t();
          for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
            {
              // compute residual of concentration equation
              Tensor<1, dim, vector_t> u =
                (vel_values_old.get_value(q) + vel_values_old_old.get_value(q));
              vector_t dc_dt =
                (ls_values_old.get_value(q) - ls_values_old_old.get_value(q)) /
                this->time_stepping.old_step_size();
              vector_t residual = std::abs(
                dc_dt +
                u * (ls_values_old.get_gradient(q) + ls_values_old_old.get_gradient(q)) *
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
        &evaluated_convection[cell * ls_values.n_q_points];

      for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
        {
          // compute right hand side
          vector_t old_value =
            this->time_stepping.weight_old() * ls_values_old.get_value(q);
          if (this->time_stepping.scheme() == TimeStepping::bdf_2 &&
              this->time_stepping.step_no() > 1)
            old_value +=
              this->time_stepping.weight_old_old() * ls_values_old_old.get_value(q);
          const vector_t                 ls_val  = ls_values.get_value(q);
          const Tensor<1, dim, vector_t> ls_grad = ls_values.get_gradient(q);
          vector_t residual = -(ls_val * this->time_stepping.weight() +
                                vel_values.get_value(q) * ls_grad + old_value);
          ls_values.submit_value(residual, q);
          if (this->parameters.convection_stabilization)
            ls_values.submit_gradient(-artificial_viscosities[cell] * ls_grad, q);
          velocities[q] = vel_values.get_value(q);
        }
      ls_values.integrate(true, this->parameters.convection_stabilization);
      ls_values.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int ls_degree, typename Number>
void
LevelSetOKZSolver<dim>::local_compute_normal(
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
LevelSetOKZSolver<dim>::local_compute_normal_rhs(
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
template <int ls_degree, bool diffuse_only>
void
LevelSetOKZSolver<dim>::local_reinitialize(
  const MatrixFree<dim, double> &                   data,
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  const double dtau_inv = std::max(0.95 / (1. / (dim * dim) * this->minimal_edge_length /
                                           this->parameters.concentration_subdivisions),
                                   1. / (5. * this->time_stepping.step_size()));

  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function)
  FEEvaluation<dim, ls_degree, 2 * ls_degree> phi(data, 2, 2);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(src);
      phi.evaluate(true, true, false);

      VectorizedArray<double> cell_diameter = this->cell_diameters[cell];
      VectorizedArray<double> diffusion =
        std::max(make_vectorized_array(this->epsilon_used),
                 cell_diameter / static_cast<double>(ls_degree));

      const Tensor<1, dim, VectorizedArray<double>> *normal =
        &evaluated_convection[cell * phi.n_q_points];
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

      phi.integrate(true, true);
      phi.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int ls_degree, bool diffuse_only>
void
LevelSetOKZSolver<dim>::local_reinitialize_rhs(
  const MatrixFree<dim, double> &             data,
  LinearAlgebra::distributed::Vector<double> &dst,
  const LinearAlgebra::distributed::Vector<double> &,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function) and 4 means constraints_normals
  FEEvaluation<dim, ls_degree, 2 * ls_degree>      phi(data, 2, 2);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, dim> normals(data, 4, 2);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values_plain(this->solution.block(0));
      phi.evaluate(true, true, false);

      normals.reinit(cell);
      normals.read_dof_values_plain(this->normal_vector_field);
      normals.evaluate(true, false, false);

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
                Tensor<1, dim, VectorizedArray<double>> normal = normals.get_value(q);
                normal /= std::max(make_vectorized_array(1e-4), normal.norm());
                evaluated_convection[cell * phi.n_q_points + q] = normal;
              }
            // take normal as it was for the first reinit step
            Tensor<1, dim, VectorizedArray<double>> normal =
              evaluated_convection[cell * phi.n_q_points + q];
            phi.submit_gradient(normal *
                                  (0.5 * (1. - phi.get_value(q) * phi.get_value(q)) -
                                   (normal * grad * diffusion)),
                                q);
          }
        else
          {
            phi.submit_gradient(-diffusion * phi.get_gradient(q), q);
          }

      phi.integrate(false, true);
      phi.distribute_local_to_global(dst);
    }
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



template <int dim>
void
LevelSetOKZSolver<dim>::advance_concentration_vmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src) const
{
  dst = 0.;
#define OPERATION(c_degree, u_degree)                                                  \
  this->matrix_free.cell_loop(                                                         \
    &LevelSetOKZSolver<dim>::template local_advance_concentration<c_degree, u_degree>, \
    this,                                                                              \
    dst,                                                                               \
    src)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION


  if (this->parameters.convection_stabilization)
    {
      // Boundary part of stabilization-term:
      FEFaceValues<dim>                    fe_face_values(*this->fe,
                                       this->matrix_free.get_face_quadrature(1),
                                       update_values | update_gradients |
                                         update_JxW_values | update_normal_vectors);
      Vector<double>                       cell_rhs(this->fe->dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(this->fe->dofs_per_cell);
      std::vector<Tensor<1, dim>> local_gradients(fe_face_values.get_quadrature().size());
      src.update_ghost_values();

      for (unsigned int mcell = 0; mcell < this->matrix_free.n_macro_cells(); ++mcell)
        for (unsigned int v = 0; v < this->matrix_free.n_components_filled(mcell); ++v)
          {
            typename DoFHandler<dim>::active_cell_iterator cell =
              this->matrix_free.get_cell_iterator(mcell, v, 2);
            cell_rhs = 0;

            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell->face(face)->at_boundary())
                  {
                    if (this->boundary->symmetry.find(cell->face(face)->boundary_id()) ==
                        this->boundary->symmetry.end())
                      {
                        fe_face_values.reinit(cell, face);
                        fe_face_values.get_function_gradients(src, local_gradients);
                        for (unsigned int i = 0; i < this->fe->dofs_per_cell; ++i)
                          {
                            for (unsigned int q = 0;
                                 q < fe_face_values.get_quadrature().size();
                                 ++q)
                              {
                                cell_rhs(i) += -((fe_face_values.shape_value(i, q) *
                                                  fe_face_values.normal_vector(q) *
                                                  artificial_viscosities[mcell][v] *
                                                  local_gradients[q]) *
                                                 fe_face_values.JxW(q));
                              }
                          }
                      }
                  }
              }

            cell->get_dof_indices(local_dof_indices);
            this->constraints.distribute_local_to_global(cell_rhs,
                                                         local_dof_indices,
                                                         dst);
          }

      dst.compress(VectorOperation::add);
    }

  for (unsigned int i = 0; i < this->matrix_free.get_constrained_dofs(2).size(); ++i)
    dst.local_element(this->matrix_free.get_constrained_dofs(2)[i]) =
      preconditioner.get_vector().local_element(
        this->matrix_free.get_constrained_dofs(2)[i]) *
      src.local_element(this->matrix_free.get_constrained_dofs(2)[i]);
}



template <int dim>
struct AdvanceConcentrationMatrix
{
  AdvanceConcentrationMatrix(const LevelSetOKZSolver<dim> &problem)
    : problem(problem)
  {}

  void
  vmult(LinearAlgebra::distributed::Vector<double> &      dst,
        const LinearAlgebra::distributed::Vector<double> &src) const
  {
    problem.advance_concentration_vmult(dst, src);
  }

  const LevelSetOKZSolver<dim> &problem;
};



template <int dim>
void
LevelSetOKZSolver<dim>::compute_normal_vmult(
  LinearAlgebra::distributed::BlockVector<double> &      dst,
  const LinearAlgebra::distributed::BlockVector<double> &src) const
{
  dst = 0.;
#define OPERATION(c_degree, u_degree)                                         \
  this->matrix_free.cell_loop(                                                \
    &LevelSetOKZSolver<dim>::template local_compute_normal<c_degree, double>, \
    this,                                                                     \
    dst,                                                                      \
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
LevelSetOKZSolver<dim>::compute_normal_vmult(
  LinearAlgebra::distributed::BlockVector<float> &      dst,
  const LinearAlgebra::distributed::BlockVector<float> &src) const
{
  dst = 0.;
#define OPERATION(c_degree, u_degree)                                        \
  matrix_free_float.cell_loop(                                               \
    &LevelSetOKZSolver<dim>::template local_compute_normal<c_degree, float>, \
    this,                                                                    \
    dst,                                                                     \
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
  ComputeNormalMatrix(const LevelSetOKZSolver<dim> &problem)
    : problem(problem)
  {}

  template <typename Number>
  void
  vmult(LinearAlgebra::distributed::BlockVector<Number> &      dst,
        const LinearAlgebra::distributed::BlockVector<Number> &src) const
  {
    problem.compute_normal_vmult(dst, src);
  }

  const LevelSetOKZSolver<dim> &problem;
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



template <int dim>
void
LevelSetOKZSolver<dim>::reinitialization_vmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const bool                                        diffuse_only) const
{
  dst = 0.;
  if (diffuse_only)
    {
#define OPERATION(c_degree, u_degree)                                     \
  this->matrix_free.cell_loop(                                            \
    &LevelSetOKZSolver<dim>::template local_reinitialize<c_degree, true>, \
    this,                                                                 \
    dst,                                                                  \
    src)

      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }
  else
    {
#define OPERATION(c_degree, u_degree)                                      \
  this->matrix_free.cell_loop(                                             \
    &LevelSetOKZSolver<dim>::template local_reinitialize<c_degree, false>, \
    this,                                                                  \
    dst,                                                                   \
    src)

      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

  for (unsigned int i = 0; i < this->matrix_free.get_constrained_dofs(2).size(); ++i)
    dst.local_element(this->matrix_free.get_constrained_dofs(2)[i]) =
      preconditioner.get_vector().local_element(
        this->matrix_free.get_constrained_dofs(2)[i]) *
      src.local_element(this->matrix_free.get_constrained_dofs(2)[i]);
}



template <int dim>
struct ReinitializationMatrix
{
  ReinitializationMatrix(const LevelSetOKZSolver<dim> &problem, const bool diffuse_only)
    : problem(problem)
    , diffuse_only(diffuse_only)
  {}

  void
  vmult(LinearAlgebra::distributed::Vector<double> &      dst,
        const LinearAlgebra::distributed::Vector<double> &src) const
  {
    problem.reinitialization_vmult(dst, src, diffuse_only);
  }

  const LevelSetOKZSolver<dim> &problem;
  const bool                    diffuse_only;
};



// @sect4{LevelSetOKZSolver::advance_concentration}
template <int dim>
void
LevelSetOKZSolver<dim>::advance_concentration()
{
  TimerOutput::Scope timer(*this->timer, "LS advance concentration.");

  // apply boundary values
  {
    std::map<types::boundary_id, const Function<dim> *> dirichlet;
    Functions::ConstantFunction<dim>                    plus_func(1., 1);
    for (typename std::set<types::boundary_id>::const_iterator it =
           this->boundary->fluid_type_plus.begin();
         it != this->boundary->fluid_type_plus.end();
         ++it)
      dirichlet[*it] = &plus_func;
    Functions::ConstantFunction<dim> minus_func(-1., 1);
    for (typename std::set<types::boundary_id>::const_iterator it =
           this->boundary->fluid_type_minus.begin();
         it != this->boundary->fluid_type_minus.end();
         ++it)
      dirichlet[*it] = &minus_func;

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(this->mapping,
                                             this->dof_handler,
                                             dirichlet,
                                             boundary_values);

    for (typename std::map<types::global_dof_index, double>::const_iterator it =
           boundary_values.begin();
         it != boundary_values.end();
         ++it)
      if (this->solution.block(0).locally_owned_elements().is_element(it->first))
        this->solution.block(0)(it->first) = it->second;
    this->solution.block(0).update_ghost_values();
  }

  // compute right hand side
  global_max_velocity                                   = this->get_maximal_velocity();
  LinearAlgebra::distributed::Vector<double> &rhs       = this->system_rhs.block(0);
  LinearAlgebra::distributed::Vector<double> &increment = this->solution_update.block(0);
  rhs                                                   = 0;

#define OPERATION(c_degree, u_degree)                                            \
  this->matrix_free.cell_loop(                                                   \
    &LevelSetOKZSolver<dim>::template local_advance_concentration_rhs<c_degree,  \
                                                                      u_degree>, \
    this,                                                                        \
    rhs,                                                                         \
    this->solution.block(0))

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  AdvanceConcentrationMatrix<dim> matrix(*this);



  if (this->parameters.convection_stabilization)
    {
      // Boundary part of stabilization-term:
      FEFaceValues<dim> fe_face_values(this->mapping,
                                       *this->fe,
                                       this->matrix_free.get_face_quadrature(1),
                                       update_values | update_gradients |
                                         update_JxW_values | update_normal_vectors);

      Vector<double>                       cell_rhs(this->fe->dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(this->fe->dofs_per_cell);
      std::vector<Tensor<1, dim>> local_gradients(fe_face_values.get_quadrature().size());

      for (unsigned int mcell = 0; mcell < this->matrix_free.n_macro_cells(); ++mcell)
        for (unsigned int v = 0; v < this->matrix_free.n_components_filled(mcell); ++v)
          {
            typename DoFHandler<dim>::active_cell_iterator cell =
              this->matrix_free.get_cell_iterator(mcell, v, 2);
            cell_rhs = 0;

            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell->face(face)->at_boundary())
                  {
                    if (this->boundary->symmetry.find(cell->face(face)->boundary_id()) ==
                        this->boundary->symmetry.end())
                      {
                        fe_face_values.reinit(cell, face);
                        fe_face_values.get_function_gradients(this->solution.block(0),
                                                              local_gradients);

                        for (unsigned int i = 0; i < this->fe->dofs_per_cell; ++i)
                          {
                            for (unsigned int q = 0;
                                 q < fe_face_values.get_quadrature().size();
                                 ++q)
                              {
                                cell_rhs(i) += ((fe_face_values.shape_value(i, q) *
                                                 fe_face_values.normal_vector(q) *
                                                 artificial_viscosities[mcell][v] *
                                                 local_gradients[q]) *
                                                fe_face_values.JxW(q));
                              }
                          }
                      }
                  }
              }

            cell->get_dof_indices(local_dof_indices);
            this->constraints.distribute_local_to_global(cell_rhs,
                                                         local_dof_indices,
                                                         this->system_rhs);
          }
      this->system_rhs.compress(VectorOperation::add);
    }


  // solve linear system with Bicgstab (non-symmetric system!)
  unsigned int n_iterations     = 0;
  double       initial_residual = 0.;
  try
    {
      ReductionControl control(30, 0.05 * this->parameters.tol_nl_iteration, 1e-8);
      SolverBicgstab<LinearAlgebra::distributed::Vector<double>>::AdditionalData
        bicg_data;
      bicg_data.exact_residual = false;
      SolverBicgstab<LinearAlgebra::distributed::Vector<double>> solver(control,
                                                                        bicg_data);
      increment = 0;
      solver.solve(matrix, increment, rhs, preconditioner);
      n_iterations     = control.last_step();
      initial_residual = control.initial_value();
    }
  catch (const SolverControl::NoConvergence &)
    {
      // GMRES is typically slower but much more robust
      ReductionControl control(3000, 0.05 * this->parameters.tol_nl_iteration, 1e-8);
      SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver(control);
      solver.solve(matrix, increment, rhs, preconditioner);
      n_iterations = 30 + control.last_step();
    }
  if (!this->parameters.do_iteration)
    this->pcout << "  Concentration advance: advect [" << initial_residual << "/"
                << n_iterations << "]";

  this->constraints.distribute(increment);
  this->solution.block(0) += increment;
  this->solution.block(0).update_ghost_values();
}



// @sect4{LevelSetOKZSolver::compute_normal}
template <int dim>
void
LevelSetOKZSolver<dim>::compute_normal(const bool fast_computation)
{
  // This function computes the normal from a projection of $\nabla C$ onto
  // the space of linear finite elements (with some small damping)

  TimerOutput::Scope timer(*this->timer, "LS compute normal.");

  // compute right hand side
  this->normal_vector_rhs = 0;
#define OPERATION(c_degree, u_degree)                                     \
  this->matrix_free.cell_loop(                                            \
    &LevelSetOKZSolver<dim>::template local_compute_normal_rhs<c_degree>, \
    this,                                                                 \
    this->normal_vector_rhs,                                              \
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
              compute_curvature_vmult(this->normal_vector_rhs.block(block),
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
                                     const bool)
{
  // This function assembles and solves for a given profile using the approach
  // described in the paper by Olsson, Kreiss, and Zahedi.

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

      TimerOutput::Scope timer(*this->timer, "LS reinitialization step.");

      // compute right hand side
      LinearAlgebra::distributed::Vector<double> &rhs = this->system_rhs.block(0);
      LinearAlgebra::distributed::Vector<double> &increment =
        this->solution_update.block(0);
      rhs = 0;

      if (tau < actual_diff_steps)
        {
#define OPERATION(c_degree, u_degree)                                         \
  this->matrix_free.cell_loop(                                                \
    &LevelSetOKZSolver<dim>::template local_reinitialize_rhs<c_degree, true>, \
    this,                                                                     \
    rhs,                                                                      \
    this->solution.block(0))

          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
        }
      else
        {
#define OPERATION(c_degree, u_degree)                                          \
  this->matrix_free.cell_loop(                                                 \
    &LevelSetOKZSolver<dim>::template local_reinitialize_rhs<c_degree, false>, \
    this,                                                                      \
    rhs,                                                                       \
    this->solution.block(0))

          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
        }

      // solve linear system
      {
        ReinitializationMatrix<dim> matrix(*this, tau < actual_diff_steps);
        increment = 0;

        // reduce residual by 1e-6. To obtain good interface shapes, it is
        // essential that this tolerance is relative to the rhs
        // (ReductionControl steered solver, last argument determines the
        // solver)
        ReductionControl solver_control(2000, 1e-50, 1e-6);
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
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

      this->solution.block(0) += increment;
      this->solution.block(0).update_ghost_values();

      // check residual
      const double update_norm = increment.l2_norm();
      if (update_norm < 1e-6)
        break;

      if (!this->parameters.do_iteration && tau < actual_diff_steps + stab_steps - 1)
        this->pcout << " + ";
    }

  if (!this->parameters.do_iteration)
    this->pcout << ")" << std::endl << std::flush;
}



template class LevelSetOKZSolver<2>;
template class LevelSetOKZSolver<3>;
