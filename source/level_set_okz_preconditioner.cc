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

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <adaflo/level_set_okz_preconditioner.h>
#include <adaflo/util.h>



#define EXPAND_OPERATIONS(OPERATION)                                      \
  if (fe.reference_cell() != ReferenceCells::get_hypercube<dim>())        \
    {                                                                     \
      OPERATION(-1, -1);                                                  \
    }                                                                     \
  else                                                                    \
    {                                                                     \
      AssertThrow(ls_degree >= 1 && ls_degree <= 4, ExcNotImplemented()); \
      if (ls_degree == 1)                                                 \
        {                                                                 \
          OPERATION(1, 0);                                                \
        }                                                                 \
      else if (ls_degree == 2)                                            \
        {                                                                 \
          OPERATION(2, 0);                                                \
        }                                                                 \
      else if (ls_degree == 3)                                            \
        {                                                                 \
          OPERATION(3, 0);                                                \
        }                                                                 \
      else if (ls_degree == 4)                                            \
        {                                                                 \
          OPERATION(4, 0);                                                \
        }                                                                 \
    }

using namespace dealii;


template <int dim, typename Number, typename VectorizedArrayType>
void
adaflo::initialize_projection_matrix(
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
  const AffineConstraints<Number> &                   constraints_normals,
  const unsigned int                                  dof_index,
  const unsigned int                                  quad_index,
  const Number &                                      epsilon_used,
  const Number &                                      epsilon,
  const AlignedVector<VectorizedArrayType> &          cell_diameters,
  adaflo::BlockMatrixExtension &                      projection_matrix,
  adaflo::BlockILUExtension &                         ilu_projection_matrix)
{
  const auto &dof_handler = matrix_free.get_dof_handler(dof_index);
  const auto &fe          = dof_handler.get_fe();
  const auto &quadrature  = matrix_free.get_quadrature(quad_index);
  const auto &mapping     = *matrix_free.get_mapping_info().mapping;

  // create sparse matrix for projection systems.
  //
  // First off is the creation of a mask that only adds those entries of
  // FE_Q_iso_Q0 that are going to have a non-zero matrix entry -> this
  // ensures as compact a matrix as for Q1 on the fine mesh. To find them,
  // check terms in a mass matrix.
  Table<2, bool> dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);
  {
    FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
    fe_values.reinit(dof_handler.begin());
    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
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
    DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
    TrilinosWrappers::SparsityPattern csp;
    csp.reinit(dof_handler.locally_owned_dofs(),
               dof_handler.locally_owned_dofs(),
               relevant_dofs,
               dof_handler.get_communicator());
    std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);
          constraints_normals.add_entries_local_to_global(local_dof_indices,
                                                          csp,
                                                          false,
                                                          dof_mask);
        }
    csp.compress();
    projection_matrix.reinit(csp);
  }
  {
    AssemblyData::Data scratch_data(fe.dofs_per_cell);
    auto               scratch_local =
      std::make_shared<Threads::ThreadLocalStorage<AssemblyData::Data>>(scratch_data);
    unsigned int dummy = 0;
    matrix_free.template cell_loop<
      std::shared_ptr<Threads::ThreadLocalStorage<AssemblyData::Data>>,
      unsigned int>(
      [&](const auto &data, auto &scratch_data, const auto &, const auto cell_range) {
        const unsigned int ls_degree = fe.tensor_degree();

#define OPERATION(c_degree, u_degree)                                                   \
  const unsigned int n_q_points = c_degree == -1 ? 0 : 2 * c_degree;                    \
  FEEvaluation<dim, c_degree, n_q_points, 1, double> phi(data, dof_index, quad_index);  \
  AssemblyData::Data &                               scratch = scratch_data->get();     \
                                                                                        \
  const VectorizedArray<double> min_diameter =                                          \
    make_vectorized_array(epsilon_used / epsilon);                                      \
                                                                                        \
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)          \
    {                                                                                   \
      phi.reinit(cell);                                                                 \
      const VectorizedArray<double> damping =                                           \
        4. *                                                                            \
        Utilities::fixed_power<2>(                                                      \
          std::max(min_diameter,                                                        \
                   cell_diameters[cell] / static_cast<double>(fe.tensor_degree())));    \
                                                                                        \
      for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)                              \
        {                                                                               \
          for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)                          \
            phi.begin_dof_values()[j] = VectorizedArray<double>();                      \
          phi.begin_dof_values()[i] = 1.;                                               \
          phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);           \
          for (unsigned int q = 0; q < phi.n_q_points; ++q)                             \
            {                                                                           \
              phi.submit_value(phi.get_value(q), q);                                    \
              phi.submit_gradient(phi.get_gradient(q) * damping, q);                    \
            }                                                                           \
          phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);          \
          for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v) \
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)                        \
              scratch.matrices[v](phi.get_shape_info().lexicographic_numbering[j],      \
                                  phi.get_shape_info().lexicographic_numbering[i]) =    \
                phi.begin_dof_values()[j][v];                                           \
        }                                                                               \
      for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)     \
        {                                                                               \
          typename DoFHandler<dim>::active_cell_iterator dcell =                        \
            matrix_free.get_cell_iterator(cell, v, dof_index);                          \
          dcell->get_dof_indices(scratch.dof_indices);                                  \
          constraints_normals.distribute_local_to_global(                               \
            scratch.matrices[v],                                                        \
            scratch.dof_indices,                                                        \
            static_cast<TrilinosWrappers::SparseMatrix &>(projection_matrix));          \
        }                                                                               \
    }

        EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
      },
      scratch_local,
      dummy);
    projection_matrix.compress(VectorOperation::add);
    ilu_projection_matrix.initialize(projection_matrix);
  }
}
template void
adaflo::initialize_projection_matrix<1, double, VectorizedArray<double>>(
  const MatrixFree<1, double, VectorizedArray<double>> &matrix_free,
  const AffineConstraints<double> &                     constraints_normals,
  const unsigned int                                    dof_index,
  const unsigned int                                    quad_index,
  const double &                                        epsilon_used,
  const double &                                        epsilon,
  const AlignedVector<VectorizedArray<double>> &        cell_diameters,
  BlockMatrixExtension &                                projection_matrix,
  BlockILUExtension &                                   ilu_projection_matrix);

template void
adaflo::initialize_projection_matrix<2, double, VectorizedArray<double>>(
  const MatrixFree<2, double, VectorizedArray<double>> &matrix_free,
  const AffineConstraints<double> &                     constraints_normals,
  const unsigned int                                    dof_index,
  const unsigned int                                    quad_index,
  const double &                                        epsilon_used,
  const double &                                        epsilon,
  const AlignedVector<VectorizedArray<double>> &        cell_diameters,
  BlockMatrixExtension &                                projection_matrix,
  BlockILUExtension &                                   ilu_projection_matrix);

template void
adaflo::initialize_projection_matrix<3, double, VectorizedArray<double>>(
  const MatrixFree<3, double, VectorizedArray<double>> &matrix_free,
  const AffineConstraints<double> &                     constraints_normals,
  const unsigned int                                    dof_index,
  const unsigned int                                    quad_index,
  const double &                                        epsilon_used,
  const double &                                        epsilon,
  const AlignedVector<VectorizedArray<double>> &        cell_diameters,
  BlockMatrixExtension &                                projection_matrix,
  BlockILUExtension &                                   ilu_projection_matrix);
