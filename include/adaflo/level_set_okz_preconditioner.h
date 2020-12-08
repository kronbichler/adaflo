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


#ifndef __adaflo_level_set_okz_preconditioner_h
#define __adaflo_level_set_okz_preconditioner_h

#include <deal.II/matrix_free/matrix_free.h>

#include <adaflo/block_matrix_extension.h>
#include <adaflo/diagonal_preconditioner.h>

using namespace dealii;

template <int dim,
          typename Number,
          typename VectorizedArrayType = VectorizedArray<Number>>
inline void
initialize_mass_matrix_diagonal(
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
  const AffineConstraints<Number> &                   hanging_node_constraints,
  const unsigned int                                  dof_index,
  const unsigned int                                  quad_index,
  DiagonalPreconditioner<double> &                    preconditioner)
{
  LinearAlgebra::distributed::Vector<Number> diagonal;
  matrix_free.initialize_dof_vector(diagonal, dof_index);

  const auto &dof_handler = matrix_free.get_dof_handler(dof_index);
  const auto &fe          = dof_handler.get_fe();
  const auto &quadrature  = matrix_free.get_quadrature(quad_index);
  const auto &mapping     = *matrix_free.get_mapping_info().mapping;

  {
    diagonal = 0;
    FEValues<dim>  fe_values(mapping, fe, quadrature, update_values | update_JxW_values);
    Vector<Number> local_rhs(fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              Number value = 0;
              for (const auto q : fe_values.quadrature_point_indices())
                value += fe_values.shape_value(i, q) * fe_values.shape_value(i, q) *
                         fe_values.JxW(q);
              local_rhs(i) = value;
            }
          cell->get_dof_indices(local_dof_indices);
          hanging_node_constraints.distribute_local_to_global(local_rhs,
                                                              local_dof_indices,
                                                              diagonal);
        }
    diagonal.compress(VectorOperation::add);
    preconditioner.reinit(diagonal);
  }
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



template <int dim,
          typename Number,
          typename VectorizedArrayType = VectorizedArray<Number>>
void
initialize_projection_matrix(
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
  const AffineConstraints<Number> &                   constraints_normals,
  const unsigned int                                  dof_index,
  const unsigned int                                  quad_index,
  const unsigned int                                  concentration_subdivisions,
  const Number &                                      epsilon_used,
  const Number &                                      epsilon,
  const AlignedVector<VectorizedArrayType> &          cell_diameters,
  BlockMatrixExtension &                              projection_matrix,
  BlockILUExtension &                                 ilu_projection_matrix)
{
  const auto &dof_handler = matrix_free.get_dof_handler(dof_index);
  const auto &fe          = dof_handler.get_fe();
  // const auto & quadrature = matrix_free.get_quadrature(quad_index);
  QIterated<dim> quadrature(QGauss<1>(1), concentration_subdivisions);
  const auto &   mapping = *matrix_free.get_mapping_info().mapping;

  AssertThrow(fe.tensor_degree() == concentration_subdivisions, ExcNotImplemented());

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
               get_communicator(dof_handler));
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
        FEEvaluation<dim, -1, 0, 1, double> phi(data, 4, 2);
        AssemblyData::Data &                scratch = scratch_data->get();

        const VectorizedArray<double> min_diameter =
          make_vectorized_array(epsilon_used / epsilon);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            const VectorizedArray<double> damping =
              4. *
              Utilities::fixed_power<2>(
                std::max(min_diameter,
                         cell_diameters[cell] / static_cast<double>(fe.tensor_degree())));

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
                for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
                     ++v)
                  for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                    scratch.matrices[v](phi.get_shape_info().lexicographic_numbering[j],
                                        phi.get_shape_info().lexicographic_numbering[i]) =
                      phi.begin_dof_values()[j][v];
              }
            for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
              {
                typename DoFHandler<dim>::active_cell_iterator dcell =
                  matrix_free.get_cell_iterator(cell, v, 2);
                dcell->get_dof_indices(scratch.dof_indices);
                constraints_normals.distribute_local_to_global(
                  scratch.matrices[v],
                  scratch.dof_indices,
                  static_cast<TrilinosWrappers::SparseMatrix &>(projection_matrix));
              }
          }
      },
      scratch_local,
      dummy);
    projection_matrix.compress(VectorOperation::add);
    ilu_projection_matrix.initialize(projection_matrix);
  }
}

#endif
