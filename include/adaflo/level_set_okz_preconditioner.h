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
mass_matrix_diagonal(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
                     const AffineConstraints<Number> &hanging_node_constraints,
                     const unsigned int               dof_index,
                     DiagonalPreconditioner<double> & preconditioner)
{
  LinearAlgebra::distributed::Vector<Number> diagonal;
  matrix_free.initialize_dof_vector(diagonal, dof_index);

  const auto &dof_handler = matrix_free.get_dof_handler(dof_index);
  const auto &fe          = dof_handler.get_fe();
  const auto &mapping     = *matrix_free.get_mapping_info().mapping;

  const unsigned int concentration_subdivisions = fe.tensor_degree();

  {
    diagonal = 0;
    QIterated<dim> quadrature(QGauss<1>(2), concentration_subdivisions);
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
              for (unsigned int q = 0; q < quadrature.size(); ++q)
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

#endif
