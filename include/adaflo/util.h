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

#ifndef __adaflo_util_h_
#define __adaflo_util_h_

#include <deal.II/distributed/tria.h>

#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

/**
 * Return the communicator of an arbitrary mesh type. Use this function to be
 * able to code that is independent if a serial or a parallel triangulation is
 * used.
 */
template <typename MeshType>
MPI_Comm
get_communicator(const MeshType &mesh)
{
  const auto *tria_parallel = dynamic_cast<
    const parallel::TriangulationBase<MeshType::dimension, MeshType::space_dimension> *>(
    &(mesh.get_triangulation()));

  return tria_parallel != nullptr ? tria_parallel->get_communicator() : MPI_COMM_SELF;
}

/**
 * Return the locally-owned subdomain of an arbitrary mesh type. Use this
 * function to be able to code that is independent if a serial or a parallel
 * triangulation is used.
 */
template <typename MeshType>
unsigned int
locally_owned_subdomain(const MeshType &mesh)
{
  const auto *tria_parallel = dynamic_cast<
    const parallel::TriangulationBase<MeshType::dimension, MeshType::space_dimension> *>(
    &(mesh.get_triangulation()));

  return tria_parallel != nullptr ? tria_parallel->locally_owned_subdomain() : 0;
}

template <int dim>
void
compute_cell_diameters(const MatrixFree<dim, double> &         matrix_free,
                       const unsigned int                      dof_index,
                       AlignedVector<VectorizedArray<double>> &cell_diameters,
                       double &                                cell_diameter_min,
                       double &                                cell_diameter_max)
{
  cell_diameters.resize(matrix_free.n_cell_batches());

  cell_diameter_min = std::numeric_limits<double>::max();
  cell_diameter_max = 0.0;

  // to find the cell diameters, we compute the maximum and minimum eigenvalue
  // of the Jacobian transformation from the unit to the real cell. We check
  // all face centers and the center of the cell and take the respective
  // minimum and maximum there to cover most of the cell geometry
  std::vector<Point<dim>> face_centers;
  {
    Point<dim> center;
    for (unsigned int d = 0; d < dim; ++d)
      center[d] = 0.5;
    for (unsigned int d = 0; d < dim; ++d)
      {
        Point<dim> p1 = center;
        p1[d]         = 0;
        face_centers.push_back(p1);
        p1[d] = 1.;
        face_centers.push_back(p1);
      }
    face_centers.push_back(center);
  }

  const auto &dof_handler   = matrix_free.get_dof_handler(dof_index);
  const auto &triangulation = dof_handler.get_triangulation();

  LAPACKFullMatrix<double> mat(dim, dim);
  FEValues<dim>            fe_values(*matrix_free.get_mapping_info().mapping,
                          dof_handler.get_fe(),
                          Quadrature<dim>(face_centers),
                          update_jacobians);
  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      VectorizedArray<double> diameter = VectorizedArray<double>();
      for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
        {
          typename DoFHandler<dim>::active_cell_iterator dcell =
            matrix_free.get_cell_iterator(cell, v, dof_index);
          fe_values.reinit(dcell);
          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              mat = 0;
              for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int e = 0; e < dim; ++e)
                  mat(d, e) = fe_values.jacobian(q)[d][e];
              mat.compute_eigenvalues();
              for (unsigned int d = 0; d < dim; ++d)
                {
                  diameter[v] = std::max(diameter[v], std::abs(mat.eigenvalue(d)));
                  cell_diameter_min =
                    std::min(cell_diameter_min, std::abs(mat.eigenvalue(d)));
                }
            }
          if (1U + dcell->level() == triangulation.n_global_levels())
            cell_diameter_max = std::max(diameter[v], cell_diameter_max);
        }
      cell_diameters[cell] = diameter;
    }
  cell_diameter_min =
    -Utilities::MPI::max(-cell_diameter_min, get_communicator(triangulation));
  cell_diameter_max =
    Utilities::MPI::max(cell_diameter_max, get_communicator(triangulation));
}

/**
 * If dim == 1, convert a VectorizedArray<number> to a vector (rank 1 tensor).
 * This function is useful to obtain equal, vector-valued return types of
 * FEEvaluation-operations for dim == 1 and dim > 1.
 */
template <int dim, typename VectorizedArrayType = VectorizedArray<double>>
static Tensor<1, dim, VectorizedArrayType>
convert_to_vector(const VectorizedArrayType &in)
{
  AssertThrow(dim == 1, ExcMessage("This operation is not permitted for dim>1."))

    Tensor<1, dim, VectorizedArrayType>
      vec;

  for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
    vec[0][v] = in[v];

  return vec;
}

/**
 * This function overloads the previous convert_to_vector function, when
 * the input argument is already given as a rank 1 tensor.
 */
template <int dim, typename VectorizedArrayType = VectorizedArray<double>>
static Tensor<1, dim, VectorizedArrayType>
convert_to_vector(const Tensor<1, dim, VectorizedArrayType> &in)
{
  return in;
}

/**
 * If dim == 1, convert a tensor of rank-1 to a tensor of rank. This function
 * is useful to obtain equal, tensor-rank-valued return types of
 * FEEvaluation-operations for dim == 1 and dim > 1.
 */
template <int rank_, int dim, typename VectorizedArrayType = VectorizedArray<double>>
static Tensor<rank_, dim, VectorizedArrayType>
convert_to_tensor(const Tensor<rank_ - 1, dim, VectorizedArrayType> &in)
{
  AssertThrow(dim == 1, ExcMessage("This operation is not permitted for dim>1."))

    if (rank_ == 2)
  {
    Tensor<2, dim, VectorizedArrayType> tens;

    for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
      tens[0][0][v] = in[0][v];
    return tens;
  }
  else AssertThrow(false, ExcNotImplemented());
}

/**
 * This function overloads the previous convert_to_tensor function, when the input
 * tensor already has the desired tensor rank.
 */
template <int rank_, int dim, typename VectorizedArrayType = VectorizedArray<double>>
static Tensor<rank_, dim, VectorizedArrayType>
convert_to_tensor(const Tensor<rank_, dim, VectorizedArrayType> &in)
{
  return in;
}

#endif
