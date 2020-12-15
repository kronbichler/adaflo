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


#ifndef __adaflo_level_set_compute_normal_h
#define __adaflo_level_set_compute_normal_h

#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <adaflo/block_matrix_extension.h>
#include <adaflo/diagonal_preconditioner.h>
#include <adaflo/navier_stokes.h>


using namespace dealii;

/**
 * Parameters of the advection-concentration operator.
 */
struct LevelSetOKZSolverComputeNormalParameter
{
  /**
   * TODO
   */
  unsigned int dof_index_ls;

  /**
   * TODO
   */
  unsigned int dof_index_normal;

  /**
   * TODO
   */
  unsigned int quad_index;

  /**
   * TODO
   */
  double epsilon;

  /**
   * TODO
   */
  bool approximate_projections;
  /**
   * TODO
   */
  double damping_scale_factor = 4.0;
};

template <int dim>
class LevelSetOKZSolverComputeNormal
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  LevelSetOKZSolverComputeNormal(
    BlockVectorType &                              normal_vector_field,
    BlockVectorType &                              normal_vector_rhs,
    VectorType &                                   solution,
    const AlignedVector<VectorizedArray<double>> & cell_diameters,
    const double &                                 epsilon_used,
    const double &                                 minimal_edge_length,
    const AffineConstraints<double> &              constraints_normals,
    const LevelSetOKZSolverComputeNormalParameter &parameters,
    const MatrixFree<dim> &                        matrix_free,
    const DiagonalPreconditioner<double> &         preconditioner,
    const std::shared_ptr<BlockMatrixExtension> &  projection_matrix,
    const std::shared_ptr<BlockILUExtension> &     ilu_projection_matrix);

  virtual void
  compute_normal(const bool fast_computation);

  void
  compute_normal_vmult(BlockVectorType &dst, const BlockVectorType &sr) const;

private:
  template <int ls_degree, typename Number>
  void
  local_compute_normal(const MatrixFree<dim, Number> &                        data,
                       LinearAlgebra::distributed::BlockVector<Number> &      dst,
                       const LinearAlgebra::distributed::BlockVector<Number> &src,
                       const std::pair<unsigned int, unsigned int> &cell_range) const;

  template <int ls_degree>
  void
  local_compute_normal_rhs(const MatrixFree<dim, double> &              data,
                           BlockVectorType &                            dst,
                           const VectorType &                           src,
                           const std::pair<unsigned int, unsigned int> &cell_range) const;

  /**
   * Parameters
   */
  const LevelSetOKZSolverComputeNormalParameter parameters; // [i]

  /**
   * Vector section
   */
  BlockVectorType & normal_vector_field; // [o]
  BlockVectorType & normal_vector_rhs;   // [-]
  const VectorType &vel_solution;        // [i]

  /**
   * MatrixFree
   */
  const MatrixFree<dim> &          matrix_free;         // [i]
  const AffineConstraints<double> &constraints_normals; // [i]

  /**
   * Physics section
   */
  const AlignedVector<VectorizedArray<double>> &cell_diameters;      // [i]
  const double &                                epsilon_used;        // [i]
  const double &                                minimal_edge_length; // [i]

  /**
   * Solver section
   */
  const DiagonalPreconditioner<double> &       preconditioner;        // [i]
  const std::shared_ptr<BlockMatrixExtension> &projection_matrix;     // [i]
  const std::shared_ptr<BlockILUExtension> &   ilu_projection_matrix; // [i]
};

#endif
