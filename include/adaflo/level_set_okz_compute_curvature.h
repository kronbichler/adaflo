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


#ifndef __adaflo_level_compute_curvature_h
#define __adaflo_level_compute_curvature_h

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <adaflo/block_matrix_extension.h>
#include <adaflo/diagonal_preconditioner.h>
#include <adaflo/navier_stokes.h>
#include <adaflo/time_stepping.h>
#include <adaflo/util.h>


namespace adaflo
{
  using namespace dealii;

  /**
   * Parameters of the advection-concentration operator.
   */
  struct LevelSetOKZSolverComputeCurvatureParameter
  {
    /**
     * TODO
     */
    unsigned int dof_index_ls;

    /**
     * TODO
     */
    unsigned int dof_index_curvature;

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
    bool curvature_correction;
  };

  template <int dim>
  class LevelSetOKZSolverComputeCurvature
  {
  public:
    LevelSetOKZSolverComputeCurvature(
      const AlignedVector<VectorizedArray<double>> &         cell_diameters,
      const LinearAlgebra::distributed::BlockVector<double> &normal_vector_field,
      const AffineConstraints<double> &                      constraints_curvature,
      const AffineConstraints<double> &                      constraints,
      const double &                                         epsilon_used,
      LinearAlgebra::distributed::Vector<double> &           system_rhs,
      const LevelSetOKZSolverComputeCurvatureParameter &     parameters,
      LinearAlgebra::distributed::Vector<double> &           solution_curvature,
      const LinearAlgebra::distributed::Vector<double> &     solution_ls,
      const MatrixFree<dim> &                                matrix_free,
      const DiagonalPreconditioner<double> &                 preconditioner,
      std::shared_ptr<BlockMatrixExtension> &                projection_matrix,
      std::shared_ptr<BlockILUExtension> &                   ilu_projection_matrix);

    virtual ~LevelSetOKZSolverComputeCurvature() = default;

    virtual void
    compute_curvature(const bool diffuse_large_values = false);

    void
    compute_curvature_vmult(LinearAlgebra::distributed::Vector<double> &      dst,
                            const LinearAlgebra::distributed::Vector<double> &srcc,
                            const bool apply_diffusion) const;

  private:
    // diffusion_setting: 0: both terms, 1: only mass, 2: only diffusion
    template <int ls_degree, int diffusion_setting>
    void
    local_compute_curvature(
      const MatrixFree<dim, double> &                   data,
      LinearAlgebra::distributed::Vector<double> &      dst,
      const LinearAlgebra::distributed::Vector<double> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;
    template <int ls_degree>
    void
    local_compute_curvature_rhs(
      const MatrixFree<dim, double> &                   data,
      LinearAlgebra::distributed::Vector<double> &      dst,
      const LinearAlgebra::distributed::Vector<double> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    /**
     * Parameters
     */
    const LevelSetOKZSolverComputeCurvatureParameter parameters; // [i]

    /**
     * Vector section
     */
    LinearAlgebra::distributed::Vector<double> &           solution_curvature;  // [i]
    LinearAlgebra::distributed::Vector<double> &           rhs;                 // [-]
    const LinearAlgebra::distributed::Vector<double> &     solution_ls;         // [i]
    const LinearAlgebra::distributed::BlockVector<double> &normal_vector_field; // [i]

    /**
     * MatrixFree
     */
    const MatrixFree<dim> &          matrix_free;           // [i]
    const AffineConstraints<double> &constraints_curvature; // [i]
    const AffineConstraints<double> &constraints;           // [i]

    /**
     * Physics section
     */
    const AlignedVector<VectorizedArray<double>> &cell_diameters; // [i]
    const double &                                epsilon_used;   // [i]

    /**
     * Solver section
     */
    const DiagonalPreconditioner<double> & preconditioner;        // [i]
    std::shared_ptr<BlockMatrixExtension> &projection_matrix;     // [i]
    std::shared_ptr<BlockILUExtension> &   ilu_projection_matrix; // [i]
  };
} // namespace adaflo

#endif
