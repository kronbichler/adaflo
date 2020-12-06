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


#ifndef __adaflo_level_set_okz_h
#define __adaflo_level_set_okz_h

#include <adaflo/diagonal_preconditioner.h>
#include <adaflo/level_set_base.h>


using namespace dealii;


// forward declarations
class BlockMatrixExtension;
class BlockILUExtension;
namespace AssemblyData
{
  struct Data;
}



template <int dim>
class LevelSetOKZSolver : public LevelSetBaseAlgorithm<dim>
{
public:
  LevelSetOKZSolver(const FlowParameters &parameters, Triangulation<dim> &triangulation);
  virtual ~LevelSetOKZSolver()
  {}

  void
  advance_concentration_vmult(
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src) const;

  void
  compute_normal_vmult(LinearAlgebra::distributed::BlockVector<double> &      dst,
                       const LinearAlgebra::distributed::BlockVector<double> &sr) const;

  void
  compute_normal_vmult(LinearAlgebra::distributed::BlockVector<float> &      dst,
                       const LinearAlgebra::distributed::BlockVector<float> &sr) const;

  void
  compute_curvature_vmult(LinearAlgebra::distributed::Vector<double> &      dst,
                          const LinearAlgebra::distributed::Vector<double> &srcc,
                          const bool apply_diffusion) const;

  virtual void
  initialize_data_structures();

  virtual void
  transform_distance_function(LinearAlgebra::distributed::Vector<double> &vector) const;

private:
  // compute the force term and variable density/viscosity for the
  // Navier--Stokes equations
  virtual void
  compute_force();

  // advection step
  virtual void
  advance_concentration();

  // computes normal direction vector by projection of level set gradient (not
  // scaled to have norm 1!)
  virtual void
  compute_normal(const bool fast_computation);

  // computes curvature by projecting the divergence of the normal vector
  // (scaled to norm 1 now)
  virtual void
  compute_curvature(const bool diffuse_large_values = false);

  // performs reinitialization
  virtual void
  reinitialize(const unsigned int stab_steps,
               const unsigned int diff_steps                              = 0,
               const bool         diffuse_cells_with_large_curvature_only = false);

  virtual void
  compute_heaviside();

  // matrix-free worker operations for various operations
  template <int ls_degree, int velocity_degree>
  void
  local_compute_force(const MatrixFree<dim, double> &                   data,
                      LinearAlgebra::distributed::Vector<double> &      dst,
                      const LinearAlgebra::distributed::Vector<double> &src,
                      const std::pair<unsigned int, unsigned int> &     cell_range);

  template <int ls_degree, int velocity_degree>
  void
  local_advance_concentration(
    const MatrixFree<dim, double> &                   data,
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const;

  template <int ls_degree, int velocity_degree>
  void
  local_advance_concentration_rhs(
    const MatrixFree<dim, double> &                   data,
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range);
  template <int ls_degree, typename Number>
  void
  local_compute_normal(const MatrixFree<dim, Number> &                        data,
                       LinearAlgebra::distributed::BlockVector<Number> &      dst,
                       const LinearAlgebra::distributed::BlockVector<Number> &src,
                       const std::pair<unsigned int, unsigned int> &cell_range) const;
  template <int ls_degree>
  void
  local_compute_normal_rhs(const MatrixFree<dim, double> &                   data,
                           LinearAlgebra::distributed::BlockVector<double> & dst,
                           const LinearAlgebra::distributed::Vector<double> &src,
                           const std::pair<unsigned int, unsigned int> &cell_range) const;

  // diffusion_setting: 0: both terms, 1: only mass, 2: only diffusion
  template <int ls_degree, int diffusion_setting>
  void
  local_compute_curvature(const MatrixFree<dim, double> &                   data,
                          LinearAlgebra::distributed::Vector<double> &      dst,
                          const LinearAlgebra::distributed::Vector<double> &src,
                          const std::pair<unsigned int, unsigned int> &cell_range) const;
  template <int ls_degree>
  void
  local_compute_curvature_rhs(
    const MatrixFree<dim, double> &                   data,
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const;

  void
  local_projection_matrix(
    const MatrixFree<dim, double> &                                   data,
    std::shared_ptr<Threads::ThreadLocalStorage<AssemblyData::Data>> &scratch,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &cell_range);

  template <int ls_degree>
  void
  local_projection_matrix(
    const MatrixFree<dim, double> &                                   data,
    std::shared_ptr<Threads::ThreadLocalStorage<AssemblyData::Data>> &scratch,
    const std::pair<unsigned int, unsigned int> &                     cell_range);

  AlignedVector<VectorizedArray<double>>                 artificial_viscosities;
  AlignedVector<Tensor<1, dim, VectorizedArray<double>>> evaluated_convection;
  bool                                                   first_reinit_step;
  double                                                 global_max_velocity;
  DiagonalPreconditioner<double>                         preconditioner;

  // In case we can better combine float/double solvers at some point...
  MatrixFree<dim, float>                matrix_free_float;
  AlignedVector<VectorizedArray<float>> cell_diameters_float;
  // GrowingVectorMemory<LinearAlgebra::distributed::BlockVector<float> >
  // vectors_normal; DiagonalPreconditioner<float> preconditioner_float;

  std::shared_ptr<BlockMatrixExtension> projection_matrix;
  std::shared_ptr<BlockILUExtension>    ilu_projection_matrix;
};


#endif
