// --------------------------------------------------------------------------
//
// Copyright (C) 2011 - 2016 by the adaflo authors
//
// This file is part of the adaflow library.
//
// The adaflo library is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.  The full text of the
// license can be found in the file LICENSE at the top level of the adaflo
// distribution.
//
// --------------------------------------------------------------------------

#ifndef __adaflo_navier_stokes_matrix_h
#define __adaflo_navier_stokes_matrix_h

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <adaflo/parameters.h>
#include <adaflo/time_stepping.h>


using namespace dealii;


// This collection of variables distinguishes local assembly operations in the
// Navier-Stokes matrix. Includes vmult (cell-wise matrix-vector product),
// residual (i.e., evaluation of the nonlinear Navier-Stokes operator), and
// just a vmult on the velocity subsystem
namespace NavierStokesOps
{
  const int vmult          = 0;
  const int residual       = 1;
  const int vmult_velocity = 2;
} // namespace NavierStokesOps



// This is the implementation of the matrix for the Navier-Stokes
// operation. It implements a matrix-vector product, nonlinear residual and
// contributions from old time steps. It also holds variable coefficients
// (densities, viscosities) and linearized velocities as needed for
// matrix-vector products.
template <int dim>
class NavierStokesMatrix
{
public:
  typedef std::pair<Tensor<1, dim, VectorizedArray<double>>,
                    Tensor<2, dim, VectorizedArray<double>>>
    velocity_stored;
  NavierStokesMatrix(
    const FlowParameters &                                 parameters,
    const unsigned int                                     dof_index_u,
    const unsigned int                                     dof_index_p,
    const unsigned int                                     quad_index_u,
    const unsigned int                                     quad_index_p,
    const LinearAlgebra::distributed::BlockVector<double> &solution_old,
    const LinearAlgebra::distributed::BlockVector<double> &solution_old_old);

  void
  initialize(const MatrixFree<dim> &matrix_free_in,
             const TimeStepping &   time_stepping_in,
             const bool             pressure_average_fix);

  void
  clear();

  unsigned int
  n_dofs_p() const
  {
    return matrix_free->get_dof_handler(dof_index_p).n_dofs();
  }

  void
  initialize_u_vector(LinearAlgebra::distributed::Vector<double> &vec) const
  {
    matrix_free->initialize_dof_vector(vec, dof_index_u);
  }

  void
  initialize_p_vector(LinearAlgebra::distributed::Vector<double> &vec) const
  {
    matrix_free->initialize_dof_vector(vec, dof_index_p);
  }

  unsigned int
  n_dofs_u() const
  {
    return matrix_free->get_dof_handler(dof_index_u).n_dofs();
  }

  unsigned int
  pressure_degree() const
  {
    return matrix_free->get_dof_handler(dof_index_p).get_fe().degree;
  }

  const MatrixFree<dim> &
  get_matrix_free() const
  {
    Assert(matrix_free != 0, ExcNotInitialized());
    return *matrix_free;
  }

  // If the pressure is only determined up to a constant, we solve modified
  // linear systems instead of constraining the degrees of freedom directly in
  // the matrix entries by forcing the mean value of the pressure to zero
  // while solving the linear system. Since we might be using FE_Q_DG0
  // elements which have two zero modes, need two such vectors, each
  // corresponding to the mass weights for the respective zero mode.
  void
  apply_pressure_average_projection(
    LinearAlgebra::distributed::Vector<double> &vector) const;

  void
  apply_pressure_shift(const double                                shift,
                       LinearAlgebra::distributed::Vector<double> &pressure) const;

  void
  vmult(LinearAlgebra::distributed::BlockVector<double> &      dst,
        const LinearAlgebra::distributed::BlockVector<double> &src) const;

  void
  residual(LinearAlgebra::distributed::BlockVector<double> &      residual_vector,
           const LinearAlgebra::distributed::BlockVector<double> &src,
           const LinearAlgebra::distributed::BlockVector<double> &user_rhs) const;

  void
  divergence_vmult_add(LinearAlgebra::distributed::Vector<double> &      dst,
                       const LinearAlgebra::distributed::Vector<double> &src,
                       const bool weight_by_viscosity = false) const;

  void
  velocity_vmult(LinearAlgebra::distributed::Vector<double> &      dst,
                 const LinearAlgebra::distributed::Vector<double> &src) const;

  void
  pressure_poisson_vmult(LinearAlgebra::distributed::Vector<double> &      dst,
                         const LinearAlgebra::distributed::Vector<double> &src) const;

  void
  pressure_mass_vmult(LinearAlgebra::distributed::Vector<double> &      dst,
                      const LinearAlgebra::distributed::Vector<double> &src) const;

  void
  pressure_convdiff_vmult(LinearAlgebra::distributed::Vector<double> &      dst,
                          const LinearAlgebra::distributed::Vector<double> &src) const;

  // fix the linearization point in additional data points (used for AMG
  // preconditioners that should not get out of sync with the matrix they are
  // based upon)
  void
  fix_linearization_point() const;

  // access to density and viscosity fields
  const VectorizedArray<double> *
  begin_densities(const unsigned int macro_cell) const;
  VectorizedArray<double> *
  begin_densities(const unsigned int macro_cell);

  const VectorizedArray<double> *
  begin_viscosities(const unsigned int macro_cell) const;
  VectorizedArray<double> *
  begin_viscosities(const unsigned int macro_cell);

  const velocity_stored *
  begin_linearized_velocities(const unsigned int macro_cell) const;
  bool
  use_variable_coefficients() const;

  std::size_t
  memory_consumption() const;
  void
  print_memory_consumption(std::ostream &stream) const;

  // Returns statistics of the sum of all matrix-vector product since the last
  // call to this function in terms of minimum, average, and maximum of times
  // as seen over all MPI processes (first argument) as well as the number of
  // times the mat-vec was invoked. After returning the data, the internal
  // counters are re-set, so in case you wish global statistics, make sure to
  // accumulate the results of this call.
  //
  // Note: This is a collective call and must be invoked on all processors.
  std::pair<Utilities::MPI::MinMaxAvg, unsigned int>
  get_matvec_statistics() const;

  // Return the underlying time stepping structure for querying the current
  // time and other information
  const TimeStepping &
  get_time_stepping() const
  {
    return *time_stepping;
  }

private:
  // residual and vmult computations use the same function to minimize risk
  // for errors when changing the terms. Template argument LocalOp
  // distinguishes them (see above for the various values).
  template <int degree_p, typename VectorType, int LocalOp>
  void
  local_operation(const MatrixFree<dim> &                      data,
                  VectorType &                                 dst,
                  const VectorType &                           src,
                  const std::pair<unsigned int, unsigned int> &cell_range) const;

  template <int degree_p, bool weight_by_viscosity>
  void
  local_divergence(const MatrixFree<dim> &                           data,
                   LinearAlgebra::distributed::Vector<double> &      dst,
                   const LinearAlgebra::distributed::Vector<double> &src,
                   const std::pair<unsigned int, unsigned int> &     cell_range) const;

  template <int degree_p>
  void
  local_pressure_poisson(const MatrixFree<dim> &                           data,
                         LinearAlgebra::distributed::Vector<double> &      dst,
                         const LinearAlgebra::distributed::Vector<double> &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;
  template <int degree_p>
  void
  local_pressure_mass(const MatrixFree<dim> &                           data,
                      LinearAlgebra::distributed::Vector<double> &      dst,
                      const LinearAlgebra::distributed::Vector<double> &src,
                      const std::pair<unsigned int, unsigned int> &     cell_range) const;

  template <int degree_p>
  void
  local_pressure_convdiff(const MatrixFree<dim> &                           data,
                          LinearAlgebra::distributed::Vector<double> &      dst,
                          const LinearAlgebra::distributed::Vector<double> &src,
                          const std::pair<unsigned int, unsigned int> &cell_range) const;

  template <int degree_p>
  void
  local_pressure_mass_weight(
    const MatrixFree<dim> &                     data,
    LinearAlgebra::distributed::Vector<double> &dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &cell_range) const;

  const MatrixFree<dim> *matrix_free;
  const TimeStepping *   time_stepping;
  const FlowParameters & parameters;

public:
  const unsigned int dof_index_u;
  const unsigned int dof_index_p;
  const unsigned int quad_index_u;
  const unsigned int quad_index_p;

private:
  mutable AlignedVector<VectorizedArray<double>> variable_densities;
  mutable AlignedVector<VectorizedArray<double>> variable_viscosities;
  mutable AlignedVector<velocity_stored>         linearized_velocities;

  mutable AlignedVector<VectorizedArray<double>> variable_densities_preconditioner;
  mutable AlignedVector<VectorizedArray<double>> variable_viscosities_preconditioner;
  mutable AlignedVector<velocity_stored>         linearized_velocities_preconditioner;

  const LinearAlgebra::distributed::BlockVector<double> &solution_old;
  const LinearAlgebra::distributed::BlockVector<double> &solution_old_old;

  LinearAlgebra::distributed::Vector<double> pressure_constant_modes[2];
  LinearAlgebra::distributed::Vector<double> pressure_constant_mode_weights[2];
  double                                     inverse_pressure_average_weights[2];

  mutable std::pair<unsigned int, double> matvec_timer;
};



/* ---------------------------- Inline functions ------------------------- */



// access to density and viscosity fields
template <int dim>
inline const VectorizedArray<double> *
NavierStokesMatrix<dim>::begin_densities(const unsigned int macro_cell) const
{
  AssertIndexRange(macro_cell, matrix_free->n_cell_batches());
  AssertDimension(variable_densities.size(),
                  matrix_free->n_cell_batches() *
                    matrix_free->get_n_q_points(quad_index_u));
  return &variable_densities[matrix_free->get_n_q_points(quad_index_u) * macro_cell];
}



template <int dim>
inline VectorizedArray<double> *
NavierStokesMatrix<dim>::begin_densities(const unsigned int macro_cell)
{
  AssertIndexRange(macro_cell, matrix_free->n_cell_batches());
  AssertDimension(variable_densities.size(),
                  matrix_free->n_cell_batches() *
                    matrix_free->get_n_q_points(quad_index_u));
  return &variable_densities[matrix_free->get_n_q_points(quad_index_u) * macro_cell];
}



template <int dim>
inline const VectorizedArray<double> *
NavierStokesMatrix<dim>::begin_viscosities(const unsigned int macro_cell) const
{
  AssertIndexRange(macro_cell, matrix_free->n_cell_batches());
  AssertDimension(variable_viscosities.size(),
                  matrix_free->n_cell_batches() *
                    matrix_free->get_n_q_points(quad_index_u));
  return &variable_viscosities[matrix_free->get_n_q_points(quad_index_u) * macro_cell];
}



template <int dim>
inline VectorizedArray<double> *
NavierStokesMatrix<dim>::begin_viscosities(const unsigned int macro_cell)
{
  AssertIndexRange(macro_cell, matrix_free->n_cell_batches());
  AssertDimension(variable_viscosities.size(),
                  matrix_free->n_cell_batches() *
                    matrix_free->get_n_q_points(quad_index_u));
  return &variable_viscosities[matrix_free->get_n_q_points(quad_index_u) * macro_cell];
}



template <int dim>
inline bool
NavierStokesMatrix<dim>::use_variable_coefficients() const
{
  return variable_viscosities.size() > 0;
}



template <int dim>
inline const typename NavierStokesMatrix<dim>::velocity_stored *
NavierStokesMatrix<dim>::begin_linearized_velocities(const unsigned int macro_cell) const
{
  if (linearized_velocities.size() == 0)
    return 0;

  AssertIndexRange(macro_cell, matrix_free->n_cell_batches());
  AssertDimension(linearized_velocities.size(),
                  matrix_free->n_cell_batches() *
                    matrix_free->get_n_q_points(quad_index_u));
  return &linearized_velocities[matrix_free->get_n_q_points(quad_index_u) * macro_cell];
}


#endif
