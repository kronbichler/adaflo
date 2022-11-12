// --------------------------------------------------------------------------
//
// Copyright (C) 2009 - 2016 by the adaflo authors
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

#ifndef __adaflo_navier_stokes_h
#define __adaflo_navier_stokes_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/thread_local_storage.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <adaflo/flow_base_algorithm.h>
#include <adaflo/navier_stokes_matrix.h>
#include <adaflo/navier_stokes_preconditioner.h>
#include <adaflo/parameters.h>
#include <adaflo/time_stepping.h>

using namespace dealii;


// Indicate that Navier-Stokes solver has not converged.
DeclExceptionMsg(ExcNavierStokesNoConvergence,
                 "The Navier-Stokes solver did not converge.");

template <int dim>
class NavierStokes : public FlowBaseAlgorithm<dim>
{
public:
  unsigned int dof_index_u  = 0;
  unsigned int dof_index_p  = 1;
  unsigned int quad_index_u = 0;
  unsigned int quad_index_p = 1;

  NavierStokes(const Mapping<dim> &                              mapping,
               const FlowParameters &                            parameters,
               Triangulation<dim> &                              triangulation,
               TimerOutput *                                     external_timer = 0,
               std::shared_ptr<helpers::BoundaryDescriptor<dim>> boundary_descriptor =
                 std::shared_ptr<helpers::BoundaryDescriptor<dim>>());

  NavierStokes(const FlowParameters &                            parameters,
               Triangulation<dim> &                              triangulation,
               TimerOutput *                                     external_timer = 0,
               std::shared_ptr<helpers::BoundaryDescriptor<dim>> boundary_descriptor =
                 std::shared_ptr<helpers::BoundaryDescriptor<dim>>());

  virtual ~NavierStokes();

  std::pair<unsigned int, unsigned int>
  n_dofs() const;
  void
  print_n_dofs() const;

  const FiniteElement<dim> &
  get_fe_u() const;
  const FiniteElement<dim> &
  get_fe_p() const;

  const DoFHandler<dim> &
  get_dof_handler_u() const;
  const AffineConstraints<double> &
  get_constraints_u() const;
  const AffineConstraints<double> &
  get_hanging_node_constraints_u() const;

  const DoFHandler<dim> &
  get_dof_handler_p() const;
  const AffineConstraints<double> &
  get_constraints_p() const;
  const AffineConstraints<double> &
  get_hanging_node_constraints_p() const;

  AffineConstraints<double> &
  modify_constraints_u();
  AffineConstraints<double> &
  modify_constraints_p();

  void
  distribute_dofs();
  void
  initialize_data_structures();
  virtual void
  setup_problem(
    const Function<dim> &initial_velocity_field,
    const Function<dim> &initial_distance_function = Functions::ZeroFunction<dim>());
  void
  initialize_matrix_free(MatrixFree<dim> *  external_matrix_free = 0,
                         const unsigned int dof_index_u          = 0,
                         const unsigned int dof_index_p          = 1,
                         const unsigned int quad_index_u         = 0,
                         const unsigned int quad_index_p         = 1);

  void
  init_time_advance(const bool print_time_info = true);
  std::pair<unsigned int, unsigned int>
  evaluate_time_step();
  virtual std::pair<unsigned int, unsigned int>
  advance_time_step();

  virtual void
  output_solution(const std::string  output_base_name,
                  const unsigned int n_subdivisions = 0) const;

  /**
   * When solving a problem with boundary conditions that start at a non-zero
   * value but with an initial field that is all zero, one will in general
   * not get a good velocity field. This function can be used to create a
   * divergence-free velocity field by solving the stokes equations with the
   * given boundary values but without any external forces.
   */
  void
  compute_initial_stokes_field();

  /**
   * Calls VectorTools::interpolate for the pressure field. Since we might be
   * using FE_Q_DG0 elements where the usual interpolation does not make
   * sense, this class provides a seperate function for it.
   */
  void
  interpolate_pressure_field(
    const Function<dim> &                       pressure_function,
    LinearAlgebra::distributed::Vector<double> &pressure_vector) const;

  void
  assemble_preconditioner();

  void
  build_preconditioner();

  std::pair<unsigned int, double>
  solve_system(const double linear_tolerance);

  void
  vmult(LinearAlgebra::distributed::BlockVector<double> &      dst,
        const LinearAlgebra::distributed::BlockVector<double> &src) const;

  void
  refine_grid_pressure_based(const unsigned int max_grid_level,
                             const double       refine_fraction_of_cells  = 0.3,
                             const double       coarsen_fraction_of_cells = 0.05);

  // internally calls triangulation.prepare_coarsening_and_refinement
  void
  prepare_coarsening_and_refinement();

  void
  set_face_average_density(const typename Triangulation<dim>::cell_iterator &cell,
                           const unsigned int                                face,
                           const double                                      density);

  // Set a user defined material law to compute the viscous stress constribution. You have
  // to pass a lambda function, which computes @return the shear stress rate depending on the
  // @p velocity_gradient for a given cell @p cell_idx at the quadrature point @p quad_idx.
  // @p do_tangent=true means that the function is called from the tangent (vmult);
  // do_tangent=false means that the residual is computed.
  void
  set_user_defined_material(
    std::function<Tensor<2, dim, VectorizedArray<double>>(
      const Tensor<2, dim, VectorizedArray<double>> &velocity_gradient,
      const unsigned int                             cell_idx,
      const unsigned int                             quad_idx,
      const bool do_tangent)> my_user_defined_material);

  const FlowParameters &
  get_parameters() const;

  const NavierStokesMatrix<dim> &
  get_matrix() const
  {
    return navier_stokes_matrix;
  }

  NavierStokesMatrix<dim> &
  get_matrix()
  {
    return navier_stokes_matrix;
  }

  const LinearAlgebra::distributed::BlockVector<double> &
  get_system_rhs() const
  {
    return system_rhs;
  }

  bool
  get_update_preconditioner() const
  {
    return update_preconditioner;
  }

  // Computes the initial residual of the fluid field, including the part of
  // the residual that does not depend on the time step
  double
  compute_initial_residual(const bool usual_time_step = true);

  // Solves the nonlinear Navier-Stokes system by a Newton or Newton-like
  // iteration. This function expects that the initial residual is passed into
  // the function as an argument
  std::pair<unsigned int, unsigned int>
  solve_nonlinear_system(const double initial_residual);

  // return an estimate of the total memory consumption
  std::size_t
  memory_consumption() const;

  void
  print_memory_consumption(std::ostream &stream = std::cout) const;

  // vectors that are visible to the user
  LinearAlgebra::distributed::BlockVector<double> user_rhs;
  LinearAlgebra::distributed::BlockVector<double> solution, solution_old,
    solution_old_old, solution_update;

  TimeStepping time_stepping;

  // it is important to have most of the variables private so that all the
  // changes to internal data structures like the constraint matrix are
  // followed by the correct actions in assembly etc.

private:
  void
  apply_boundary_conditions();
  double
  compute_residual();

  FlowParameters parameters;

  /* MPI_Comm mpi_communicator;
   * probably won't need an extra copy of MPI_Comm in the
   * NavierStokes class as one can always get the communicator
   * easily by triangulation.get_communicator()
   */

  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  ConditionalOStream pcout;

  Triangulation<dim> &triangulation;


  const FESystem<dim> fe_u;
  const FESystem<dim> fe_p;

  DoFHandler<dim> dof_handler_u;
  DoFHandler<dim> dof_handler_p;

  AffineConstraints<double> hanging_node_constraints_u;
  AffineConstraints<double> hanging_node_constraints_p;
  AffineConstraints<double> constraints_u;
  AffineConstraints<double> constraints_p;

  NavierStokesMatrix<dim>                         navier_stokes_matrix;
  LinearAlgebra::distributed::BlockVector<double> system_rhs, const_rhs;

  std::shared_ptr<parallel::distributed::
                    SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>>
    sol_trans_u;
  std::shared_ptr<parallel::distributed::
                    SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>>
    sol_trans_p;

  NavierStokesPreconditioner<dim> preconditioner;

  GrowingVectorMemory<LinearAlgebra::distributed::BlockVector<double>> solver_memory;

  // here we store the MatrixFree that we
  // use for most of the vector assembly
  // functions and the matrix-free
  // implementation matrix-vector
  // products. There are two possible usages:
  // either we own the matrix_free by
  // ourselves (when calling the function
  // setup()) without argument, or we get it
  // from outside and share it.
public:
  std::shared_ptr<MatrixFree<dim>> matrix_free;

private:
  bool dofs_distributed;
  bool system_is_setup;

  unsigned int n_iterations_last_prec_update;
  unsigned int time_step_last_prec_update;
  bool         update_preconditioner;
  unsigned int update_preconditioner_frequency;

  std::shared_ptr<TimerOutput>    timer;
  std::pair<unsigned int, double> solver_timers[2];
};



namespace helpers
{
  // this struct is used to make std::shared_ptr not delete a structure when
  // we create it from a pointer to an external field
  template <typename CLASS>
  struct DummyDeleter
  {
    DummyDeleter(const bool do_delete = false)
      : do_delete(do_delete)
    {}

    void
    operator()(CLASS *pointer)
    {
      if (do_delete)
        delete pointer;
    }

    const bool do_delete;
  };
} // namespace helpers


/* ---------------------------- Inline functions ------------------------- */



template <int dim>
inline const FiniteElement<dim> &
NavierStokes<dim>::get_fe_u() const
{
  return fe_u;
}



template <int dim>
inline const FiniteElement<dim> &
NavierStokes<dim>::get_fe_p() const
{
  // We get simpler code by using FESystem, but we want to pretend we have a
  // usual element.
  return fe_p.base_element(0);
}



template <int dim>
inline const DoFHandler<dim> &
NavierStokes<dim>::get_dof_handler_u() const
{
  return dof_handler_u;
}



template <int dim>
inline const DoFHandler<dim> &
NavierStokes<dim>::get_dof_handler_p() const
{
  return dof_handler_p;
}



template <int dim>
inline const AffineConstraints<double> &
NavierStokes<dim>::get_constraints_u() const
{
  return constraints_u;
}



template <int dim>
inline AffineConstraints<double> &
NavierStokes<dim>::modify_constraints_u()
{
  return constraints_u;
}



template <int dim>
inline const AffineConstraints<double> &
NavierStokes<dim>::get_hanging_node_constraints_u() const
{
  return hanging_node_constraints_u;
}



template <int dim>
inline const AffineConstraints<double> &
NavierStokes<dim>::get_constraints_p() const
{
  return constraints_p;
}



template <int dim>
inline const AffineConstraints<double> &
NavierStokes<dim>::get_hanging_node_constraints_p() const
{
  return hanging_node_constraints_p;
}



template <int dim>
inline AffineConstraints<double> &
NavierStokes<dim>::modify_constraints_p()
{
  return constraints_p;
}



template <int dim>
inline const FlowParameters &
NavierStokes<dim>::get_parameters() const
{
  return parameters;
}



template <int dim>
inline void
NavierStokes<dim>::set_face_average_density(
  const typename Triangulation<dim>::cell_iterator &cell,
  const unsigned int                                face,
  const double                                      density)
{
  preconditioner.set_face_average_density(cell, face, density);
}

template <int dim>
inline void
NavierStokes<dim>::set_user_defined_material(
  std::function<Tensor<2, dim, VectorizedArray<double>>(
    const Tensor<2, dim, VectorizedArray<double>> &,
    const unsigned int,
    const unsigned int,
    const bool)> my_user_defined_material)
{
  navier_stokes_matrix.user_defined_material = my_user_defined_material;
}

#endif
