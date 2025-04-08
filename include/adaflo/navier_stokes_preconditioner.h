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

#ifndef __adaflo_navier_stokes_preconditioner_h
#define __adaflo_navier_stokes_preconditioner_h

#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <adaflo/diagonal_preconditioner.h>
#include <adaflo/flow_base_algorithm.h>
#include <adaflo/navier_stokes_matrix.h>
#include <adaflo/time_stepping.h>


namespace dealii
{
  namespace TrilinosWrappers
  {
    class PreconditionBase;
    class PreconditionILU;
    class PreconditionAMG;
  } // namespace TrilinosWrappers
} // namespace dealii


namespace adaflo
{
  using namespace dealii;


  // forward declarations
  class MatrixFreeWrapper;
  class Precondition_LinML;
  template <int dim>
  class ComponentILUExtension;
  namespace AssemblyData
  {
    template <int dim>
    struct Preconditioner;
  }

  template <int dim>
  class NavierStokes;

  template <int dim>
  class NavierStokesPreconditioner
  {
  public:
    NavierStokesPreconditioner(const FlowParameters            &parameters,
                               const NavierStokes<dim>         &base_algorithm,
                               const Triangulation<dim>        &tria,
                               const AffineConstraints<double> &constraints_u);

    void
    clear();

    void
    compute();

    void
    vmult(LinearAlgebra::distributed::BlockVector<double>       &dst,
          const LinearAlgebra::distributed::BlockVector<double> &src) const;

    std::pair<unsigned int, double>
    solve_projection_system(
      const LinearAlgebra::distributed::BlockVector<double> &solution,
      LinearAlgebra::distributed::BlockVector<double>       &solution_update,
      LinearAlgebra::distributed::BlockVector<double>       &system_rhs,
      LinearAlgebra::distributed::Vector<double>            &projection_update,
      TimerOutput                                           &timer) const;

    void
    solve_pressure_mass(LinearAlgebra::distributed::Vector<double>       &dst,
                        const LinearAlgebra::distributed::Vector<double> &src) const;

    void
    initialize_matrices(const DoFHandler<dim>           &dof_handler_u,
                        const DoFHandler<dim>           &dof_handler_p,
                        const AffineConstraints<double> &constraints_p);
    void
    set_system_matrix(const NavierStokesMatrix<dim> &matrix);
    void
    assemble_matrices();

    void
    set_face_average_density(const typename Triangulation<dim>::cell_iterator &cell,
                             const unsigned int                                face,
                             const double                                      density);

    double
    get_face_average_density(const typename Triangulation<dim>::cell_iterator &cell,
                             const unsigned int                                face);

    bool
    is_variable() const;

    // Give an estimate of the memory consumption of this class. Note that not
    // all Trilinos-internal data structure can be quantitatively characterized,
    // so the actual memory usage is likely more.
    std::size_t
    memory_consumption() const;
    void
    print_memory_consumption(std::ostream &stream) const;

    // Returns statistics of the sum of all preconditioner applications since
    // the last call to this function in terms of minimum, average, and maximum
    // of times as seen over all MPI processes (first argument) as well as the
    // number of times the mat-vec was invoked. After returning the data, the
    // internal counters are re-set, so in case you wish global statistics, make
    // sure to accumulate the results of this call (note that adaflo has a
    // global timer object that accumulates such data in a neat way).
    //
    // The data returned is:
    // - the time spent in the velocity block
    // - the time spent in the divergence block
    // - the time spent in the pressure mass (and related) block
    // - the time spent in the pressure Poisson solver (if enabled)
    // - the total time in the preconditioner (sum of the above four, but with
    //   separate timer in order to eliminate possible waiting/synchronization)
    //
    // Note: This is a collective call and must be invoked on all processors.
    std::pair<Utilities::MPI::MinMaxAvg[5], unsigned int>
    get_timer_statistics() const;

    bool do_inner_solves;
    bool initialized;

  private:
    const AffineConstraints<double> &constraints_u;

    DoFHandler<dim>           dof_handler_u_scalar;
    AffineConstraints<double> constraints_u_scalar;
    AffineConstraints<double> constraints_schur_complement;
    std::vector<unsigned int> constraints_schur_complement_only;

    mutable LinearAlgebra::distributed::Vector<double> temp_vector, temp_vector2;

    TrilinosWrappers::SparseMatrix matrix_u;
    TrilinosWrappers::SparseMatrix matrix_p;
    TrilinosWrappers::SparseMatrix mass_matrix_p;

    DiagonalPreconditioner<double> pressure_diagonal_preconditioner;

    std::vector<unsigned int> scalar_dof_indices;

    std::shared_ptr<TrilinosWrappers::PreconditionILU> uu_ilu;
    std::shared_ptr<ComponentILUExtension<dim>>        uu_ilu_scalar;
    std::shared_ptr<MatrixFreeWrapper>                 uu_amg_mat;
    std::shared_ptr<Precondition_LinML>                uu_amg;

    std::shared_ptr<MatrixFreeWrapper>                  pp_mass_mat;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> pp_mass;
    std::shared_ptr<MatrixFreeWrapper>                  pp_poisson_mat;
    std::shared_ptr<Precondition_LinML>                 pp_poisson;

    std::vector<std::vector<bool>> constant_modes_u;
    std::vector<std::vector<bool>> constant_modes_p;

    const NavierStokesMatrix<dim> *matrix;

    struct IntegrationHelper
    {
      IntegrationHelper();

      void
      set_local_ordering_u(const FiniteElement<dim> &fe_u);

      void
      initialize_linear_elements(const FiniteElement<dim> &fe_u,
                                 const FiniteElement<dim> &fe_p);

      void
      get_indices_sub_elements(const FiniteElement<dim>               &fe,
                               std::vector<std::vector<unsigned int>> &dof_to_lin) const;

      void
      get_indices_sub_quad(const unsigned int                      degree,
                           std::vector<std::vector<unsigned int>> &quad_to_lin) const;

      std::vector<std::vector<unsigned int>> local_ordering_u;

      // data structures for linear elements
      unsigned int n_subelements_u;
      unsigned int n_subelements_p;

      static constexpr unsigned int n_dofs = GeometryInfo<dim>::vertices_per_cell;
      Tensor<1, dim>                grads_unit_cell[n_dofs][n_dofs];
      double                        values_unit_cell[n_dofs][n_dofs];
      std::vector<std::vector<unsigned int>> dof_to_lin_u;
      std::vector<std::vector<unsigned int>> dof_to_lin_p;

      std::unique_ptr<Quadrature<dim>>       quadrature_sub_u;
      std::unique_ptr<Quadrature<dim>>       quadrature_sub_p;
      std::vector<std::vector<unsigned int>> quad_to_lin_u;
      std::vector<std::vector<unsigned int>> quad_to_lin_p;
    };

    IntegrationHelper integration_helper;

    const FlowParameters    &parameters;
    const NavierStokes<dim> &flow_algorithm;

    // A table containing variable densities on faces for face integrals in
    // augmented Taylor--Hood
    std::vector<Table<2, double>> face_densities;

    mutable std::pair<unsigned int, double[5]> precond_timer;

    void
    local_assemble_preconditioner(
      const MatrixFree<dim, double> &matrix_free,
      std::shared_ptr<Threads::ThreadLocalStorage<AssemblyData::Preconditioner<dim>>>
        &in_data,
      const unsigned int &,
      const std::pair<unsigned int, unsigned int> &cell_range);

    friend struct AssemblyData::Preconditioner<dim>;
  };
} // namespace adaflo



template <int dim>
inline void
adaflo::NavierStokesPreconditioner<dim>::set_face_average_density(
  const typename Triangulation<dim>::cell_iterator &cell,
  const unsigned int                                face,
  const double                                      density)
{
  Assert(density > 0, ExcMessage("Density must be positive"));
  AssertDimension(face_densities.size(), cell->get_triangulation().n_levels());
  AssertIndexRange(static_cast<unsigned int>(cell->level()), face_densities.size());
  face_densities[cell->level()](cell->index(), face) = density;
}



template <int dim>
inline double
adaflo::NavierStokesPreconditioner<dim>::get_face_average_density(
  const typename Triangulation<dim>::cell_iterator &cell,
  const unsigned int                                face)
{
  if (this->parameters.density_diff == 0.)
    return this->parameters.density;
  if (this->parameters.linearization == FlowParameters::projection)
    return std::min(this->parameters.density,
                    this->parameters.density + this->parameters.density_diff);
  AssertIndexRange(static_cast<unsigned int>(cell->level()), face_densities.size());
  const double density = face_densities[cell->level()](cell->index(), face);
  Assert(density > 0, ExcMessage("Density on face has not been set"));
  return density;
}

#endif
