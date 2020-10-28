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

#ifndef __adaflo_phase_field_h_
#define __adaflo_phase_field_h_

#include <deal.II/base/timer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>

#include <adaflo/two_phase_base.h>

#include <fstream>
#include <iostream>

using namespace dealii;


namespace dealii
{
  namespace TrilinosWrappers
  {
    class PreconditionAMG;
  }
}


template <int dim>
class PhaseFieldSolver : public TwoPhaseBaseAlgorithm<dim>
{
public:
  PhaseFieldSolver (const FlowParameters &parameters,
                    parallel::distributed::Triangulation<dim> &triangulation);

  virtual ~PhaseFieldSolver() {}

  virtual void distribute_dofs ();
  virtual void initialize_data_structures ();
  virtual unsigned int advance_time_step ();

  void vmult (LinearAlgebra::distributed::BlockVector<double> &dst,
              const LinearAlgebra::distributed::BlockVector<double> &src) const;
  void mass_vmult (LinearAlgebra::distributed::Vector<double> &dst,
                   const LinearAlgebra::distributed::Vector<double> &src) const;

  const LinearAlgebra::distributed::Vector<double> *velocity_vector;

  virtual void transform_distance_function (LinearAlgebra::distributed::Vector<double> &vector) const;

protected:
  virtual void compute_force ();
  virtual void advance_cahn_hilliard ();

  double compute_residual();

  virtual void print_n_dofs() const;

  // compute the density on faces needed for the Navier-Stokes preconditioner
  // with FE_Q_DG0 elements
  void compute_density_on_faces ();

  void create_cahn_hilliard_preconditioner();
  void solve_cahn_hilliard ();

  virtual bool mark_cells_for_refinement ();

  // matrix-free worker operations for various operations
  template <int ls_degree, int velocity_degree>
  void local_compute_force (const MatrixFree<dim,double> &data,
                            LinearAlgebra::distributed::Vector<double> &dst,
                            const LinearAlgebra::distributed::Vector<double> &src,
                            const std::pair<unsigned int,unsigned int> &cell_range);

  template <int ls_degree, int velocity_degree>
  void local_residual (const MatrixFree<dim,double> &data,
                       LinearAlgebra::distributed::BlockVector<double> &dst,
                       const LinearAlgebra::distributed::BlockVector<double> &src,
                       const std::pair<unsigned int,unsigned int> &cell_range) const;

  template <int ls_degree>
  void local_vmult (const MatrixFree<dim,double> &data,
                    LinearAlgebra::distributed::BlockVector<double> &dst,
                    const LinearAlgebra::distributed::BlockVector<double> &src,
                    const std::pair<unsigned int,unsigned int> &cell_range) const;

  template <int ls_degree>
  void local_mass (const MatrixFree<dim,double> &data,
                   LinearAlgebra::distributed::Vector<double> &dst,
                   const LinearAlgebra::distributed::Vector<double> &src,
                   const std::pair<unsigned int,unsigned int> &cell_range) const;

  template <int operation>
  void apply_contact_bc (LinearAlgebra::distributed::BlockVector<double> &dst,
                         const LinearAlgebra::distributed::BlockVector<double> &src) const;

  const FlowParameters &parameters;

  TrilinosWrappers::SparseMatrix    preconditioner_matrix;
  std::shared_ptr<TrilinosWrappers::PreconditionAMG> amg_preconditioner;

  FullMatrix<double>   interpolation_concentration_pressure;

  mutable AlignedVector<Tensor<1,dim,VectorizedArray<double> > > evaluated_convection;
  mutable AlignedVector<VectorizedArray<double> > evaluated_phi;
  mutable std::vector<double> face_evaluated_c;
  std::vector<unsigned int> face_indices;
  std::vector<double> face_JxW;
  Table<2,double> face_matrix;
};


#endif
