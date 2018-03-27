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

#include <adaflo/phase_field.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <fstream>
#include <iostream>

using namespace dealii;



template <int dim>
PhaseFieldSolver<dim>::PhaseFieldSolver (const FlowParameters &parameters_in,
                                         parallel::distributed::Triangulation<dim> &tria_in)
  :
  TwoPhaseBaseAlgorithm<dim> (parameters_in,
                              std_cxx11::shared_ptr<FiniteElement<dim> >(new FE_Q_iso_Q1<dim>(parameters_in.concentration_subdivisions)),
                              tria_in),
  parameters (this->TwoPhaseBaseAlgorithm<dim>::parameters)
{
  // computes the interpolation matrix from level set functions to pressure
  // which is needed for evaluating surface tension
  AssertThrow(dynamic_cast<const FE_Q<dim>*>(&this->navier_stokes.get_fe_p()) != 0,
              ExcNotImplemented());
  const FE_Q<dim> &fe_p = dynamic_cast<const FE_Q<dim>&>(this->navier_stokes.get_fe_p());
  interpolation_concentration_pressure.reinit (fe_p.dofs_per_cell,
                                               this->fe->dofs_per_cell);
  const std::vector<unsigned int> lexicographic_p = fe_p.get_poly_space_numbering_inverse();
  const FE_Q_iso_Q1<dim> &fe_mine = dynamic_cast<const FE_Q_iso_Q1<dim>&>(*this->fe);
  const std::vector<unsigned int> lexicographic_ls = fe_mine.get_poly_space_numbering_inverse();
  for (unsigned int j=0; j<fe_p.dofs_per_cell; ++j)
    {
      const Point<dim> p = fe_p.get_unit_support_points()[lexicographic_p[j]];
      for (unsigned int i=0; i<this->fe->dofs_per_cell; ++i)
        interpolation_concentration_pressure(j, i) =
          this->fe->shape_value(lexicographic_ls[i], p);
    }


  const QIterated<dim-1> face_quadrature(QGauss<1>(2), this->fe->degree);
  face_matrix.reinit(this->fe->dofs_per_face, face_quadrature.size());
  FE_Q<dim-1> fe_face(this->fe->degree);
  AssertDimension(fe_face.dofs_per_cell, this->fe->dofs_per_face);
  for (unsigned int i=0; i<this->fe->dofs_per_face; ++i)
    for (unsigned int q=0; q<face_quadrature.size(); ++q)
      face_matrix(i,q) = fe_face.shape_value(i,face_quadrature.point(q));

  this->curvature_name = "chemical_potential";
}



template <int dim>
void PhaseFieldSolver<dim>::distribute_dofs ()
{
  preconditioner_matrix.clear();
  this->TwoPhaseBaseAlgorithm<dim>::distribute_dofs();
}



template <int dim>
void PhaseFieldSolver<dim>::transform_distance_function (parallel::distributed::Vector<double> &vector) const
{
  for (unsigned int i=0; i<vector.local_size(); i++)
    vector.local_element(i) =
      -std::tanh(vector.local_element(i)/(this->epsilon_used));
}



template <int dim>
void PhaseFieldSolver<dim>::initialize_data_structures ()
{
  // now to the boundary conditions: the matrix system gets zero boundary
  // conditions on open boundaries
  ZeroFunction<dim> zero_func(1);
  typename FunctionMap<dim>::type homogeneous_dirichlet;
  for (typename std::set<types::boundary_id>::const_iterator
       it = this->boundary->fluid_type_plus.begin();
       it != this->boundary->fluid_type_plus.end(); ++it)
    homogeneous_dirichlet[*it] = &zero_func;
  for (typename std::set<types::boundary_id>::const_iterator
       it = this->boundary->fluid_type_minus.begin();
       it != this->boundary->fluid_type_minus.end(); ++it)
    homogeneous_dirichlet[*it] = &zero_func;
  VectorTools::interpolate_boundary_values(this->dof_handler, homogeneous_dirichlet,
                                           this->constraints);
  VectorTools::interpolate_boundary_values(this->dof_handler, homogeneous_dirichlet,
                                           this->constraints_curvature);

  this->TwoPhaseBaseAlgorithm<dim>::initialize_data_structures();

  evaluated_convection.resize(this->matrix_free.n_macro_cells()*
                              this->matrix_free.get_n_q_points(2));
  evaluated_phi.resize(this->matrix_free.n_macro_cells()*
                       this->matrix_free.get_n_q_points(2));

  velocity_vector = &this->navier_stokes.solution.block(0);

  // setup the data for matrix-free evaluation of face data
  face_indices.clear();
  face_JxW.clear();
  face_evaluated_c.clear();

  if (this->parameters.contact_angle != 0.)
    {
      const QIterated<dim-1> face_quadrature(QGauss<1>(2), this->fe->degree);
      FEFaceValues<dim> fe_face_values(this->mapping, *this->fe,
                                       face_quadrature, update_JxW_values);
      std::vector<types::global_dof_index> local_face_indices (this->fe->dofs_per_face);
      AssertDimension(face_matrix.n_cols(), face_quadrature.size());

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->dof_handler.begin_active(), endc = this->dof_handler.end();
      for ( ; cell != endc; ++cell)
        if (cell->is_locally_owned())
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->at_boundary(face) &&
                (this->boundary->dirichlet_conditions_u.find (cell->face(face)->boundary_id())
                 !=
                 this->boundary->dirichlet_conditions_u.end()
                 ||
                 this->boundary->no_slip.find (cell->face(face)->boundary_id())
                 !=
                 this->boundary->no_slip.end()))
              {
                fe_face_values.reinit(cell, face);
                for (unsigned int q=0; q<face_quadrature.size(); ++q)
                  face_JxW.push_back(fe_face_values.JxW(q));
                cell->face(face)->get_dof_indices(local_face_indices);
                for (unsigned int i=0; i<this->fe->dofs_per_face; ++i)
                  face_indices.push_back(local_face_indices[i]);
              }
      face_evaluated_c.resize(face_JxW.size());
    }
}



template <int dim>
void PhaseFieldSolver<dim>::print_n_dofs() const
{
  std::pair<unsigned int, unsigned int> ns_dofs = this->navier_stokes.n_dofs();
  this->pcout << std::endl
              << "Number of active cells: "
              << this->triangulation.n_global_active_cells () << "."
              << std::endl
              << "Number of Navier-Stokes degrees of freedom: "
              << ns_dofs.first + ns_dofs.second << " ("
              << ns_dofs.first << " + " << ns_dofs.second << ")."
              << std::endl
              << "Number of phase field degrees of freedom: "
              << this->dof_handler.n_dofs()*2 << " ("
              << this->dof_handler.n_dofs() << " + " << this->dof_handler.n_dofs() << ")."
              << std::endl;
}



template <int dim>
void
PhaseFieldSolver<dim>::compute_density_on_faces()
{
  if (this->parameters.augmented_taylor_hood == false ||
      this->parameters.density_diff == 0 ||
      this->parameters.linearization == FlowParameters::projection)
    return;

  FEValues<dim> fe_values(this->mapping, *this->fe,
                          this->face_center_quadrature,
                          update_values);
  std::vector<double> concentration_values(fe_values.n_quadrature_points);
  AssertDimension(concentration_values.size(), GeometryInfo<dim>::faces_per_cell);

  for (typename DoFHandler<dim>::active_cell_iterator cell =
         this->dof_handler.begin_active(); cell != this->dof_handler.end(); ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(this->solution.block(0), concentration_values);
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          {
            const double heaviside_val =
              std::min(1., std::max(0., 0.5*(concentration_values[f]+1.)));
            this->navier_stokes.set_face_average_density(cell, f,
                                                         this->parameters.density +
                                                         heaviside_val *
                                                         this->parameters.density_diff);
          }
      }
}




// @sect4{PhaseFieldSolver::advance_concentration}
template <int dim>
void
PhaseFieldSolver<dim>::create_cahn_hilliard_preconditioner ()
{
  this->timer->enter_subsection("Cahn-Hilliard preconditioner.");

  // Now, we turn to the generation of the
  // sparsity pattern for the level set
  // equation part.
  if (preconditioner_matrix.m() == 0)
    {
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(this->dof_handler, relevant_dofs);
      TrilinosWrappers::SparsityPattern csp (this->dof_handler.locally_owned_dofs(),
                                             this->dof_handler.locally_owned_dofs(),
                                             relevant_dofs,
                                             this->triangulation.get_communicator());
      DoFTools::make_sparsity_pattern (this->dof_handler, csp);
      csp.compress();
      preconditioner_matrix.reinit(csp);
    }

  const QIterated<dim> quadrature_formula(QGauss<1>(2), this->fe->degree);
  //QGauss<dim>   quadrature_formula(this->fe->degree+1);

  FEValues<dim> fe_values (this->mapping, *this->fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_JxW_values);

  const unsigned int   dofs_per_cell   = this->fe->dofs_per_cell;
  const unsigned int   n_q_points      = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  const double coefficient = std::sqrt(0.75 * this->parameters.surface_tension *
                                       this->epsilon_used *
                                       this->parameters.diffusion_length *
                                       this->parameters.diffusion_length /
                                       this->time_stepping.weight());

  preconditioner_matrix = 0;

  typename DoFHandler<dim>::active_cell_iterator
  cell    = this->dof_handler.begin_active(),
  endc    = this->dof_handler.end();
  for ( ; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit (cell);
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              double sum = 0;
              for (unsigned int q=0; q<n_q_points; ++q)
                sum += (fe_values.shape_value(i,q) * fe_values.shape_value(j,q)
                        +
                        coefficient *
                        (fe_values.shape_grad(i,q) * fe_values.shape_grad(j,q))
                       ) * fe_values.JxW(q);
              cell_matrix(i,j) = sum;
            }
        this->constraints.distribute_local_to_global
        (cell_matrix, local_dof_indices, preconditioner_matrix);
      }

  preconditioner_matrix.compress(VectorOperation::add);
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.aggregation_threshold = 0.02;
  amg_preconditioner.reset(new TrilinosWrappers::PreconditionAMG);
  amg_preconditioner->initialize (preconditioner_matrix,
                                  amg_data);
  this->timer->leave_subsection();
}



template <int dim>
struct MassMatrix
{
  MassMatrix (const PhaseFieldSolver<dim> &two_phase_in)
    :
    two_phase (two_phase_in)
  {}

  void vmult (parallel::distributed::Vector<double> &dst,
              const parallel::distributed::Vector<double> &src) const
  {
    two_phase.mass_vmult(dst,src);
  }

  const PhaseFieldSolver<dim> &two_phase;
};



// solve the linear system
template <int dim>
void
PhaseFieldSolver<dim>::advance_cahn_hilliard ()
{
  if (this->parameters.output_verbosity > 0)
    this->pcout << "  Advance Cahn-Hilliard: ";
  if (this->time_stepping.weight_has_changed() ||
      preconditioner_matrix.m() != this->dof_handler.n_dofs())
    create_cahn_hilliard_preconditioner();
  if (this->parameters.ch_do_newton == true)
    {
      for (unsigned int i=0; i<this->parameters.max_nl_iteration; ++i)
        {
          const double residual = compute_residual();
          if (this->parameters.output_verbosity > 0)
            this->pcout << "[" << residual << "/";
          if (i > 0 && residual < 0.001 * this->parameters.tol_nl_iteration)
            {
              if (this->parameters.output_verbosity > 0)
                this->pcout << "conv.]";
              break;
            }
          else
            solve_cahn_hilliard();
          if (this->parameters.output_verbosity > 0)
            this->pcout << "] " << std::flush;
        }
    }
  else
    {
      AssertThrow (false, ExcNotImplemented());
    }
  if (this->parameters.output_verbosity > 0)
    this->pcout << std::endl;
}




template <typename Preconditioner, int dim>
class BlockPreconditionerSimple
{
public:
  BlockPreconditionerSimple (const Preconditioner &preconditioner,
                             const MassMatrix<dim> &mass_matrix,
                             const double factor)
    :
    preconditioner (preconditioner),
    mass_matrix    (mass_matrix),
    factor         (factor)
  {}

  void vmult (parallel::distributed::BlockVector<double> &dst,
              const parallel::distributed::BlockVector<double> &src) const
  {
    AssertDimension (src.n_blocks(), 2);
    AssertDimension (dst.n_blocks(), 2);
    if (temp1.size() == 0)
      temp1.reinit(src.block(0), true);

    temp1 = src.block(0);
    temp1.add(factor, src.block(1));
    preconditioner.vmult (dst.block(1), temp1);
    mass_matrix.vmult (temp1, dst.block(1));
    temp1.add (-factor, src.block(1));
    preconditioner.vmult (dst.block(0), temp1);
    dst.block(1).sadd (1./factor, -1./factor, dst.block(0));
  }

private:
  const Preconditioner &preconditioner;
  const MassMatrix<dim> mass_matrix;
  const double factor;
  mutable parallel::distributed::Vector<double> temp1;
};





template <int dim>
void PhaseFieldSolver<dim>::solve_cahn_hilliard ()
{
  this->timer->enter_subsection("Cahn-Hilliard solve.");
  // similar factor as in the matrix assembly, but now sqrt(delta)/epsilon
  // instead of sqrt(delta * epsilon)
  const double factor_4 = (0.75 * this->parameters.surface_tension *
                           this->epsilon_used);
  const double factor_mobility = (this->parameters.diffusion_length*
                                  this->parameters.diffusion_length/
                                  this->time_stepping.weight());
  const double delta_eps = std::sqrt(factor_mobility / factor_4);

  PrimitiveVectorMemory< parallel::distributed::BlockVector<double> > mem;
  BlockPreconditionerSimple<TrilinosWrappers::PreconditionAMG, dim>
  preconditioner (*amg_preconditioner,
                  MassMatrix<dim>(*this),
                  delta_eps);

  const double tolerance = this->parameters.ch_do_newton == true ?
                           std::max (0.001 * this->parameters.tol_nl_iteration,
                                     0.1 * this->parameters.tol_lin_iteration * this->system_rhs.l2_norm())
                           : 0.01 * this->parameters.tol_nl_iteration;

  SolverControl solver_control (this->parameters.max_lin_iteration, tolerance);
  SolverGMRES<parallel::distributed::BlockVector<double> >::AdditionalData data(50,true);
  SolverGMRES<parallel::distributed::BlockVector<double> >
  solver (solver_control, mem, data);
  try
    {
      solver.solve (*this, this->solution_update, this->system_rhs, preconditioner);
    }
  catch (SolverControl::NoConvergence)
    {
    }

  if (this->parameters.output_verbosity > 0)
    this->pcout << solver_control.last_step();

  this->constraints.distribute (this->solution_update.block(0));
  this->constraints_curvature.distribute (this->solution_update.block(1));
  if (this->parameters.ch_do_newton == true)
    this->solution -= this->solution_update;
  else
    this->solution = this->solution_update;
  this->solution.update_ghost_values();
  this->timer->leave_subsection();
}



template <int dim>
unsigned int PhaseFieldSolver<dim>::advance_time_step()
{
  this->init_time_advance ();
  advance_cahn_hilliard();
  compute_force();
  return this->navier_stokes.evaluate_time_step();
}



template <int dim>
bool PhaseFieldSolver<dim>::mark_cells_for_refinement()
{
  if (this->parameters.adaptive_refinements == 0 ||
      this->time_stepping.step_no() % 5 != 0)
    return false;

  this->timer->enter_subsection("Probe grid refinement.");

  const int upper_level_limit = this->parameters.adaptive_refinements + this->refine_lower_level_limit;
  Vector<double> local_concentration(this->fe->dofs_per_cell);

  bool must_refine = false;
  typename DoFHandler<dim>::active_cell_iterator
  cell = this->dof_handler.begin_active(),
  endc = this->dof_handler.end();
  for ( ; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      {
        cell->clear_coarsen_flag();
        cell->clear_refine_flag();

        double max_distance = 0.;
        cell->get_dof_values(this->solution.block(0), local_concentration);
        for (unsigned int i=0; i<this->fe->dofs_per_cell; ++i)
          max_distance = std::max(max_distance, std::abs(1.-local_concentration(i)*
                                                         local_concentration(i)));

        bool refine_cell = ((cell->level() < upper_level_limit) &&
                            (max_distance > 0.01));

        if (refine_cell == true)
          {
            must_refine = true;
            cell->set_refine_flag();
          }
        else if ((cell->level()>this->refine_lower_level_limit) &&
                 (max_distance < 0.01))
          {
            must_refine = true;
            cell->set_coarsen_flag();
          }
      }
  const bool global_must_refine = Utilities::MPI::max(static_cast<unsigned int>(must_refine),
                                                      this->triangulation.get_communicator());
  this->timer->leave_subsection();
  return global_must_refine;
}




// explicit instantiations

template class PhaseFieldSolver<2>;
template class PhaseFieldSolver<3>;
