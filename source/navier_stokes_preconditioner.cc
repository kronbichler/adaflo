// --------------------------------------------------------------------------
//
// Copyright (C) 2011 - 2016 by the adaflo authors
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

#include <adaflo/navier_stokes_preconditioner.h>
#include <adaflo/block_matrix_extension.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/vector_tools.h>

#include <ml_epetra_utils.h>
#include <ml_struct.h>
#include <ml_include.h>
#include <ml_MultiLevelPreconditioner.h>


#include <fstream>
#include <iostream>
#include <iomanip>


// -------------------------------------------------------------------------
// -------------------------------------------------------------------------


// a wrapper around velocity_vmult in Navier-Stokes class to make work with
// iterative solvers

class MatrixFreeWrapper : public Epetra_RowMatrix
{
public:
  MatrixFreeWrapper (const TrilinosWrappers::SparseMatrix &tri_mat)
    :
    sparse_matrix (tri_mat)
  {}

  const TrilinosWrappers::SparseMatrix &get_matrix () const
  {
    return sparse_matrix;
  }

  // interface to Epetra_RowMatrix for use in ML_MultilevelPreconditioner:
  // Multiply is our vmult operation with Epetra vectors, whereas the rest is
  // given by the sparse matrix based on linear elements
  virtual int Multiply (bool TransA,
                        const Epetra_MultiVector &X,
                        Epetra_MultiVector &Y) const = 0;
  virtual void vmult (parallel::distributed::Vector<double> &,
                      const parallel::distributed::Vector<double> &) const = 0;

  // implement all sorts of functions in Epetra_RowMatrix

  int NumMyRowEntries (int MyRow, int &NumEntries) const
  {
    return sparse_matrix.trilinos_matrix().NumMyRowEntries(MyRow, NumEntries);
  }
  int MaxNumEntries () const
  {
    return sparse_matrix.trilinos_matrix().MaxNumEntries();
  }
  int ExtractMyRowCopy (int MyRow, int Length, int &NumEntries, double *Values,
                        int *Indices) const
  {
    return sparse_matrix.trilinos_matrix().
           ExtractMyRowCopy(MyRow, Length, NumEntries, Values, Indices);
  }
  int ExtractDiagonalCopy (Epetra_Vector &Diagonal) const
  {
    return sparse_matrix.trilinos_matrix().ExtractDiagonalCopy (Diagonal);
  }
  const Epetra_Map &RowMatrixRowMap () const
  {
    return sparse_matrix.trilinos_matrix().RowMatrixRowMap();
  }
  const Epetra_Map &RowMatrixColMap () const
  {
    return sparse_matrix.trilinos_matrix().RowMatrixColMap();
  }
  const Epetra_Import *RowMatrixImporter () const
  {
    return sparse_matrix.trilinos_matrix().RowMatrixImporter();
  }
  const Epetra_BlockMap &Map() const
  {
    return sparse_matrix.trilinos_matrix().Map();
  }
  int SetUseTranspose (bool)
  {
    Assert (false, ExcNotImplemented());
    return 0;
  }
  int Apply (const Epetra_MultiVector &X, Epetra_MultiVector &Y) const
  {
    return Multiply (false, X, Y);
  }
  int ApplyInverse (const Epetra_MultiVector &, Epetra_MultiVector &) const
  {
    Assert (false, ExcNotImplemented());
    return 0;
  }
  double NormInf() const
  {
    return sparse_matrix.trilinos_matrix().NormInf();
  }
  const char *Label () const
  {
    return "MatrixFreeWrapper";
  }
  bool UseTranspose () const
  {
    return false;
  }
  bool HasNormInf () const
  {
    return false;
  }
  const Epetra_Comm &Comm () const
  {
    return sparse_matrix.trilinos_matrix().Comm();
  }
  const Epetra_Map &OperatorDomainMap () const
  {
    return sparse_matrix.trilinos_matrix().OperatorDomainMap();
  }
  const Epetra_Map &OperatorRangeMap () const
  {
    return sparse_matrix.trilinos_matrix().OperatorRangeMap();
  }
  int Solve (bool , bool , bool, const Epetra_MultiVector &,
             Epetra_MultiVector &) const
  {
    Assert (false, ExcNotImplemented());
    return 0;
  }
  int LeftScale (const Epetra_Vector &)
  {
    Assert (false, ExcNotImplemented());
    return 0;
  }
  int RightScale (const Epetra_Vector &)
  {
    Assert (false, ExcNotImplemented());
    return 0;
  }
  int InvRowSums (Epetra_Vector &X) const
  {
    return sparse_matrix.trilinos_matrix().InvRowSums (X);
  }
  int InvColSums (Epetra_Vector &X) const
  {
    return sparse_matrix.trilinos_matrix().InvColSums (X);
  }
  bool Filled () const
  {
    return sparse_matrix.trilinos_matrix().Filled();
  }
  double NormOne () const
  {
    return sparse_matrix.trilinos_matrix().NormOne();
  }
  int NumGlobalNonzeros () const
  {
    return sparse_matrix.trilinos_matrix().NumGlobalNonzeros();
  }
  long long NumGlobalNonzeros64() const
  {
    return NumGlobalNonzeros();
  }
  int NumGlobalRows () const
  {
    return sparse_matrix.trilinos_matrix().NumGlobalRows();
  }
  long long NumGlobalRows64() const
  {
    return sparse_matrix.trilinos_matrix().NumGlobalRows64();
  }
  int NumGlobalCols () const
  {
    return sparse_matrix.trilinos_matrix().NumGlobalCols();
  }
  long long NumGlobalCols64() const
  {
    return sparse_matrix.trilinos_matrix().NumGlobalCols64();
  }
  int NumGlobalDiagonals () const
  {
    return sparse_matrix.trilinos_matrix().NumGlobalDiagonals();
  }
  long long NumGlobalDiagonals64() const
  {
    return sparse_matrix.trilinos_matrix().NumGlobalDiagonals64();
  }
  int NumMyNonzeros () const
  {
    return sparse_matrix.trilinos_matrix().NumMyNonzeros();
  }
  int NumMyRows () const
  {
    return sparse_matrix.trilinos_matrix().NumMyRows();
  }
  int NumMyCols () const
  {
    return sparse_matrix.trilinos_matrix().NumMyCols();
  }
  int NumMyDiagonals () const
  {
    return sparse_matrix.trilinos_matrix().NumMyDiagonals();
  }
  bool LowerTriangular () const
  {
    return sparse_matrix.trilinos_matrix().LowerTriangular();
  }
  bool UpperTriangular () const
  {
    return sparse_matrix.trilinos_matrix().UpperTriangular();
  }

  std::size_t memory_consumption_cell () const
  {
    return sizeof(this);
  }

  std::size_t memory_consumption_mat () const
  {
    return static_cast<std::size_t>(12) *
           (static_cast<std::size_t>(sparse_matrix.trilinos_matrix().NumMyNonzeros())
            + sparse_matrix.m());
  }

protected:
  const TrilinosWrappers::SparseMatrix &sparse_matrix;
};



template <int dim>
class VelocityMatrix : public MatrixFreeWrapper
{
public:
  VelocityMatrix (const NavierStokesMatrix<dim> &ns_matrix,
                  const TrilinosWrappers::SparseMatrix &tri_mat)
    :
    MatrixFreeWrapper (tri_mat),
    ns_matrix (ns_matrix)
  {
    ns_matrix.initialize_u_vector(src);
    ns_matrix.initialize_u_vector(dst);
    AssertDimension (tri_mat.m(), ns_matrix.n_dofs_u());
  }

  void vmult (parallel::distributed::Vector<double> &dst,
              const parallel::distributed::Vector<double> &src) const
  {
    ns_matrix.velocity_vmult (dst, src);
  }

  int Multiply (bool ,
                const Epetra_MultiVector &X,
                Epetra_MultiVector &Y) const
  {
    Assert (X.NumVectors() == 1, ExcNotImplemented());
    AssertDimension(src.local_size(), static_cast<unsigned int>(X.MyLength()));
    VectorView<double> my_src(src.local_size(), src.begin());
    VectorView<double> my_x(src.local_size(), X[0]);
    VectorView<double> my_dst(src.local_size(), dst.begin());
    VectorView<double> my_y(src.local_size(), Y[0]);

    my_src = my_x;
    vmult (dst, src);
    my_y = my_dst;

    return 0;
  }

private:
  const NavierStokesMatrix<dim> &ns_matrix;
  mutable parallel::distributed::Vector<double> dst;
  parallel::distributed::Vector<double> src;
};



template <int dim>
class PressurePoissonMatrix : public MatrixFreeWrapper
{
public:
  PressurePoissonMatrix (const NavierStokesMatrix<dim> &ns_matrix,
                         const TrilinosWrappers::SparseMatrix &tri_mat,
                         const bool use_trilinos_matrix,
                         const std::vector<unsigned int> &constraints_schur_complement_only)
    :
    MatrixFreeWrapper (tri_mat),
    ns_matrix (ns_matrix),
    use_trilinos_matrix (use_trilinos_matrix),
    constraints_schur_complement_only (constraints_schur_complement_only)
  {
    ns_matrix.initialize_p_vector(src);
    ns_matrix.initialize_p_vector(dst);
    AssertDimension (tri_mat.m(), ns_matrix.n_dofs_p());
    schur_complement_tmp_values.resize(constraints_schur_complement_only.size());

    // extract the diagonal values on entries constrained Schur complement
    // entries to get the correct magnitude
    constrained_diagonal_values.resize(constraints_schur_complement_only.size());
    for (unsigned int i=0; i<constraints_schur_complement_only.size(); ++i)
      {
        const types::global_dof_index glob_i = ns_matrix.get_matrix_free().get_dof_handler(1).
          locally_owned_dofs().nth_index_in_set(constraints_schur_complement_only[i]);
        constrained_diagonal_values[i] = tri_mat.el(glob_i,glob_i);
      }
  }

  void vmult (parallel::distributed::Vector<double> &dst,
              const parallel::distributed::Vector<double> &src) const
  {
    if (use_trilinos_matrix)
      sparse_matrix.vmult(dst, src);
    else
      {
        // temporarily change the input vector for imposing zero conditions on
        // the pressure
        const unsigned int n_constraints = constraints_schur_complement_only.size();
        for (unsigned int i=0; i<n_constraints; ++i)
          {
            schur_complement_tmp_values[i] = src.local_element(constraints_schur_complement_only[i]);
            const_cast<parallel::distributed::Vector<double> &>(src).
              local_element(constraints_schur_complement_only[i]) = 0;
          }

        ns_matrix.pressure_poisson_vmult (dst, src);

        for (unsigned int i=0; i<n_constraints; ++i)
          {
            const_cast<parallel::distributed::Vector<double> &>(src).
              local_element(constraints_schur_complement_only[i]) = schur_complement_tmp_values[i];
            dst.local_element(constraints_schur_complement_only[i]) =
              constrained_diagonal_values[i] * schur_complement_tmp_values[i];
          }
      }
  }

  int Multiply (bool ,
                const Epetra_MultiVector &X,
                Epetra_MultiVector &Y) const
  {
    Assert (X.NumVectors() == 1, ExcNotImplemented());
    if (use_trilinos_matrix)
      sparse_matrix.trilinos_matrix().Multiply(false, X, Y);
    else
      {
        VectorView<double> my_src(src.local_size(), src.begin());
        VectorView<double> my_x(src.local_size(), X[0]);
        VectorView<double> my_dst(src.local_size(), dst.begin());
        VectorView<double> my_y(src.local_size(), Y[0]);

        my_src = my_x;
        vmult (dst, src);
        my_y = my_dst;
      }
    return 0;
  }

private:
  const NavierStokesMatrix<dim> &ns_matrix;
  const bool use_trilinos_matrix;
  mutable parallel::distributed::Vector<double> dst;
  parallel::distributed::Vector<double> src;
  const std::vector<unsigned int> &constraints_schur_complement_only;
  std::vector<double> constrained_diagonal_values;
  mutable std::vector<double> schur_complement_tmp_values;
};



template <int dim>
class PressureMassMatrix
{
public:
  PressureMassMatrix (const NavierStokesMatrix<dim> &ns_matrix)
    :
    ns_matrix (ns_matrix)
  {
  }

  void vmult (parallel::distributed::Vector<double> &dst,
              const parallel::distributed::Vector<double> &src) const
  {
    ns_matrix.pressure_mass_vmult (dst, src);
  }

private:
  const NavierStokesMatrix<dim> &ns_matrix;
};



class Precondition_LinML
{
public:
  Precondition_LinML () {};

  void initialize (const MatrixFreeWrapper &matrix,
                   const std::vector<std::vector<bool> > &constant_modes,
                   const bool is_velocity)
  {
    preconditioner.reset();
    Teuchos::ParameterList parameter_list;
    ML_Epetra::SetDefaults("SA",parameter_list);
    parameter_list.set("smoother: type", "Chebyshev");
    parameter_list.set("smoother: Chebyshev alpha", 10.);
    parameter_list.set("aggregation: threshold", 0.02);
    parameter_list.set("aggregation: type", "Uncoupled");
    parameter_list.set("smoother: sweeps", 2);
    if (!is_velocity)
      parameter_list.set("cycle applications", 2);
    parameter_list.set("coarse: max size", 2000);

    const unsigned int dim = constant_modes.size();
    /*
    parameter_list.set("repartition: enable",1);
    parameter_list.set("repartition: max min ratio",1.3);
    parameter_list.set("repartition: min per proc",500);
    parameter_list.set("repartition: partitioner","Zoltan");
    parameter_list.set("repartition: Zoltan dimensions",dim>0?(int)dim:3);
    */
    parameter_list.set("ML output", 0);
    const Epetra_Map &domain_map = matrix.OperatorDomainMap();
    Epetra_MultiVector distributed_constant_modes (domain_map, dim>0?dim:1);
    std::vector<double> dummy (dim);

    if (dim > 1)
      {
        const unsigned int my_size = domain_map.NumMyElements();
        AssertDimension(my_size, constant_modes[0].size());

        // Reshape null space as a contiguous vector of doubles so that
        // Trilinos can read from it.

        for (unsigned int d=0; d<dim; ++d)
          for (unsigned int row=0; row<my_size; ++row)
            {
              distributed_constant_modes[d][row] = constant_modes[d][row];
            }

        parameter_list.set("null space: type", "pre-computed");
        parameter_list.set("null space: dimension",
                           distributed_constant_modes.NumVectors());
        if (my_size > 0)
          parameter_list.set("null space: vectors",
                             distributed_constant_modes.Values());
        // We need to set a valid pointer to data even if there is no data on
        // the current processor. Therefore, pass a dummy in that case
        else
          parameter_list.set("null space: vectors",
                             &dummy[0]);
      }
    preconditioner.reset (new ML_Epetra::MultiLevelPreconditioner
                          (matrix, parameter_list));
  }

  void vmult (parallel::distributed::Vector<double> &dst,
              const parallel::distributed::Vector<double> &src) const
  {
    AssertDimension (static_cast<int>(dst.local_size()),
                     preconditioner->OperatorDomainMap().NumMyElements());
    AssertDimension (static_cast<int>(src.local_size()),
                     preconditioner->OperatorRangeMap().NumMyElements());
    Epetra_Vector tril_dst (View, preconditioner->OperatorDomainMap(),
                            dst.begin());
    Epetra_Vector tril_src (View, preconditioner->OperatorRangeMap(),
                            const_cast<double *>(src.begin()));

    const int ierr = preconditioner->ApplyInverse (tril_src, tril_dst);
    AssertThrow (ierr == 0, ExcTrilinosError(ierr));
  }

  std::size_t memory_consumption () const
  {
    return sizeof (ML_Epetra::MultiLevelPreconditioner);
  }

private:
  std_cxx11::shared_ptr<ML_Epetra::MultiLevelPreconditioner> preconditioner;
};



// -------------------------------------------------------------------------
// ------------------ vmult operation of preconditioner class --------------
// -------------------------------------------------------------------------

// a wrapper around velocity_vmult in Navier-Stokes class to make work with
// iterative solvers
namespace helper
{
  template <int dim>
  struct NavierStokesVelocityMatrix
  {
    NavierStokesVelocityMatrix (const NavierStokesMatrix<dim> &ns_matrix)
      :
      ns_matrix (ns_matrix)
    {};

    void vmult (parallel::distributed::Vector<double> &dst,
                const parallel::distributed::Vector<double> &src) const
    {
      ns_matrix.velocity_vmult (dst, src);
    }

    const NavierStokesMatrix<dim> &ns_matrix;
  };
}



template <int dim>
void
NavierStokesPreconditioner<dim>::vmult (parallel::distributed::BlockVector<double> &dst,
                                        const parallel::distributed::BlockVector<double> &src) const
{
  Assert (initialized == true, ExcNotInitialized());
  precond_timer.first++;
  Timer total, time;

  // if we do not make inner solves, just need to multiply with the
  // preconditioner once. Otherwise use a linear solver
  if (do_inner_solves == false)
    {
      if (parameters.precondition_velocity == FlowParameters::u_ilu)
        {
          Assert (uu_ilu.get() != 0, ExcNotInitialized());
          uu_ilu->vmult (dst.block(0), src.block(0));
        }
      else if (parameters.precondition_velocity == FlowParameters::u_ilu_scalar)
        {
          Assert (uu_ilu_scalar.get() != 0, ExcNotInitialized());
          uu_ilu_scalar->multiply (dst.block(0), src.block(0));
        }
      else
        {
          Assert (uu_amg.get() != 0, ExcNotInitialized());
          uu_amg->vmult (dst.block(0), src.block(0));
        }
    }
  else
    {
      SolverControl solver_control (100, 3e-2*src.block(0).l2_norm());
      SolverBicgstab<parallel::distributed::Vector<double> >::AdditionalData bicg_data;
      bicg_data.exact_residual = false;
      SolverBicgstab<parallel::distributed::Vector<double> > solver (solver_control,
          bicg_data);
      const helper::NavierStokesVelocityMatrix<dim> uu_mat (*matrix);
      try
        {
          if (parameters.precondition_velocity == FlowParameters::u_ilu)
            {
              Assert (uu_ilu.get() != 0, ExcNotInitialized());
              solver.solve (uu_mat, dst.block(0), src.block(0), *uu_ilu);
            }
          else if (parameters.precondition_velocity == FlowParameters::u_ilu_scalar)
            {
              Assert (uu_ilu_scalar.get() != 0, ExcNotInitialized());
              solver.solve (uu_mat, dst.block(0), src.block(0), *uu_ilu_scalar);
            }
          else
            {
              Assert (uu_amg.get() != 0, ExcNotInitialized());
              solver.solve (uu_mat, dst.block(0), src.block(0), *uu_amg);
            }
        }
      catch (...)
        Assert (false,
                ExcMessage("Solver for velocity-velocity matrix did not converge."));
    }
  precond_timer.second[0] += time.wall_time();

  // multiply by divergence matrix in block-triangular part of preconditioner
  time.restart();
  temp_vector.equ(-1.0, src.block(1));
  matrix->divergence_vmult_add (temp_vector, dst.block(0));
  precond_timer.second[1] += time.wall_time();

  // finally, apply the pressure preconditioner
  time.restart();

  // For the stationary case, we apply the pressure convection-diffusion
  // operator proposed by Kay, Loghin and Wathen (SIAM J Sci Comput 24, 2002).
  if (parameters.physical_type == FlowParameters::incompressible_stationary)
    {
      try
        {
          SolverControl solver_control (30, 1e-2*temp_vector.l2_norm());
          SolverCG<parallel::distributed::Vector<double> > solver (solver_control);
          dst.block(1) = 0;
          solver.solve (matrix_p, dst.block(1), temp_vector, *pp_poisson);
        }
      catch (...)
        {
        }
      precond_timer.second[3] += time.wall_time();
      time.restart();

      // Need to apply the constraints in the pressure Poisson matrix also
      // on the convection-diffusion matrix
      for (unsigned int i=0; i<constraints_schur_complement_only.size(); ++i)
        dst.block(1).local_element(constraints_schur_complement_only[i]) = 0.;
      matrix->pressure_convdiff_vmult(temp_vector2, dst.block(1));
      for (unsigned int i=0; i<constraints_schur_complement_only.size(); ++i)
        temp_vector2.local_element(constraints_schur_complement_only[i]) =
          temp_vector.local_element(constraints_schur_complement_only[i]);

      solve_pressure_mass(dst.block(1), temp_vector2);
      precond_timer.second[2] += time.wall_time();
      precond_timer.second[4] += total.wall_time();

      return;
    }

  // For the time-dependent case, use a combination of pressure Poisson and
  // pressure mass matrix (Cahouet-Chabbard)
  solve_pressure_mass(dst.block(1), temp_vector);
  precond_timer.second[2] += time.wall_time();

  if (parameters.density > 0)
    {
      time.restart();

      if (do_inner_solves == false)
        pp_poisson->vmult (temp_vector2, temp_vector);
      else
        {
          try
            {
              SolverControl solver_control (30, 3e-2*temp_vector.l2_norm());
              SolverCG<parallel::distributed::Vector<double> > solver (solver_control);
              temp_vector2 = 0;
              solver.solve (matrix_p, temp_vector2, temp_vector, *pp_poisson);
            }
          catch (...)
            {
            }
        }
      dst.block(1) += temp_vector2;
      precond_timer.second[3] += time.wall_time();
    }
  precond_timer.second[4] += total.wall_time();
}



template <int dim>
void
NavierStokesPreconditioner<dim>
::solve_pressure_mass(parallel::distributed::Vector<double> &dst,
                      const parallel::distributed::Vector<double> &src) const
{
  if (pp_mass.get() == 0)
    {
      if (0)
        {
          dst = src;
          pressure_diagonal_preconditioner.vmult(dst, dst);
        }
      else
        {
          try
            {
              PressureMassMatrix<dim> p_mat(*matrix);
              dst = 0;
              ReductionControl solver_control (100, 1e-50, 1e-2);
              SolverCG<parallel::distributed::Vector<double> > solver (solver_control);
              solver.solve (p_mat, dst, src, pressure_diagonal_preconditioner);
            }
          catch (...)
            {
              Assert (false,
                      ExcMessage ("Solver for pressure mass matrix did not converge."));
            }
        }
    }
  else
    pp_mass->vmult(dst, src);
}



template <int dim>
std::pair<unsigned int,double> NavierStokesPreconditioner<dim>
::solve_projection_system(const parallel::distributed::BlockVector<double> &solution,
                          parallel::distributed::BlockVector<double> &solution_update,
                          parallel::distributed::BlockVector<double> &system_rhs,
                          parallel::distributed::Vector<double> &projection_update,
                          TimerOutput &timer) const
{
  SolverControl solver_control (parameters.max_lin_iteration,
                                0.5 * parameters.tol_nl_iteration);
  parallel::distributed::Vector<double> solution_u_copy;
  {
    TimerOutput::Scope scope(timer, "NS solve velocity");
    solution_u_copy = solution.block(0);
    solution_update = 0;
    SolverGMRES<parallel::distributed::Vector<double> >
    solver (solver_control,
            SolverGMRES<parallel::distributed::Vector<double> >::AdditionalData(50, true));

    if (parameters.precondition_velocity == FlowParameters::u_ilu)
      solver.solve (*uu_amg_mat, solution_update.block(0), system_rhs.block(0), *uu_ilu);
    else if (parameters.precondition_velocity == FlowParameters::u_ilu_scalar)
      solver.solve (*uu_amg_mat, solution_update.block(0), system_rhs.block(0), *uu_ilu_scalar);
    else
      solver.solve (*uu_amg_mat, solution_update.block(0), system_rhs.block(0), *uu_amg);

    constraints_u.distribute (solution_update.block(0));
    solution_u_copy += solution_update.block(0);
  }

  {
    TimerOutput::Scope scope(timer, "NS solve pressure");
    system_rhs.block(1) = 0;
    matrix->divergence_vmult_add(system_rhs.block(1), solution_u_copy);

    SolverControl solver_control (1000, 0.1 * parameters.time_step_size_start /
                                  std::min(parameters.density, parameters.density +
                                           parameters.density_diff) *
                                  parameters.tol_nl_iteration);
    SolverCG<parallel::distributed::Vector<double> >  solver (solver_control);

    projection_update = 0;
    solver.solve (*pp_poisson_mat, projection_update, system_rhs.block(1), *pp_poisson);
    constraints_schur_complement.distribute(projection_update);

    // This is the update for the rotational part
    system_rhs.block(1) = 0;
    matrix->divergence_vmult_add (system_rhs.block(1), solution_u_copy, true);
    ReductionControl solver_control_mass (1000, 1e-50, 0.1 * parameters.tol_lin_iteration);
    SolverCG<parallel::distributed::Vector<double> > solver2 (solver_control_mass);
    PressureMassMatrix<dim> pressure_mass (*matrix);
    solution_update.block(1) = 0;

    if (pp_mass == 0)
      solver2.solve (pressure_mass, solution_update.block(1), system_rhs.block(1),
                     pressure_diagonal_preconditioner);
    else
      solver2.solve (pressure_mass, solution_update.block(1), system_rhs.block(1),
                     *pp_mass);

    constraints_schur_complement.distribute(solution_update.block(1));
    solution_update.block(1) += projection_update;
  }
  return std::make_pair(solver_control.last_step(), solver_control.last_value());
}



// -------------------------------------------------------------------------
// ----------- initialization and computation of preconditioner class ------
// -------------------------------------------------------------------------



template <int dim>
void
NavierStokesPreconditioner<dim>::compute ()
{
  if (parameters.precondition_velocity != FlowParameters::u_ilu_scalar)
    uu_amg_mat.reset (new VelocityMatrix<dim>(*matrix, matrix_u));
  TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
  ilu_data.overlap = 0;
  if (parameters.precondition_velocity == FlowParameters::u_ilu)
    {
      if (uu_ilu.get() == 0)
        uu_ilu.reset (new TrilinosWrappers::PreconditionILU);
      uu_ilu->initialize (matrix_u, ilu_data);
    }
  else if (parameters.precondition_velocity == FlowParameters::u_ilu_scalar)
    {
      if (uu_ilu_scalar.get() == 0)
        uu_ilu_scalar.reset (new ComponentILUExtension<dim>());
      uu_ilu_scalar->initialize (matrix_u, ilu_data, scalar_dof_indices);
    }
  else
    {
      uu_amg.reset (new Precondition_LinML);
      uu_amg->initialize(*uu_amg_mat, constant_modes_u, true);
    }

  if (parameters.density > 0)
    {
      pp_poisson_mat.reset (new PressurePoissonMatrix<dim>(*matrix, matrix_p,
                                                           parameters.augmented_taylor_hood ||
                                                           matrix->pressure_degree()==1,
                                                           constraints_schur_complement_only));
      pp_poisson.reset (new Precondition_LinML);
      pp_poisson->initialize(*pp_poisson_mat, constant_modes_p, false);
    }

  // Augmented Taylor-Hood always needs AMG on the pressure mass matrix
  if (parameters.augmented_taylor_hood == true)
    {
      TrilinosWrappers::PreconditionAMG::AdditionalData data;
      data.constant_modes = constant_modes_p;
      pp_mass.reset (new TrilinosWrappers::PreconditionAMG);
      dynamic_cast<TrilinosWrappers::PreconditionAMG *>(pp_mass.get())
      ->initialize (mass_matrix_p, data);
    }
  else if (parameters.precondition_pressure == FlowParameters::p_mass_ilu)
    {
      pp_mass.reset (new TrilinosWrappers::PreconditionILU);
      dynamic_cast<TrilinosWrappers::PreconditionILU *>(pp_mass.get())
      ->initialize (mass_matrix_p, ilu_data);
    }

  // compute and invert diagonal mass matrix
  else
    {
      temp_vector = 1.;
      matrix->pressure_mass_vmult(temp_vector2, temp_vector);
      pressure_diagonal_preconditioner.reinit(temp_vector2);
    }

  initialized = true;
}



template <int dim>
NavierStokesPreconditioner<dim>
::NavierStokesPreconditioner (const FlowParameters         &parameters,
                              const FlowBaseAlgorithm<dim> &flow_algorithm,
                              const parallel::distributed::Triangulation<dim> &tria,
                              const ConstraintMatrix       &constraints_u)
  :
  do_inner_solves  (false),
  initialized (false),
  constraints_u (constraints_u),
  dof_handler_u_scalar (tria),
  parameters(parameters),
  flow_algorithm(flow_algorithm)
{}



template <int dim>
void
NavierStokesPreconditioner<dim>::clear ()
{
  matrix_u.clear();
  matrix_p.clear();
  mass_matrix_p.clear();
  uu_ilu.reset();
  uu_ilu_scalar.reset();
  uu_amg_mat.reset();
  uu_amg.reset();
  pp_poisson.reset();
  pp_mass.reset();
  constraints_schur_complement.clear();
  constraints_schur_complement_only.clear();
  std::vector<unsigned int> dummy;
  scalar_dof_indices.swap(dummy);
  initialized = false;
}



template <int dim>
void
NavierStokesPreconditioner<dim>
::initialize_matrices (const DoFHandler<dim>  &dof_handler_u,
                       const DoFHandler<dim>  &dof_handler_p,
                       const ConstraintMatrix &constraints_p)
{
  const FiniteElement<dim> &fe_u = dof_handler_u.get_fe();
  const FiniteElement<dim> &fe_p = dof_handler_p.get_fe();
  const parallel::distributed::Triangulation<dim> &triangulation =
    dynamic_cast<const parallel::distributed::Triangulation<dim> &>(dof_handler_u.get_triangulation());
  const unsigned int this_mpi_process =
    Utilities::MPI::this_mpi_process(triangulation.get_communicator());

  integration_helper.set_local_ordering_u (fe_u);
  integration_helper.initialize_linear_elements(fe_u, fe_p);

  // For scalar ILU, need to distributed DoFs and fill in the various constraints
  if (parameters.precondition_velocity == FlowParameters::u_ilu_scalar)
    {
      dof_handler_u_scalar.distribute_dofs(fe_u.base_element(0));
      DoFRenumbering::Cuthill_McKee(dof_handler_u_scalar, false, true);
      IndexSet relevant_dofs_u;
      DoFTools::extract_locally_relevant_dofs (dof_handler_u_scalar, relevant_dofs_u);
      constraints_u_scalar.reinit(relevant_dofs_u);
      DoFTools::make_hanging_node_constraints (dof_handler_u_scalar,
                                               constraints_u_scalar);

      for (unsigned int d=0; d<dim; ++d)
        if (flow_algorithm.boundary->periodic_boundaries[d] !=
            std::pair<types::boundary_id,types::boundary_id>(-1, -1))
          {
            const types::boundary_id in = flow_algorithm.boundary->periodic_boundaries[d].first;
            const types::boundary_id out = flow_algorithm.boundary->periodic_boundaries[d].second;
            DoFTools::make_periodicity_constraints (dof_handler_u_scalar, in, out, d,
                                                    constraints_u_scalar);
          }

      ZeroFunction<dim> zero_func(1);
      typename FunctionMap<dim>::type homogeneous_dirichlet;
      for (typename std::map<types::boundary_id,
           std_cxx11::shared_ptr<Function<dim> > >::
           const_iterator it = flow_algorithm.boundary->dirichlet_conditions_u.begin();
           it != flow_algorithm.boundary->dirichlet_conditions_u.end(); ++it)
        homogeneous_dirichlet[it->first] = &zero_func;
      for (typename std::set<types::boundary_id>::const_iterator it =
             flow_algorithm.boundary->no_slip.begin();
           it != flow_algorithm.boundary->no_slip.end(); ++it)
        homogeneous_dirichlet[*it] = &zero_func;
      VectorTools::interpolate_boundary_values(flow_algorithm.mapping,
                                               dof_handler_u_scalar,
                                               homogeneous_dirichlet,
                                               constraints_u_scalar);
      constraints_u_scalar.close();
    }

  // Build constraints for the Schur complement
  constraints_schur_complement.clear();
  constraints_schur_complement.reinit(constraints_p.get_local_lines());
  constraints_schur_complement.merge(constraints_p);

  // On open boundaries, the pressure Laplace matrix for the preconditioner
  // will get Dirichlet conditions. This is stored in the
  // constraints_schur_complement matrix for standard elements. For augmented
  // Taylor--Hood elements, we instead apply the Dirichlet constraints weakly
  if (!flow_algorithm.boundary->open_conditions_p.empty() &&
      !parameters.augmented_taylor_hood)
    {
      ZeroFunction<dim> zero_func(1);
      typename FunctionMap<dim>::type homogeneous_dirichlet;
      for (typename std::map<types::boundary_id,
           std_cxx11::shared_ptr<Function<dim> > >::
           const_iterator it = flow_algorithm.boundary->open_conditions_p.begin();
           it != flow_algorithm.boundary->open_conditions_p.end(); ++it)
        {
          homogeneous_dirichlet[it->first] = &zero_func;
        }
      VectorTools::interpolate_boundary_values(flow_algorithm.mapping, dof_handler_p,
                                               homogeneous_dirichlet,
                                               constraints_schur_complement);
    }

  // Fix the pressure constant if requested
  if (!flow_algorithm.boundary->pressure_fix.empty())
    {
      typename FunctionMap<dim>::type dirichlet_p;
      ZeroFunction<dim> zero_function;
      for (typename std::map<types::boundary_id,
           std_cxx11::shared_ptr<Function<dim> > >::
           iterator it = flow_algorithm.boundary->pressure_fix.begin();
           it != flow_algorithm.boundary->pressure_fix.end(); ++it)
        {
          AssertThrow(flow_algorithm.boundary->open_conditions_p.empty(),
                      ExcMessage("Cannot fix pressure when it is prescribed "
                                 "on some other boundaries"));
          dirichlet_p[it->first] = &zero_function;
        }
      std::map<types::global_dof_index,double> boundary_values;
      VectorTools::interpolate_boundary_values (flow_algorithm.mapping, dof_handler_p,
                                                dirichlet_p, boundary_values);

      // To fix the index, take the one with the lowest index on all MPI
      // processes
      int min_local_index = std::numeric_limits<int>::min();
      for (std::map<types::global_dof_index,double>::iterator it=boundary_values.begin();
           it!=boundary_values.end(); ++it)
        if (!constraints_schur_complement.is_constrained(it->first))
          {
            min_local_index = -boundary_values.begin()->first;
            break;
          }
      const types::global_dof_index min_index =
        -Utilities::MPI::max(min_local_index, triangulation.get_communicator());
      AssertThrow (min_index < dof_handler_p.n_dofs(),
                   ExcMessage("Could not locate pressure boundary dof."));
      if (constraints_schur_complement.can_store_line(min_index))
        constraints_schur_complement.add_line(min_index);
    }

  if (parameters.augmented_taylor_hood)
    {
      std::vector<std::vector<bool> > constant_modes;
      DoFTools::extract_constant_modes(dof_handler_p, std::vector<bool>(1,true),
                                       constant_modes);
      AssertDimension(constant_modes.size(), 2);
      int min_local_index = std::numeric_limits<int>::min();
      for (unsigned int i=0; i<constant_modes[1].size(); ++i)
        if (constant_modes[1][i] == true)
          {
            min_local_index = -dof_handler_p.locally_owned_dofs().nth_index_in_set(i);
            break;
          }
      const types::global_dof_index min_index =
        -Utilities::MPI::max(min_local_index, triangulation.get_communicator());
      AssertThrow (min_index < dof_handler_p.n_dofs(),
                   ExcMessage("Could not locate pressure average dof."));
      if (constraints_schur_complement.can_store_line(min_index))
        constraints_schur_complement.add_line(min_index);
    }
  constraints_schur_complement.close();

  constraints_schur_complement_only.clear();
  for (unsigned int i=0; i<dof_handler_p.locally_owned_dofs().n_elements(); ++i)
    if (constraints_schur_complement.is_constrained(dof_handler_p.locally_owned_dofs().nth_index_in_set(i)) &&
        !constraints_p.is_constrained(dof_handler_p.locally_owned_dofs().nth_index_in_set(i)))
      constraints_schur_complement_only.push_back(i);


  // Build sparsity pattern for the velocity
  {
    TrilinosWrappers::SparsityPattern csp;
    csp.reinit (dof_handler_u.locally_owned_dofs(), dof_handler_u.locally_owned_dofs(),
                constraints_u.get_local_lines(), triangulation.get_communicator());

    if (parameters.precondition_velocity == FlowParameters::u_amg ||
        parameters.precondition_velocity == FlowParameters::u_ilu)
      {
        Table<2,DoFTools::Coupling> coupling (dim, dim);
        for (unsigned int c=0; c<dim; ++c)
          for (unsigned int d=0; d<dim; ++d)
            if ( c == d )
              coupling[c][d] = DoFTools::always;
            else
              coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern (dof_handler_u, coupling, csp,
                                         constraints_u, false,
                                         this_mpi_process);
      }
    else if (parameters.precondition_velocity == FlowParameters::u_ilu_scalar)
      {
        // Build sparsity pattern for scalar variables of Navier-Stokes
        const IndexSet &reduced_owned = dof_handler_u_scalar.locally_owned_dofs();
        csp.reinit (reduced_owned, reduced_owned, constraints_u_scalar.get_local_lines(),
                    triangulation.get_communicator());
        DoFTools::make_sparsity_pattern (dof_handler_u_scalar, csp,
                                         constraints_u_scalar, false, this_mpi_process);

        // Next, find the right numbering between the components in the scalar
        // field and the original vector components
        scalar_dof_indices.clear();
        scalar_dof_indices.resize(dim*reduced_owned.n_elements());
        std::vector<types::global_dof_index> dof_indices(fe_u.base_element(0).dofs_per_cell), vdof_indices(fe_u.dofs_per_cell);
        for (typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_u_scalar.begin_active(), vcell = dof_handler_u.begin_active(); cell != dof_handler_u_scalar.end(); ++cell, ++vcell)
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(dof_indices);
              vcell->get_dof_indices(vdof_indices);
              for (unsigned int i=0; i<dof_indices.size(); ++i)
                if (reduced_owned.is_element(dof_indices[i]))
                  {
                    const unsigned int local_index = reduced_owned.index_within_set(dof_indices[i]);
                    for (unsigned int d=0; d<dim; ++d)
                      {
                        const unsigned int global_index =
                          dof_handler_u.locally_owned_dofs().index_within_set
                          (vdof_indices[fe_u.component_to_system_index(d, i)]);
                        scalar_dof_indices[local_index*dim+d] = global_index;
                      }
                  }
            }
      }
    else
      {
        // if we subdivide the elements into linear subelements with the same
        // number of degrees of freedom, the system matrix is sparser and we
        // need to manually fill in the entries by a loop over
        // cells. Fortunately, it isn't too complicated to replicate
        // make_sparsity_pattern for that purpose

        std::vector<types::global_dof_index>
        linear_dof_indices (GeometryInfo<dim>::vertices_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (fe_u.dofs_per_cell);

        const std::vector<std::vector<unsigned int> > &dof_to_lin_u
          = integration_helper.dof_to_lin_u;
        const std::vector<std::vector<unsigned int> > &local_ordering_u
          = integration_helper.local_ordering_u;

        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_u.begin_active(),
        endc = dof_handler_u.end();
        for ( ; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices (local_dof_indices);
              for (unsigned int i=0; i<dof_to_lin_u.size(); ++i)
                for (unsigned int d=0; d<dim; ++d)
                  {
                    for (unsigned int k=0;
                         k<GeometryInfo<dim>::vertices_per_cell; ++k)
                      linear_dof_indices[k] =
                        local_dof_indices[local_ordering_u[d][dof_to_lin_u[i][k]]];
                    constraints_u.add_entries_local_to_global (linear_dof_indices,
                                                               csp, false);
                  }
            }
      }
    csp.compress();
    matrix_u.reinit(csp);
  }


  // Build sparsity pattern for the pressure
  {
    TrilinosWrappers::SparsityPattern csp (dof_handler_p.locally_owned_dofs(),
                                           dof_handler_p.locally_owned_dofs(),
                                           constraints_p.get_local_lines(),
                                           triangulation.get_communicator());

    if (parameters.precondition_velocity != FlowParameters::u_amg_linear ||
        parameters.augmented_taylor_hood)
      DoFTools::make_sparsity_pattern (dof_handler_p, csp,
                                       constraints_schur_complement, false,
                                       this_mpi_process);
    else
      {
        // manual construction of sparsity pattern for linear elements
        std::vector<types::global_dof_index>
        linear_dof_indices (GeometryInfo<dim>::vertices_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (fe_p.dofs_per_cell);
        const std::vector<std::vector<unsigned int> > &dof_to_lin_p
          = integration_helper.dof_to_lin_p;

        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_p.begin_active(),
        endc = dof_handler_p.end();
        for ( ; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices (local_dof_indices);
              for (unsigned int i=0; i<dof_to_lin_p.size(); ++i)
                {
                  for (unsigned int k=0; k<GeometryInfo<dim>::vertices_per_cell; ++k)
                    linear_dof_indices[k] = local_dof_indices[dof_to_lin_p[i][k]];
                  constraints_schur_complement.
                  add_entries_local_to_global(linear_dof_indices, csp, false);
                }
            }
      }
    csp.compress();
    if (parameters.density > 0 && parameters.augmented_taylor_hood)
      {
        TrilinosWrappers::SparsityPattern sparsity (dof_handler_p.locally_owned_dofs(),
                                                    dof_handler_p.locally_owned_dofs(),
                                                    constraints_p.get_local_lines(),
                                                    triangulation.get_communicator());
        std::vector<types::global_dof_index> local_dof_indices (fe_p.dofs_per_cell);
        std::vector<types::global_dof_index> neigh_dof_indices (fe_p.dofs_per_cell);
        std::vector<types::global_dof_index> const_dof_index(1);
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_p.begin_active(),
        endc = dof_handler_p.end();
        for ( ; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices (local_dof_indices);
              constraints_schur_complement.add_entries_local_to_global(local_dof_indices,
                                                                       sparsity, false);
              for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                if (!cell->at_boundary(face))
                  {
                    if (cell->face(face)->has_children())
                      for (unsigned int i=0; i<cell->face(face)->number_of_children();
                           ++i)
                        {
                          cell->neighbor_child_on_subface(face, i)->get_dof_indices
                          (neigh_dof_indices);
                          const_dof_index[0] = neigh_dof_indices.back();
                          constraints_schur_complement.
                          add_entries_local_to_global(const_dof_index,
                                                      local_dof_indices,
                                                      sparsity, false);
                          constraints_schur_complement.
                          add_entries_local_to_global(local_dof_indices,
                                                      const_dof_index,
                                                      sparsity, false);
                        }
                    else
                      {
                        cell->neighbor(face)->get_dof_indices (neigh_dof_indices);
                        const_dof_index[0] = neigh_dof_indices.back();
                        constraints_schur_complement.
                        add_entries_local_to_global(const_dof_index,
                                                    local_dof_indices,
                                                    sparsity, false);
                        constraints_schur_complement.
                        add_entries_local_to_global(local_dof_indices,
                                                    const_dof_index,
                                                    sparsity, false);
                      }
                  }
            }

        sparsity.compress();
        matrix_p.reinit (sparsity);
      }
    else if (parameters.density > 0)
      matrix_p.reinit(csp);
    if (parameters.precondition_pressure == FlowParameters::p_mass_ilu ||
        parameters.augmented_taylor_hood)
      mass_matrix_p.reinit (csp);
  }

  if (parameters.precondition_velocity == FlowParameters::u_amg ||
      parameters.precondition_velocity == FlowParameters::u_amg_linear)
    DoFTools::extract_constant_modes(dof_handler_u, std::vector<bool>(dim,true),
                                     constant_modes_u);

  if (parameters.density > 0 || parameters.augmented_taylor_hood)
    DoFTools::extract_constant_modes(dof_handler_p, std::vector<bool>(1,true),
                                     constant_modes_p);


  if (parameters.augmented_taylor_hood && parameters.density_diff != 0.)
    {
      // Need to find maximum index a query to cell->level() can have. There
      // seems to be no direct function in deal.II, so need to manually find
      // it...
      std::vector<unsigned int> max_index_per_cell(triangulation.n_levels());
      for (typename Triangulation<dim>::active_cell_iterator cell=
             triangulation.begin_active(); cell!=triangulation.end(); ++cell)
        if (!cell->is_artificial())
          max_index_per_cell[cell->level()] = std::max(max_index_per_cell[cell->level()],
                                                       (unsigned int)cell->index());
      face_densities.resize(triangulation.n_levels());
      for (unsigned int l=0; l<triangulation.n_levels(); ++l)
        face_densities[l].reinit(max_index_per_cell[l]+1,
                                 GeometryInfo<dim>::faces_per_cell);
    }
}



template <int dim>
void
NavierStokesPreconditioner<dim>::set_system_matrix(const NavierStokesMatrix<dim> &matrix)
{
  this->matrix = &matrix;
}


namespace AssemblyData
{
  // This collects all the data we need for assembly of the preconditioner
  // matrices
  template <int dim>
  struct Preconditioner
  {
    Preconditioner (const NavierStokesPreconditioner<dim> &preconditioner,
                    const Mapping<dim>       &mapping,
                    const FiniteElement<dim> &fe_u,
                    const FiniteElement<dim> &fe_p);

    Preconditioner (const Preconditioner &data);

    const FiniteElement<dim> &fe_u;
    FEValues<dim> fe_values_u;
    FEValues<dim> fe_values_p;
    FEFaceValues<dim> fe_face_values_p;
    FESubfaceValues<dim> fe_subface_values_p;

    // the indices from local to global
    std::vector<types::global_dof_index> local_dof_indices_u;
    std::vector<types::global_dof_index> local_dof_indices_u_scalar;
    std::vector<types::global_dof_index> local_dof_indices_p;
    std::vector<types::global_dof_index> local_dof_indices_p_neigh;
    std::vector<types::global_dof_index> local_dof_indices_p_dg0;

    // the matrices that hold the local results
    FullMatrix<double> velocity_matrix;
    FullMatrix<double> pressure_matrix;
    FullMatrix<double> pressure_mass_matrix;
    FullMatrix<double> pressure_ip_matrix_1;
    FullMatrix<double> pressure_ip_matrix_2;

    // temporary matrices that we need to compute the local matrices
    FullMatrix<double> pressure_laplace_part_matrix;
    FullMatrix<double> pressure_mass_part_matrix;
    FullMatrix<double> velocity_matrix_permuted;
    FullMatrix<double> velocity_phi_matrix;
    FullMatrix<double> velocity_phi_and_grad_matrix;

    FEValues<dim> fe_values_lin_u;
    FEValues<dim> fe_values_lin_p;

    FullMatrix<double> linear_matrix;

    std::vector<types::global_dof_index> linear_dof_indices;
  };

  template <int dim>
  Preconditioner<dim>::Preconditioner (const NavierStokesPreconditioner<dim> &preconditioner,
                                       const Mapping<dim>       &mapping,
                                       const FiniteElement<dim> &fe_u,
                                       const FiniteElement<dim> &fe_p)
    :
    fe_u        (fe_u),
    fe_values_u (mapping, fe_u.base_element(0), QGauss<dim>(fe_u.degree+1),
                 update_values | update_gradients | update_JxW_values),
    fe_values_p (mapping, fe_p, fe_values_u.get_quadrature(),
                 update_values | update_gradients | update_JxW_values),
    fe_face_values_p (mapping, fe_p, QGauss<dim-1>(fe_p.degree+1),
                      update_values | update_gradients | update_JxW_values |
                      update_normal_vectors),
    fe_subface_values_p (mapping, fe_p, fe_face_values_p.get_quadrature(),
                         update_values | update_gradients | update_JxW_values |
                         update_normal_vectors),

    local_dof_indices_u          (fe_u.dofs_per_cell),
    local_dof_indices_u_scalar   (fe_u.base_element(0).dofs_per_cell),
    local_dof_indices_p          (fe_p.dofs_per_cell),
    local_dof_indices_p_neigh    (fe_p.dofs_per_cell),
    local_dof_indices_p_dg0      (1),
    velocity_matrix              (fe_u.dofs_per_cell,
                                  fe_u.dofs_per_cell),
    pressure_matrix              (fe_p.dofs_per_cell,
                                  fe_p.dofs_per_cell),
    pressure_mass_matrix         (fe_p.dofs_per_cell,
                                  fe_p.dofs_per_cell),
    pressure_ip_matrix_1         (1, fe_p.dofs_per_cell),
    pressure_ip_matrix_2         (fe_p.dofs_per_cell, 1),
    pressure_laplace_part_matrix (fe_values_p.n_quadrature_points *dim,
                                  fe_p.dofs_per_cell),
    pressure_mass_part_matrix    (fe_values_p.n_quadrature_points,
                                  fe_p.dofs_per_cell),
    velocity_matrix_permuted     (fe_u.base_element(0).dofs_per_cell,
                                  fe_u.dofs_per_cell),
    velocity_phi_matrix          (fe_values_u.n_quadrature_points*(dim+1),
                                  fe_u.base_element(0).dofs_per_cell),
    velocity_phi_and_grad_matrix (fe_values_u.n_quadrature_points*(dim+1),
                                  fe_u.dofs_per_cell),
    fe_values_lin_u   (mapping, fe_p,
                       *preconditioner.integration_helper.quadrature_sub_u,
                       update_JxW_values | update_inverse_jacobians),
    fe_values_lin_p   (mapping, fe_p,
                       *preconditioner.integration_helper.quadrature_sub_p,
                       update_JxW_values | update_inverse_jacobians),
    linear_matrix      (GeometryInfo<dim>::vertices_per_cell,
                        GeometryInfo<dim>::vertices_per_cell),
    linear_dof_indices (GeometryInfo<dim>::vertices_per_cell)
  {}

  template <int dim>
  Preconditioner<dim>::Preconditioner (const Preconditioner &data)
    :
    fe_u        (data.fe_u),
    fe_values_u (data.fe_values_u.get_mapping(),
                 data.fe_values_u.get_fe(),
                 data.fe_values_u.get_quadrature(),
                 data.fe_values_u.get_update_flags()),
    fe_values_p (data.fe_values_p.get_mapping(),
                 data.fe_values_p.get_fe(),
                 data.fe_values_p.get_quadrature(),
                 data.fe_values_p.get_update_flags()),
    fe_face_values_p (data.fe_values_p.get_mapping(),
                      data.fe_values_p.get_fe(),
                      data.fe_face_values_p.get_quadrature(),
                      data.fe_face_values_p.get_update_flags()),
    fe_subface_values_p (data.fe_values_p.get_mapping(),
                         data.fe_values_p.get_fe(),
                         data.fe_face_values_p.get_quadrature(),
                         data.fe_face_values_p.get_update_flags()),
    local_dof_indices_u          (data.local_dof_indices_u),
    local_dof_indices_u_scalar   (data.local_dof_indices_u_scalar),
    local_dof_indices_p          (data.local_dof_indices_p),
    local_dof_indices_p_neigh    (data.local_dof_indices_p_neigh),
    local_dof_indices_p_dg0      (1),
    velocity_matrix              (data.velocity_matrix),
    pressure_matrix              (data.pressure_matrix),
    pressure_mass_matrix         (data.pressure_mass_matrix),
    pressure_ip_matrix_1         (data.pressure_ip_matrix_1),
    pressure_ip_matrix_2         (data.pressure_ip_matrix_2),
    pressure_laplace_part_matrix (data.pressure_laplace_part_matrix),
    pressure_mass_part_matrix    (data.pressure_mass_part_matrix),
    velocity_matrix_permuted     (data.velocity_matrix_permuted),
    velocity_phi_matrix          (data.velocity_phi_matrix),
    velocity_phi_and_grad_matrix (data.velocity_phi_and_grad_matrix),
    fe_values_lin_u (data.fe_values_lin_u.get_mapping(),
                     data.fe_values_lin_u.get_fe(),
                     data.fe_values_lin_u.get_quadrature(),
                     data.fe_values_lin_u.get_update_flags()),
    fe_values_lin_p (data.fe_values_lin_p.get_mapping(),
                     data.fe_values_lin_p.get_fe(),
                     data.fe_values_lin_p.get_quadrature(),
                     data.fe_values_lin_p.get_update_flags()),
    linear_matrix     (GeometryInfo<dim>::vertices_per_cell,
                       GeometryInfo<dim>::vertices_per_cell),
    linear_dof_indices (GeometryInfo<dim>::vertices_per_cell)
  {}
}



namespace
{
  // A function that computes the interior penalty matrices for FE_Q_DG0
  // elements: As compared to standard face integrals with fully discontinuous
  // elements where we compute the full interior penalty matrices (see deal.II
  // step-39 tutorial program), here we only need to compute jump terms of the
  // gradients of the continuous part and discontinuous values. The
  // discontinuous functions simply evaluate to one, so we do not even need to
  // create an FEValues object for them.
  template <int dim>
  void compute_ip_matrix_q_dg0(const FEFaceValuesBase<dim> &fe_face_values,
                               const unsigned int           face1,
                               const typename Triangulation<dim>::active_cell_iterator &neighbor,
                               const double                 scale,
                               FullMatrix<double>          &matrix_1,
                               FullMatrix<double>          &matrix_2)
  {
    // Compute penalty parameter
    const unsigned int face2 = fe_face_values.get_cell()->neighbor_face_no(face1);
    const unsigned int normal1 = GeometryInfo<dim>::unit_normal_direction[face1];
    const unsigned int normal2 = GeometryInfo<dim>::unit_normal_direction[face2];
    const unsigned int degsq = fe_face_values.get_fe().degree *
                               (fe_face_values.get_fe().degree+1);

    double penalty1 = degsq / fe_face_values.get_cell()->extent_in_direction(normal1);
    double penalty2 = degsq / neighbor->extent_in_direction(normal2);
    if (fe_face_values.get_cell()->neighbor(face1)->has_children())
      penalty1 *= 2.;
    else if (neighbor->neighbor(face2)->has_children())
      penalty2 *= 2.;
    const double penalty = 0.5*(penalty1 + penalty2);

    AssertDimension(matrix_1.n_rows(), 1);
    AssertDimension(matrix_1.n_cols(), fe_face_values.dofs_per_cell);
    AssertDimension(matrix_2.n_cols(), 1);
    AssertDimension(matrix_2.n_rows(), fe_face_values.dofs_per_cell);

    for (unsigned int i=0; i<fe_face_values.dofs_per_cell-1; ++i)
      {
        // Compute term: 2 ({{nabla u}},{{v n}})_F + 2 ({{nabla v}},{{un}})_F
        double sum = 0;
        for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
          sum += fe_face_values.shape_grad(i,q) * fe_face_values.normal_vector(q)
                 * fe_face_values.JxW(q);
        matrix_1(0, i) = sum * scale * 0.5;
        matrix_2(i, 0) = sum * scale * 0.5;
      }
    // Compute term: -4 (({{un}},{{vn}})_F  (only discontinuous shape function)
    double sum = 0;
    for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
      sum -= fe_face_values.JxW(q);
    matrix_1(0, fe_face_values.dofs_per_cell-1) = scale * penalty * sum;
  }


  // Computes the standard Nitsche matrix for weak imposition of Dirichlet
  // boundary conditions
  template <int dim>
  void compute_nitsche_matrix(const FEFaceValuesBase<dim> &fe_face_values,
                              const unsigned int           face1,
                              const double                 scale,
                              FullMatrix<double>          &matrix)
  {
    const unsigned int normal1 = GeometryInfo<dim>::unit_normal_direction[face1];
    const unsigned int degsq = fe_face_values.get_fe().degree *
                               (fe_face_values.get_fe().degree+1);
    const double penalty = degsq / fe_face_values.get_cell()->extent_in_direction(normal1);
    for (unsigned int i=0; i<fe_face_values.dofs_per_cell; ++i)
      for (unsigned int j=0; j<fe_face_values.dofs_per_cell; ++j)
        {
          double sum = 0;
          for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
            sum += ((2. * penalty * fe_face_values.shape_value(i,q) -
                     fe_face_values.shape_grad(i,q) * fe_face_values.normal_vector(q)) *
                    fe_face_values.shape_value(j,q) -
                    fe_face_values.shape_grad(j,q) * fe_face_values.normal_vector(q) *
                    fe_face_values.shape_value(i,q)
                   ) * fe_face_values.JxW(q);
          matrix(i,j) += sum * scale;
        }

  }

}



template <int dim>
void NavierStokesPreconditioner<dim>::local_assemble_preconditioner
(const MatrixFree<dim> &matrix_free,
 Threads::ThreadLocalStorage<AssemblyData::Preconditioner<dim> > &in_data,
 const unsigned int &,
 const std::pair<unsigned int,unsigned int> &cell_range)
{
  // Assume that in_data is
  // ThreadLocalStorage<Assembly::Preconditioner<dim>>

  AssemblyData::Preconditioner<dim> &data = in_data.get();
  const unsigned int dofs_per_u_component =
    integration_helper.local_ordering_u[0].size();
  const unsigned int dofs_per_cell_p = data.local_dof_indices_p.size();
  const unsigned int n_q_points      = data.fe_values_u.get_quadrature().size();
  AssertDimension (n_q_points, matrix_free.get_n_q_points(0));

  typedef VectorizedArray<double> vector_t;
  const std::vector<std::vector<unsigned int> > &local_ordering_u =
    integration_helper.local_ordering_u;
  const bool use_variable_parameters = matrix->use_variable_coefficients();

  const vector_t *densities = use_variable_parameters ?
                              matrix->begin_densities(cell_range.first) : 0;
  const vector_t *viscosities = use_variable_parameters ?
                                matrix->begin_viscosities(cell_range.first) : 0;

  const typename NavierStokesMatrix<dim>::velocity_stored *linearized =
    matrix->begin_linearized_velocities(cell_range.first);
  const TimeStepping &time_stepping = matrix->get_time_stepping();
  const double tau = time_stepping.tau1();

  const unsigned int n_dofs = GeometryInfo<dim>::vertices_per_cell;
  double data_for_val[n_dofs][n_dofs];
  Tensor<1,dim> data_for_grad[n_dofs][n_dofs];

  // since we compute Jacobians on macro elements, need to adjust the inverse
  // Jacobian by a factor of fe.degree (that says how many subdivisions we
  // make per direction and scales the Jacobian). for JxW, the quadrature
  // formula will take care of the factor

  const double factor_jacobian_u = data.fe_u.degree;
  const double factor_jacobian_p = data.fe_values_p.get_fe().degree;
  const Triangulation<dim> &triangulation = matrix_free.get_dof_handler(0).get_triangulation();

  for (unsigned int mcell=cell_range.first; mcell<cell_range.second; ++mcell)
    {
      for (unsigned int vec=0; vec<matrix_free.n_components_filled(mcell); ++vec)
        {
          typename DoFHandler<dim>::active_cell_iterator
          cell_u = matrix_free.get_cell_iterator (mcell, vec, 0),
          cell_p = matrix_free.get_cell_iterator (mcell, vec, 1);
          typename parallel::distributed::Triangulation<dim>::active_cell_iterator
          cell (&triangulation, cell_u->level(), cell_u->index());
          Assert(cell->is_locally_owned(), ExcInternalError());

          // if we should not use linear elements for the preconditioner, use
          // an optimized version of classical assembly: Instead of manually
          // writing three nested loops for assembly for each matrix row and
          // column, we write the data into two full matrices and hand it off
          // to matrix multiplication, which is much more optimized than three
          // nested loops with access to FEValues::shape_value and friends.

          if (parameters.precondition_velocity != FlowParameters::u_amg_linear)
            {
              data.fe_values_u.reinit (cell);
              data.fe_values_p.reinit (cell);

              const std::vector<double> &JxW = data.fe_values_u.get_JxW_values();

              const bool scalar_ilu =
                parameters.precondition_velocity == FlowParameters::u_ilu_scalar;

              if (scalar_ilu)
                {
                  data.velocity_phi_and_grad_matrix.reinit((dim+1)*n_q_points,
                                                           dofs_per_u_component, true);
                  data.velocity_matrix_permuted.reinit(dofs_per_u_component,
                                                       dofs_per_u_component, true);
                }

              // fill matrices with basis function data
              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  if (use_variable_parameters)
                    {
                      Assert (densities[q][vec]>=0,
                              ExcMessage("Density must be non-negative."));
                      Assert (viscosities[q][vec]>0,
                              ExcMessage("Viscosity must be positive."));
                    }
                  const double actual_nu = use_variable_parameters ?
                                           viscosities[q][vec] : parameters.viscosity;
                  const double actual_rho = use_variable_parameters ?
                                            densities[q][vec] : parameters.density;
                  const double weight_nu  = JxW[q] * (parameters.tau_grad_div+
                                                      actual_nu*tau);
                  const double weight_rho = JxW[q] * actual_rho;
                  for (unsigned int i=0; i<dofs_per_u_component; ++i)
                    {
                      const Tensor<1,dim> &phi_grad_iq = data.fe_values_u.shape_grad(i,q);
                      const double phi_iq = data.fe_values_u.shape_value(i,q);
                      for (unsigned int d=0; d<dim; ++d)
                        {
                          const double val = phi_grad_iq[d] * weight_nu;
                          data.velocity_phi_matrix((dim+1)*q+d,i) = val;
                          if (scalar_ilu)
                            data.velocity_phi_and_grad_matrix((dim+1)*q+d,i) = phi_grad_iq[d];
                          else
                            {
                              for (unsigned int e=0; e<dim; ++e)
                                data.velocity_phi_and_grad_matrix((dim+1)*q+d,i*dim+e)
                                  = phi_grad_iq[d];
                              data.velocity_phi_and_grad_matrix((dim+1)*q+d,i*dim+d)
                              += phi_grad_iq[d];
                            }
                        }
                      data.velocity_phi_matrix((dim+1)*q+dim,i) =
                        phi_iq * weight_rho;
                      double phi_and_grad = parameters.physical_type ==
                                            FlowParameters::incompressible_stationary ? 0. :
                                            phi_iq * time_stepping.weight();
                      if (linearized != 0)
                        {
                          for (unsigned int d=0; d<dim; ++d)
                            phi_and_grad += tau * phi_grad_iq[d]*linearized[q].first[d][vec];
                          if (!scalar_ilu)
                            for (unsigned int d=0; d<dim; ++d)
                              data.velocity_phi_and_grad_matrix((dim+1)*q+dim,i*dim+d) =
                                phi_and_grad + linearized[q].second[d][d][vec] * phi_iq * tau;
                        }
                      else if (!scalar_ilu)
                        for (unsigned int d=0; d<dim; ++d)
                          data.velocity_phi_and_grad_matrix((dim+1)*q+dim,i*dim+d) = phi_and_grad;
                      if (scalar_ilu)
                        data.velocity_phi_and_grad_matrix((dim+1)*q+dim,i) = phi_and_grad;
                    }
                }

              // velocity matrix: rho * val(phi_i) * ( val(phi_j)/dt + val(u)
              // grad (phi_j) + val (phi_j) * grad u ) + nu * grad(phi_i) *
              // grads(phi_j).

              // We only assemble the components within one vector component,
              // so to fill the FullMatrix, we have to copy the data to the
              // diagonal part of the velocity matrix now. Then, write the
              // local contributions into the global matrix.

              data.velocity_phi_matrix.Tmmult (data.velocity_matrix_permuted,
                                               data.velocity_phi_and_grad_matrix);
              if (!scalar_ilu)
                {
                  for (unsigned int i=0; i<dofs_per_u_component; ++i)
                    for (unsigned int j=0; j<dofs_per_u_component; ++j)
                      {
                        for (unsigned int d=0; d<dim; ++d)
                          data.velocity_matrix(local_ordering_u[d][i],
                                               local_ordering_u[d][j])
                            = data.velocity_matrix_permuted(i,j*dim+d);
                      }

                  cell_u->get_dof_indices (data.local_dof_indices_u);
                  constraints_u.distribute_local_to_global
                  (data.velocity_matrix, data.local_dof_indices_u,
                   matrix_u);
                }
              else
                {
                  typename DoFHandler<dim>::active_cell_iterator
                  cell_u_scalar (&triangulation, cell_u->level(), cell_u->index(),
                                 &dof_handler_u_scalar);
                  cell_u_scalar->get_dof_indices(data.local_dof_indices_u_scalar);
                  constraints_u_scalar.distribute_local_to_global
                  (data.velocity_matrix_permuted, data.local_dof_indices_u_scalar,
                   matrix_u);
                }
            }

          // Now treat the case with preconditioner based on linear elements
          else
            {
              // start with velocity
              {
                data.fe_values_lin_u.reinit (cell);
                cell_u->get_dof_indices (data.local_dof_indices_u);

                const std::vector<std::vector<unsigned int> > &
                dof_to_lin_u = integration_helper.dof_to_lin_u;
                const std::vector<std::vector<unsigned int> > &
                quad_to_lin_u = integration_helper.quad_to_lin_u;

                for (unsigned int sub=0; sub<dof_to_lin_u.size(); ++sub)
                  {
                    // TODO: for variable parameters, need to find good values
                    // for density, viscosity, linearized velocity at the
                    // points we consider here (as these live on different
                    // quadrature points for the matrix than for the linear
                    // preconditioner). right now, choose the value in the
                    // first quadrature point.
                    const double actual_nu = use_variable_parameters ?
                                             viscosities[0][vec] : parameters.viscosity;
                    const double actual_rho = use_variable_parameters ?
                                              densities[0][vec] : parameters.density;

                    for (unsigned int d=0; d<dim; ++d)
                      {
                        // get dof_indices for the local linear subelement and
                        // the given coordinate direction
                        for (unsigned int k=0; k<n_dofs; ++k)
                          data.linear_dof_indices[k] =
                            data.local_dof_indices_u[local_ordering_u[d]
                                                     [dof_to_lin_u[sub][k]]];

                        for (unsigned int q=0; q<n_dofs; ++q)
                          {
                            Tensor<2,dim> jac = data.fe_values_lin_u.
                                                inverse_jacobian (quad_to_lin_u[sub][q]);
                            jac = factor_jacobian_u * transpose(jac);
                            const double weight_nu  =
                              data.fe_values_lin_u.JxW(quad_to_lin_u[sub][q]) * actual_nu * tau;
                            const double weight_tau_gd =
                              data.fe_values_lin_u.JxW(quad_to_lin_u[sub][q]) * parameters.tau_grad_div;
                            const double weight_rho =
                              data.fe_values_lin_u.JxW(quad_to_lin_u[sub][q]) * actual_rho;
                            for (unsigned int i=0; i<n_dofs; ++i)
                              {
                                Tensor<1,dim> phi_grad_iq;
                                for (unsigned int e=0; e<dim; ++e)
                                  for (unsigned int f=0; f<dim; ++f)
                                    phi_grad_iq[e] += jac[e][f] *
                                                      integration_helper.grads_unit_cell[i][q][f];

                                double phi_and_grad = parameters.physical_type ==
                                                      FlowParameters::incompressible_stationary ? 0. :
                                                      integration_helper.values_unit_cell[i][q] *
                                                      time_stepping.weight();
                                if (linearized != 0)
                                  {
                                    for (unsigned int e=0; e<dim; ++e)
                                      phi_and_grad += tau * phi_grad_iq[e] *
                                                      linearized[0].first[e][vec];
                                    phi_and_grad += tau *
                                                    integration_helper.values_unit_cell[i][q] *
                                                    linearized[0].second[d][d][vec];
                                  }
                                data_for_val[i][q] = weight_rho * phi_and_grad;

                                Tensor<1,dim> grad_for_test;
                                for (unsigned int e=0; e<dim; ++e)
                                  grad_for_test[e] = phi_grad_iq[e] * weight_nu;
                                grad_for_test[d] += phi_grad_iq[d] * weight_nu;
                                grad_for_test[d] += phi_grad_iq[d] * weight_tau_gd;
                                for (unsigned int e=0; e<dim; ++e)
                                  {
                                    data_for_grad[i][q][e] = jac[0][e] * grad_for_test[0];
                                    for (unsigned int f=1; f<dim; ++f)
                                      data_for_grad[i][q][e] += jac[f][e] * grad_for_test[f];
                                  }
                              }
                          }

                        // compute full matrix-matrix product
                        for (unsigned int i=0; i<n_dofs; ++i)
                          for (unsigned int j=0; j<n_dofs; ++j)
                            {
                              double result = 0;
                              for (unsigned int q=0; q<n_dofs; ++q)
                                {
                                  result +=
                                    integration_helper.values_unit_cell[i][q] *
                                    data_for_val[j][q];
                                  for (unsigned int e=0; e<dim; ++e)
                                    result +=
                                      integration_helper.grads_unit_cell[i][q][e] *
                                      data_for_grad[j][q][e];
                                }
                              data.linear_matrix(i,j) = result;
                            }
                        constraints_u.distribute_local_to_global
                        (data.linear_matrix, data.linear_dof_indices,
                         matrix_u);
                      }
                  }
              }

              // then, do the pressure
              if (!parameters.augmented_taylor_hood)
                {
                  data.fe_values_lin_p.reinit (cell);
                  cell_p->get_dof_indices (data.local_dof_indices_p);
                  const std::vector<std::vector<unsigned int> > &
                  dof_to_lin_p = integration_helper.dof_to_lin_p;
                  const std::vector<std::vector<unsigned int> > &
                  quad_to_lin_p = integration_helper.quad_to_lin_p;

                  for (unsigned int sub=0; sub<dof_to_lin_p.size(); ++sub)
                    {
                      // TODO: for variable parameters, need to find good values
                      // for density, viscosity, linearized velocity at the
                      // points we consider here (as these live on different
                      // quadrature points for the matrix than for the linear
                      // preconditioner). right now, choose the value in the
                      // first quadrature point.
                      const bool variable = use_variable_parameters &&
                                            parameters.linearization != FlowParameters::projection;
                      double rho_weight = variable ?
                                          densities[0][vec] : std::min(parameters.density,
                                                                       parameters.density+parameters.density_diff);
                      rho_weight *= time_stepping.weight();
                      if (parameters.physical_type == FlowParameters::incompressible_stationary)
                        rho_weight = 1.;
                      double actual_nu =
                        ((use_variable_parameters ?
                          viscosities[0][vec] : parameters.viscosity) + parameters.tau_grad_div);
                      if (parameters.physical_type == FlowParameters::incompressible_stationary ||
                          parameters.linearization == FlowParameters::projection)
                        actual_nu = 1.;

                      // get dof_indices for the local linear subelement and the
                      // given coordinate direction

                      for (unsigned int k=0; k<n_dofs; ++k)
                        data.linear_dof_indices[k] =
                          data.local_dof_indices_p[dof_to_lin_p[sub][k]];

                      for (unsigned int q=0; q<n_dofs; ++q)
                        {
                          Tensor<2,dim> jac = data.fe_values_lin_p.inverse_jacobian (quad_to_lin_p[sub][q]);
                          jac = factor_jacobian_p * transpose(jac);
                          const double weight_p_nu  =
                            data.fe_values_lin_p.JxW(quad_to_lin_p[sub][q]) / actual_nu;
                          const double weight_p_rho =
                            data.fe_values_lin_p.JxW(quad_to_lin_p[sub][q]) / rho_weight;
                          for (unsigned int i=0; i<n_dofs; ++i)
                            {
                              data_for_val[i][q] = weight_p_nu *
                                                   integration_helper.values_unit_cell[i][q];
                              if (rho_weight <= 0)
                                continue;

                              Tensor<1,dim> phi_grad_iq;
                              for (unsigned int e=0; e<dim; ++e)
                                for (unsigned int f=0; f<dim; ++f)
                                  phi_grad_iq[e] += jac[e][f] *
                                                    integration_helper.grads_unit_cell[i][q][f];

                              Tensor<1,dim> grad_for_test;
                              for (unsigned int e=0; e<dim; ++e)
                                grad_for_test[e] = phi_grad_iq[e] * weight_p_rho;
                              for (unsigned int e=0; e<dim; ++e)
                                {
                                  data_for_grad[i][q][e] = jac[0][e] * grad_for_test[0];
                                  for (unsigned int f=1; f<dim; ++f)
                                    data_for_grad[i][q][e] += jac[f][e] * grad_for_test[f];
                                }
                            }
                        }

                      if (parameters.density > 0)
                        {
                          // compute full matrix-matrix product
                          for (unsigned int i=0; i<n_dofs; ++i)
                            for (unsigned int j=0; j<n_dofs; ++j)
                              {
                                double result = 0;
                                for (unsigned int q=0; q<n_dofs; ++q)
                                  for (unsigned int e=0; e<dim; ++e)
                                    result += integration_helper.grads_unit_cell[i][q][e] *
                                              data_for_grad[j][q][e];
                                data.linear_matrix(i,j) = result;
                              }
                          constraints_schur_complement.distribute_local_to_global
                          (data.linear_matrix, data.linear_dof_indices,
                           matrix_p);
                        }

                      // compute pressure diagonal
                      if (parameters.precondition_pressure == FlowParameters::p_mass_ilu)
                        {
                          for (unsigned int i=0; i<n_dofs; ++i)
                            for (unsigned int j=0; j<n_dofs; ++j)
                              {
                                double result = 0;
                                for (unsigned int q=0; q<n_dofs; ++q)
                                  result += integration_helper.values_unit_cell[i][q] *
                                            data_for_val[j][q];
                                data.linear_matrix(i,j) = result;
                              }
                          constraints_schur_complement.distribute_local_to_global
                          (data.linear_matrix, data.linear_dof_indices,
                           mass_matrix_p);
                        }
                    }
                }
            }
          if (parameters.precondition_velocity != FlowParameters::u_amg_linear ||
              parameters.augmented_taylor_hood)
            {
              data.fe_values_u.reinit (cell);
              data.fe_values_p.reinit (cell);

              const std::vector<double> &JxW = data.fe_values_u.get_JxW_values();

              // fill matrices with basis function data
              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  if (use_variable_parameters)
                    {
                      Assert (densities[q][vec]>=0,
                              ExcMessage("Density must be non-negative."));
                      Assert (viscosities[q][vec]>0,
                              ExcMessage("Viscosity must be positive."));
                    }
                  const bool variable = use_variable_parameters &&
                                        parameters.linearization != FlowParameters::projection;
                  double rho_weight = variable ?
                                      densities[q][vec] : std::min(parameters.density,
                                                                   parameters.density+parameters.density_diff);
                  rho_weight *= time_stepping.weight();
                  if (parameters.physical_type == FlowParameters::incompressible_stationary)
                    rho_weight = 1.;
                  double actual_nu =
                    ((use_variable_parameters ?
                      viscosities[q][vec] : parameters.viscosity) + parameters.tau_grad_div);
                  if (parameters.physical_type == FlowParameters::incompressible_stationary ||
                      parameters.linearization == FlowParameters::projection)
                    actual_nu = 1.;

                  const double weight_p_nuf = std::sqrt(JxW[q] / actual_nu);
                  const double weight_p_rho = std::sqrt(JxW[q] / rho_weight);
                  for (unsigned int i=0; i<dofs_per_cell_p; ++i)
                    {
                      const Tensor<1,dim> &phi_grad_iq = data.fe_values_p.shape_grad(i,q);
                      if (rho_weight > 0)
                        for (unsigned int d=0; d<dim; ++d)
                          {
                            const double val = phi_grad_iq[d] * weight_p_rho;
                            data.pressure_laplace_part_matrix (dim*q+d, i) = val;
                          }

                      if (parameters.precondition_pressure == FlowParameters::p_mass_ilu ||
                          parameters.augmented_taylor_hood)
                        data.pressure_mass_part_matrix (q, i) = weight_p_nuf *
                                                                data.fe_values_p.shape_value(i,q);
                    }
                }
              cell_p->get_dof_indices (data.local_dof_indices_p);
              if (parameters.density > 0)
                {
                  // pressure Laplace:
                  // dt/rho * grad(phi_i) * grad(phi_i)

                  // Perform the matrix-matrix multiplication with the
                  // transpose of this, i.e., C = A^T * B
                  data.pressure_laplace_part_matrix.
                  Tmmult (data.pressure_matrix,
                          data.pressure_laplace_part_matrix);

                  const double time_weight = parameters.physical_type ==
                                             FlowParameters::incompressible_stationary ? 1. :
                                             time_stepping.weight();
                  if (parameters.augmented_taylor_hood)
                    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                      {
                        if (cell->face(f)->has_children())
                          for (unsigned int subf=0; subf<cell->face(f)->
                               number_of_children(); ++subf)
                            {
                              data.fe_subface_values_p.reinit(cell, f, subf);
                              typename DoFHandler<dim>::cell_iterator neighbor_child
                                = cell_p->neighbor_child_on_subface (f, subf);

                              // In parallel, we might access a neighboring
                              // child cell whose data is not set -> in that
                              // case, use our own value, even though this
                              // leads to a slight non-symmetry in the
                              // pressure Poisson matrix
                              const double density =
                                neighbor_child->is_locally_owned() ?
                                get_face_average_density(neighbor_child, cell->neighbor_face_no(f))
                                :
                                get_face_average_density(cell, f);
                              compute_ip_matrix_q_dg0(data.fe_subface_values_p, f,
                                                      neighbor_child,
                                                      1./(density * time_weight),
                                                      data.pressure_ip_matrix_1,
                                                      data.pressure_ip_matrix_2);
                              neighbor_child->get_dof_indices(data.local_dof_indices_p_neigh);
                              for (unsigned int i=0; i<dofs_per_cell_p; ++i)
                                data.pressure_matrix(dofs_per_cell_p-1,i) -=
                                  data.pressure_ip_matrix_1(0,i);
                              for (unsigned int i=0; i<dofs_per_cell_p; ++i)
                                data.pressure_matrix(i,dofs_per_cell_p-1) -=
                                  data.pressure_ip_matrix_2(i,0);
                              data.local_dof_indices_p_dg0[0] = data.local_dof_indices_p_neigh.back();
                              constraints_schur_complement.distribute_local_to_global
                              (data.pressure_ip_matrix_1, data.local_dof_indices_p_dg0,
                               data.local_dof_indices_p, matrix_p);
                              constraints_schur_complement.distribute_local_to_global
                              (data.pressure_ip_matrix_2, data.local_dof_indices_p,
                               data.local_dof_indices_p_dg0, matrix_p);
                            }
                        else if (cell->at_boundary(f) == false)
                          {
                            data.fe_face_values_p.reinit(cell, f);
                            typename DoFHandler<dim>::cell_iterator neighbor
                              = cell_p->neighbor (f);
                            compute_ip_matrix_q_dg0(data.fe_face_values_p, f,
                                                    neighbor,
                                                    1./(get_face_average_density(cell, f) * time_weight),
                                                    data.pressure_ip_matrix_1,
                                                    data.pressure_ip_matrix_2);
                            neighbor->get_dof_indices(data.local_dof_indices_p_neigh);
                            for (unsigned int i=0; i<dofs_per_cell_p; ++i)
                              data.pressure_matrix(dofs_per_cell_p-1,i) -=
                                data.pressure_ip_matrix_1(0,i);
                            for (unsigned int i=0; i<dofs_per_cell_p; ++i)
                              data.pressure_matrix(i,dofs_per_cell_p-1) -=
                                data.pressure_ip_matrix_2(i,0);
                            data.local_dof_indices_p_dg0[0] = data.local_dof_indices_p_neigh.back();
                            constraints_schur_complement.distribute_local_to_global
                            (data.pressure_ip_matrix_1, data.local_dof_indices_p_dg0,
                             data.local_dof_indices_p, matrix_p);
                            constraints_schur_complement.distribute_local_to_global
                            (data.pressure_ip_matrix_2, data.local_dof_indices_p,
                             data.local_dof_indices_p_dg0, matrix_p);
                          }
                        else if (flow_algorithm.boundary->open_conditions_p.find
                                 (cell->face(f)->boundary_id())
                                 !=
                                 flow_algorithm.boundary->open_conditions_p.end())
                          {
                            // Dirichlet boundary: compute usual weak Nitsche
                            // term
                            data.fe_face_values_p.reinit(cell,f);
                            compute_nitsche_matrix (data.fe_face_values_p, f,
                                                    1./(get_face_average_density(cell, f) * time_weight),
                                                    data.pressure_matrix);
                          }
                      }

                  constraints_schur_complement.distribute_local_to_global
                  (data.pressure_matrix, data.local_dof_indices_p,
                   matrix_p);
                }
              if (parameters.precondition_pressure == FlowParameters::p_mass_ilu ||
                  parameters.augmented_taylor_hood)
                {
                  data.pressure_mass_part_matrix.
                  Tmmult (data.pressure_mass_matrix,
                          data.pressure_mass_part_matrix);
                  constraints_schur_complement.distribute_local_to_global
                  (data.pressure_mass_matrix, data.local_dof_indices_p,
                   mass_matrix_p);
                }
            }
        }
      if (linearized != 0)
        linearized += n_q_points;
      if (use_variable_parameters)
        {
          densities += n_q_points;
          viscosities += n_q_points;
        }
    }
}



template <int dim>
void
NavierStokesPreconditioner<dim>
::assemble_matrices ()
{
  uu_amg.reset();
  uu_ilu.reset();
  uu_ilu_scalar.reset();
  pp_poisson.reset();
  pp_mass.reset();

  matrix->get_matrix_free().initialize_dof_vector(temp_vector, 1);
  matrix->get_matrix_free().initialize_dof_vector(temp_vector2, 1);

  matrix_u = 0;

  if (parameters.density > 0)
    matrix_p = 0;
  if (parameters.precondition_pressure == FlowParameters::p_mass_ilu ||
      parameters.augmented_taylor_hood)
    mass_matrix_p = 0;

  AssemblyData::Preconditioner<dim> scratch_data(*this, flow_algorithm.mapping,
                                                 matrix->get_matrix_free().get_dof_handler(0).get_fe(),
                                                 matrix->get_matrix_free().get_dof_handler(1).get_fe());
  Threads::ThreadLocalStorage<AssemblyData::Preconditioner<dim> > scratch_local (scratch_data);

  unsigned int dummy = 0;
  matrix->get_matrix_free().cell_loop (&NavierStokesPreconditioner<dim>::local_assemble_preconditioner,
                                       this,
                                       scratch_local, dummy);

  matrix_u.compress(VectorOperation::add);
  if (parameters.density > 0)
    matrix_p.compress(VectorOperation::add);
  if (parameters.precondition_pressure == FlowParameters::p_mass_ilu ||
      parameters.augmented_taylor_hood)
    mass_matrix_p.compress(VectorOperation::add);

  // now we need to fix the diagonals for the constraint values: in the
  // matrix-free matrix-vector products we set them to one but the assembly
  // gives them another value. In order to get unit-size eigenvalues which is
  // what the preconditioners try to achieve for the non-constrained values,
  // we have to set them to one here
  if (parameters.precondition_velocity == FlowParameters::u_ilu_scalar)
    {
      IndexSet index = dof_handler_u_scalar.locally_owned_dofs();
      for (unsigned int i=0; i<index.n_elements(); ++i)
        {
          const types::global_dof_index idx = index.nth_index_in_set(i);
          if (constraints_u_scalar.is_constrained(idx))
            matrix_u.set(idx, idx, 1.);
        }
    }
  else
    {
      IndexSet index = matrix->get_matrix_free().get_dof_handler(0).locally_owned_dofs();
      for (unsigned int i=0; i<index.n_elements(); ++i)
        {
          const types::global_dof_index idx = index.nth_index_in_set(i);
          if (constraints_u.is_constrained(idx))
            matrix_u.set(idx, idx, 1.);
        }
    }
  matrix_u.compress(VectorOperation::insert);

  // Set the entries in the pressure Poisson matrix to one, but only those
  // entries that are not constrained on the Schur complement. The Schur
  // complement entries must have the magnitude from the matrix for a
  // well-working preconditioner without spurious eigenvalues on the
  // constrained entries
  if (parameters.density > 0)
    {
      IndexSet index = matrix->get_matrix_free().get_dof_handler(1).locally_owned_dofs();
      for (unsigned int i=0; i<index.n_elements(); ++i)
        {
          const types::global_dof_index idx = index.nth_index_in_set(i);
          if (constraints_schur_complement.is_constrained(idx) &&
              !std::binary_search(constraints_schur_complement_only.begin(),
                                  constraints_schur_complement_only.end(), i))
            matrix_p.set(idx, idx, 1.);
        }
      matrix_p.compress(VectorOperation::insert);
    }
}



template <int dim>
bool
NavierStokesPreconditioner<dim>::is_variable () const
{
  return pp_mass.get() == 0 || do_inner_solves ||
         (parameters.physical_type == FlowParameters::incompressible_stationary);
}



template <int dim>
NavierStokesPreconditioner<dim>::IntegrationHelper::IntegrationHelper()
  :
  n_subelements_u(0),
  n_subelements_p(0)
{}



template <int dim>
void
NavierStokesPreconditioner<dim>::IntegrationHelper::
set_local_ordering_u (const FiniteElement<dim> &fe_u)
{
  local_ordering_u.resize (dim,std::vector<unsigned int>(fe_u.base_element(0).
                                                         dofs_per_cell));
  const unsigned int dofs_per_cell = fe_u.dofs_per_cell;
  for (unsigned int i=0; i<dofs_per_cell; ++i)
    local_ordering_u[fe_u.system_to_component_index(i).first]
    [fe_u.system_to_component_index(i).second] = i;
}



template <int dim>
void
NavierStokesPreconditioner<dim>::IntegrationHelper::
initialize_linear_elements (const FiniteElement<dim> &fe_u,
                            const FiniteElement<dim> &fe_p)
{
  // cache shape values and gradients for linear element
  QGauss<dim> quadrature_linear (2);
  {
    FE_Q<dim> fe_q (1);
    AssertDimension (n_dofs, quadrature_linear.size());
    for (unsigned int i=0; i<n_dofs; ++i)
      for (unsigned int q=0; q<n_dofs; ++q)
        {
          values_unit_cell[i][q] = fe_q.shape_value(i,
                                                    quadrature_linear.point(q));
          grads_unit_cell[i][q] = fe_q.shape_grad(i,
                                                  quadrature_linear.point(q));
        }
  }

  // create quadratures for integration over the subelements of an element
  const unsigned int degree_u = fe_u.base_element(0).degree;
  const unsigned int degree_p = fe_p.degree;
  quadrature_sub_u.reset (new QIterated<dim> (QGauss<1>(2), degree_u));
  quadrature_sub_p.reset (new QIterated<dim> (QGauss<1>(2), degree_p));
  get_indices_sub_quad (degree_u, quad_to_lin_u);
  get_indices_sub_quad (degree_p, quad_to_lin_p);

  // find indices for subelements
  get_indices_sub_elements (fe_u.base_element(0), dof_to_lin_u);
  get_indices_sub_elements (fe_p.base_element(0), dof_to_lin_p);

  AssertDimension (quad_to_lin_u.size(), dof_to_lin_u.size());
  AssertDimension (quad_to_lin_p.size(), dof_to_lin_p.size());
}



template <int dim>
void
NavierStokesPreconditioner<dim>::IntegrationHelper::
get_indices_sub_elements (const FiniteElement<dim> &fe,
                          std::vector<std::vector<unsigned int> > &dof_to_lin) const
{
  const FE_Q<dim> *fe_q = dynamic_cast<const FE_Q<dim>*>(&fe);
  const FE_Q_DG0<dim> *fe_q_dg0 = dynamic_cast<const FE_Q_DG0<dim>*>(&fe);
  Assert(fe_q != 0 || fe_q_dg0 != 0, ExcNotImplemented());
  const unsigned int degree = fe.degree;
  dof_to_lin.resize (Utilities::fixed_power<dim>(degree),
                     std::vector<unsigned int> (n_dofs));
  std::vector<unsigned int> lexicographic (fe.dofs_per_cell);
  if (fe_q != 0)
    lexicographic = fe_q->get_poly_space_numbering_inverse();
  else if (fe_q_dg0 != 0)
    lexicographic = fe_q_dg0->get_poly_space_numbering_inverse();

  Assert (dim == 2 || dim == 3, ExcNotImplemented());
  const unsigned int dofs_per_dim = degree+1;
  const unsigned int dofs_3d = dim==3 ? degree : 1;
  const unsigned int expand_3d = dim==3 ? 2 : 1;
  for (unsigned int elz = 0; elz<dofs_3d; ++elz)
    for (unsigned int ely = 0; ely<degree; ++ely)
      for (unsigned int elx = 0; elx<degree; ++elx)
        {
          const unsigned int index = elx + ely * degree + elz * degree * degree;
          const unsigned int start_index = elx + ely * dofs_per_dim +
                                           elz * dofs_per_dim * dofs_per_dim;
          for (unsigned int i=0; i<expand_3d; ++i)
            for (unsigned int j=0; j<2; ++j)
              for (unsigned int k=0; k<2; ++k)
                {
                  const unsigned int ind = k + j*2 + i*4;
                  const unsigned int loc_index =
                    start_index + k + j*dofs_per_dim +
                    i * dofs_per_dim * dofs_per_dim;
                  AssertIndexRange (loc_index, fe.dofs_per_cell);
                  unsigned int dof_index = lexicographic[loc_index];
                  dof_to_lin[index][ind] = dof_index;
                }
        }
}



template <int dim>
void
NavierStokesPreconditioner<dim>::IntegrationHelper::
get_indices_sub_quad (const unsigned int degree,
                      std::vector<std::vector<unsigned int> > &quad_to_lin) const
{
  quad_to_lin.resize (Utilities::fixed_power<dim>(degree),
                      std::vector<unsigned int> (n_dofs));
  Assert (dim == 2 || dim == 3, ExcNotImplemented());
  const unsigned int points_per_dim = 2*degree;
  const unsigned int points_3d = dim==3 ? degree : 1;
  const unsigned int expand_3d = dim==3 ? 2 : 1;
  for (unsigned int elz = 0; elz<points_3d; ++elz)
    for (unsigned int ely = 0; ely<degree; ++ely)
      for (unsigned int elx = 0; elx<degree; ++elx)
        {
          const unsigned int index = elx + ely * degree + elz * degree * degree;
          const unsigned int start_index = 2 * (elx + ely * points_per_dim +
                                                elz * points_per_dim *
                                                points_per_dim);
          for (unsigned int i=0; i<expand_3d; ++i)
            for (unsigned int j=0; j<2; ++j)
              for (unsigned int k=0; k<2; ++k)
                {
                  const unsigned int ind = k + j*2 + i*4;
                  const unsigned int loc_index =
                    start_index + k + j*points_per_dim +
                    i * points_per_dim * points_per_dim;
                  AssertIndexRange (loc_index,
                                    Utilities::fixed_power<dim>(points_per_dim));
                  quad_to_lin[index][ind] = loc_index;
                }
        }
}



template <int dim>
std::size_t
NavierStokesPreconditioner<dim>::memory_consumption() const
{
  std::size_t memory = matrix_u.memory_consumption();
  memory += matrix_p.memory_consumption();
  memory += constraints_schur_complement.memory_consumption();
  memory += dof_handler_u_scalar.memory_consumption();
  memory += constraints_u_scalar.memory_consumption();
  memory += temp_vector.memory_consumption()*2;
  memory += 2*pressure_diagonal_preconditioner.get_vector().memory_consumption();
  if (uu_ilu.get() != 0)
    memory += matrix_u.memory_consumption();
  if (uu_amg.get() != 0)
    memory += uu_amg->memory_consumption();
  if (pp_poisson.get() != 0)
    memory += pp_poisson->memory_consumption();
  memory += mass_matrix_p.memory_consumption();
  return memory;
}



template <int dim>
void
NavierStokesPreconditioner<dim>::
print_memory_consumption(std::ostream &stream) const
{
  stream << "| Preconditioner matrices: "
         << 1e-6*double(matrix_u.memory_consumption()+
                        matrix_p.memory_consumption()+
                        mass_matrix_p.memory_consumption()+
                        pressure_diagonal_preconditioner.get_vector().memory_consumption())
         << " MB\n";
  if (uu_ilu.get() != 0 || uu_ilu_scalar.get() != 0)
    stream << "| ILU preconditioner for uu block: "
           << 1e-6*double(matrix_u.memory_consumption())
           << " MB\n";
  else if (uu_amg.get() != 0)
    stream << "| AMG preconditioner for uu block: "
           << 1e-6*double(uu_amg->memory_consumption())
           << " MB\n";
  if (pp_poisson.get() != 0)
    stream << "| AMG preconditioner for pp block: "
           << 1e-6*double(pp_poisson->memory_consumption())
           << " MB\n";
}



template <int dim>
std::pair<Utilities::MPI::MinMaxAvg[5],unsigned int>
NavierStokesPreconditioner<dim>::get_timer_statistics() const
{
  std::pair<Utilities::MPI::MinMaxAvg[5],unsigned int> data;
  data.second = precond_timer.first;
  precond_timer.first = 0;
  for (unsigned int i=0; i<5; ++i)
    {
      data.first[i] =
        Utilities::MPI::min_max_avg(precond_timer.second[i],
                                    temp_vector.get_mpi_communicator());
      precond_timer.second[i] = 0;
    }
  return data;
}




// explicit instantiations
template class NavierStokesPreconditioner<2>;
template class NavierStokesPreconditioner<3>;
