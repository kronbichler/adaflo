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

#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <adaflo/navier_stokes_matrix.h>


// TODO:

// - Need to balance different size of pressure and velocity part (should have
// time_step_weight both in divergence and pressure gradient,
// time_step_weight^2 in Schur complement part)

// -BDF-2 and implicit Euler are correct. Crank-Nicolson has some problems and
// is only 1st order, not second order as one would expect. Check!

// - Other time stepping schemes (fractional step) not checked!

/* ----------------- Implementation Navier-Stokes matrix ----------------- */



template <int dim>
NavierStokesMatrix<dim>::NavierStokesMatrix(
  const FlowParameters &                                 parameters,
  const unsigned int                                     dof_index_u,
  const unsigned int                                     dof_index_p,
  const unsigned int                                     quad_index_u,
  const unsigned int                                     quad_index_p,
  const LinearAlgebra::distributed::BlockVector<double> &solution_old,
  const LinearAlgebra::distributed::BlockVector<double> &solution_old_old)
  : matrix_free(0)
  , time_stepping(0)
  , parameters(parameters)
  , dof_index_u(dof_index_u)
  , dof_index_p(dof_index_p)
  , quad_index_u(quad_index_u)
  , quad_index_p(quad_index_p)
  , solution_old(solution_old)
  , solution_old_old(solution_old_old)
{}



#define EXPAND_OPERATIONS(OPERATION)                                \
  const unsigned int degree_p =                                     \
    matrix_free->get_dof_handler(dof_index_p).get_fe().degree;      \
                                                                    \
  AssertThrow(degree_p >= 1 && degree_p <= 5, ExcNotImplemented()); \
  if (parameters.use_simplex_mesh)                                  \
    OPERATION(-1);                                                  \
  else if (degree_p == 1)                                           \
    OPERATION(1);                                                   \
  else if (degree_p == 2)                                           \
    OPERATION(2);                                                   \
  else if (degree_p == 3)                                           \
    OPERATION(3);                                                   \
  else if (degree_p == 4)                                           \
    OPERATION(4);                                                   \
  else if (degree_p == 5)                                           \
    OPERATION(5);                                                   \
  else                                                              \
    AssertThrow(false, ExcNotImplemented());


template <int dim>
void
NavierStokesMatrix<dim>::initialize(const MatrixFree<dim> &matrix_free_in,
                                    const TimeStepping &   time_stepping_in,
                                    const bool             pressure_average_fix)
{
  matrix_free                       = &matrix_free_in;
  time_stepping                     = &time_stepping_in;
  const unsigned int n_cell_batches = matrix_free->n_cell_batches();
  const unsigned int n_q_points     = matrix_free->get_n_q_points(quad_index_u);
  const unsigned int size           = n_cell_batches * n_q_points;

  const bool use_variable_coefficients =
    parameters.density_diff != 0 || parameters.viscosity_diff != 0;

  if (use_variable_coefficients == true)
    {
      variable_densities.resize(size, make_vectorized_array<double>(parameters.density));
      variable_viscosities.resize(size,
                                  make_vectorized_array<double>(parameters.viscosity));
    }
  if (parameters.linearization != FlowParameters::coupled_velocity_explicit)
    linearized_velocities.resize(size);

  for (unsigned int mode = 0; mode < 2; ++mode)
    {
      pressure_constant_mode_weights[mode].reinit(0);
      pressure_constant_modes[mode].reinit(0);
      inverse_pressure_average_weights[mode] = 0.;
    }

  if (pressure_average_fix == true || parameters.augmented_taylor_hood)
    {
      LinearAlgebra::distributed::Vector<double> pres_mass;
      matrix_free->initialize_dof_vector(pres_mass, dof_index_p);
      {
        unsigned int dummy = 0;
#define OPERATION(degree_p)                                                  \
  matrix_free->cell_loop(                                                    \
    &NavierStokesMatrix<dim>::template local_pressure_mass_weight<degree_p>, \
    this,                                                                    \
    pres_mass,                                                               \
    dummy)

        EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
      }

      std::vector<std::vector<bool>> constant_modes;
      DoFTools::extract_constant_modes(matrix_free->get_dof_handler(dof_index_p),
                                       std::vector<bool>(1, true),
                                       constant_modes);

      for (unsigned int mode = 0; mode < 2; ++mode)
        {
          if (mode == 0 && !pressure_average_fix)
            continue;
          if (mode == 1 && !parameters.augmented_taylor_hood)
            continue;

          matrix_free->initialize_dof_vector(pressure_constant_modes[mode], dof_index_p);
          matrix_free->initialize_dof_vector(pressure_constant_mode_weights[mode],
                                             dof_index_p);
          AssertDimension(pressure_constant_modes[mode].local_size(),
                          constant_modes[mode].size());
          for (unsigned int i = 0; i < pressure_constant_modes[mode].local_size(); ++i)
            if (constant_modes[mode][i])
              {
                pressure_constant_modes[mode].local_element(i) = 1.;
                pressure_constant_mode_weights[mode].local_element(i) =
                  pres_mass.local_element(i);
              }
          // delete constrained degrees of freedom from pressure constant modes
          for (std::vector<unsigned int>::const_iterator it =
                 matrix_free->get_constrained_dofs(dof_index_p).begin();
               it != matrix_free->get_constrained_dofs(dof_index_p).end();
               ++it)
            pressure_constant_modes[mode].local_element(*it) = 0;

          inverse_pressure_average_weights[mode] =
            1. / (pressure_constant_modes[mode] * pressure_constant_mode_weights[mode]);
        }
    }
}



template <int dim>
void
NavierStokesMatrix<dim>::clear()
{
  matrix_free   = 0;
  time_stepping = 0;
  variable_densities.clear();
  variable_viscosities.clear();
  linearized_velocities.clear();
  variable_densities_preconditioner.clear();
  variable_viscosities_preconditioner.clear();
  linearized_velocities_preconditioner.clear();
}



template <int dim>
void
NavierStokesMatrix<dim>::apply_pressure_average_projection(
  LinearAlgebra::distributed::Vector<double> &vector) const
{
  if (parameters.linearization != FlowParameters::projection &&
      parameters.physical_type != FlowParameters::incompressible_stationary)
    for (unsigned int mode = 0; mode < 2; ++mode)
      if (pressure_constant_mode_weights[mode].size() > 0)
        {
          double product = pressure_constant_mode_weights[mode] * vector;
          vector.add(-product * inverse_pressure_average_weights[mode],
                     pressure_constant_modes[mode]);
        }
}



template <int dim>
void
NavierStokesMatrix<dim>::apply_pressure_shift(
  const double                                shift,
  LinearAlgebra::distributed::Vector<double> &pressure) const
{
  AssertDimension(pressure_constant_modes[0].size(), pressure.size());
  pressure.add(shift, pressure_constant_modes[0]);
}



template <int dim>
void
NavierStokesMatrix<dim>::vmult(
  LinearAlgebra::distributed::BlockVector<double> &      dst,
  const LinearAlgebra::distributed::BlockVector<double> &src) const
{
  Timer time;
  Assert(matrix_free != 0, ExcNotInitialized());
  dst = 0;

  // since there is a template argument in the local function (for efficiency
  // reasons) but no template on the NavierStokes matrix (to allow for change
  // in the degree from a parameter file), we need to select the correct
  // function corresponding to the degree at this point
#define OPERATION(degree_p)                                                  \
  matrix_free->cell_loop(&NavierStokesMatrix<dim>::template local_operation< \
                           degree_p,                                         \
                           LinearAlgebra::distributed::BlockVector<double>,  \
                           NavierStokesOps::vmult>,                          \
                         this,                                               \
                         dst,                                                \
                         src)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  for (unsigned int block = 0; block < 2; ++block)
    {
      // diagonal values of constrained degrees of freedom set to 1
      const std::vector<unsigned int> &constrained_dofs =
        matrix_free->get_constrained_dofs(block == 0 ? dof_index_u : dof_index_p);
      const double sign = block == 0 ? 1. : -1.;
      for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
        dst.block(block).local_element(constrained_dofs[i]) =
          (sign * src.block(block).local_element(constrained_dofs[i]));
    }

  apply_pressure_average_projection(dst.block(1));

  matvec_timer.first++;
  matvec_timer.second += time.wall_time();
}


template <int dim>
void
NavierStokesMatrix<dim>::residual(
  LinearAlgebra::distributed::BlockVector<double> &      system_rhs,
  const LinearAlgebra::distributed::BlockVector<double> &src,
  const LinearAlgebra::distributed::BlockVector<double> &user_rhs) const
{
  Assert(matrix_free != 0, ExcNotInitialized());

#define OPERATION(degree_p)                                                  \
  matrix_free->cell_loop(&NavierStokesMatrix<dim>::template local_operation< \
                           degree_p,                                         \
                           LinearAlgebra::distributed::BlockVector<double>,  \
                           NavierStokesOps::residual>,                       \
                         this,                                               \
                         system_rhs,                                         \
                         src)

  EXPAND_OPERATIONS(OPERATION);

#undef OPERATION

  // for Newton's method, the residual needs to have negative sign. We
  // assembled only positive terms which makes it easy to use the same code
  // for residual assembly and matrix-vector product, so change the sign here
  // before solving the linear system
  system_rhs.sadd(-1., 1., user_rhs);
}



// multiplication and addition with divergence matrix (block(1,0) in NS
// matrix)

template <int dim>
void
NavierStokesMatrix<dim>::divergence_vmult_add(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const bool                                        weight_by_viscosity) const
{
  Assert(matrix_free != 0, ExcNotInitialized());

  if (weight_by_viscosity)
    {
#define OPERATION(degree_p) \
  matrix_free->cell_loop(   \
    &NavierStokesMatrix<dim>::template local_divergence<degree_p, true>, this, dst, src)

      EXPAND_OPERATIONS(OPERATION);

#undef OPERATION
    }
  else
    {
#define OPERATION(degree_p)                                               \
  matrix_free->cell_loop(                                                 \
    &NavierStokesMatrix<dim>::template local_divergence<degree_p, false>, \
    this,                                                                 \
    dst,                                                                  \
    src)

      EXPAND_OPERATIONS(OPERATION);

#undef OPERATION
    }
}


// multiplication with velocity block ((0,0) block in matrix)

template <int dim>
void
NavierStokesMatrix<dim>::velocity_vmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src) const
{
  Assert(matrix_free != 0, ExcNotInitialized());
  dst = 0;

  // in case we use an AMG preconditioner, the velocity multiplication needs
  // to be on the same matrix that the preconditioner is based upon ->
  // temporarily exchange those data fields
  if (linearized_velocities_preconditioner.size() > 0)
    {
      linearized_velocities.swap(linearized_velocities_preconditioner);
      variable_densities.swap(variable_densities_preconditioner);
      variable_viscosities.swap(variable_viscosities_preconditioner);
    }

#define OPERATION(degree_p)                                                  \
  matrix_free->cell_loop(&NavierStokesMatrix<dim>::template local_operation< \
                           degree_p,                                         \
                           LinearAlgebra::distributed::Vector<double>,       \
                           NavierStokesOps::vmult_velocity>,                 \
                         this,                                               \
                         dst,                                                \
                         src)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  if (linearized_velocities.size() > 0)
    {
      linearized_velocities.swap(linearized_velocities_preconditioner);
      variable_densities.swap(variable_densities_preconditioner);
      variable_viscosities.swap(variable_viscosities_preconditioner);
    }

  // diagonal values of constrained degrees of freedom set to 1
  const std::vector<unsigned int> &constrained_dofs =
    matrix_free->get_constrained_dofs(dof_index_u);
  for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
    dst.local_element(constrained_dofs[i]) = src.local_element(constrained_dofs[i]);
}



template <int dim>
void
NavierStokesMatrix<dim>::pressure_poisson_vmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src) const
{
  Assert(matrix_free != 0, ExcNotInitialized());
  dst = 0;

  // in case we use an AMG preconditioner, we need to make sure the
  // multiplication is done on the correct matrix -> temporarily use the field
  // for the variable densities
  if (variable_densities_preconditioner.size() > 0)
    variable_densities.swap(variable_densities_preconditioner);

#define OPERATION(degree_p) \
  matrix_free->cell_loop(   \
    &NavierStokesMatrix<dim>::template local_pressure_poisson<degree_p>, this, dst, src)

  EXPAND_OPERATIONS(OPERATION);

#undef OPERATION

  if (variable_densities_preconditioner.size() > 0)
    variable_densities.swap(variable_densities_preconditioner);

  // diagonal values of constrained degrees of freedom set to 1
  const std::vector<unsigned int> &constrained_dofs =
    matrix_free->get_constrained_dofs(dof_index_p);
  for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
    dst.local_element(constrained_dofs[i]) = src.local_element(constrained_dofs[i]);
}



template <int dim>
void
NavierStokesMatrix<dim>::pressure_mass_vmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src) const
{
  Assert(matrix_free != 0, ExcNotInitialized());
  dst = 0;

  if (variable_viscosities_preconditioner.size() > 0)
    variable_viscosities.swap(variable_viscosities_preconditioner);

#define OPERATION(degree_p) \
  matrix_free->cell_loop(   \
    &NavierStokesMatrix<dim>::template local_pressure_mass<degree_p>, this, dst, src)

  EXPAND_OPERATIONS(OPERATION);

#undef OPERATION

  if (variable_viscosities.size() > 0)
    variable_viscosities.swap(variable_viscosities_preconditioner);

  const std::vector<unsigned int> &constrained_dofs =
    matrix_free->get_constrained_dofs(dof_index_p);
  for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
    dst.local_element(constrained_dofs[i]) = src.local_element(constrained_dofs[i]);

  if (pressure_constant_mode_weights[1].size() > 0 &&
      parameters.linearization != FlowParameters::projection)
    {
      double weight = pressure_constant_mode_weights[1] * dst;
      dst.add(-weight * inverse_pressure_average_weights[1], pressure_constant_modes[1]);
    }
}



template <int dim>
void
NavierStokesMatrix<dim>::pressure_convdiff_vmult(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src) const
{
  Assert(matrix_free != 0, ExcNotInitialized());
  dst = 0;

#define OPERATION(degree_p)                                               \
  matrix_free->cell_loop(                                                 \
    &NavierStokesMatrix<dim>::template local_pressure_convdiff<degree_p>, \
    this,                                                                 \
    dst,                                                                  \
    src)

  EXPAND_OPERATIONS(OPERATION);

#undef OPERATION

  const std::vector<unsigned int> &constrained_dofs =
    matrix_free->get_constrained_dofs(dof_index_p);
  for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
    dst.local_element(constrained_dofs[i]) = src.local_element(constrained_dofs[i]);
}



#undef EXPAND_OPERATIONS


// here come some helper functions that allow us unified access through
// different version of the local function (Vector vs BlockVector within the
// same function in local_operation)

namespace
{
  template <typename FEEval>
  inline void
  get_velocity_values(FEEval &                                               fe_eval,
                      const LinearAlgebra::distributed::BlockVector<double> &vec)
  {
    fe_eval.read_dof_values(vec.block(0));
  }

  template <typename FEEval>
  inline void
  get_velocity_values_plain(FEEval &fe_eval,
                            const LinearAlgebra::distributed::BlockVector<double> &vec)
  {
    fe_eval.read_dof_values_plain(vec.block(0));
  }

  template <typename FEEval>
  inline void
  get_velocity_values(FEEval &                                          fe_eval,
                      const LinearAlgebra::distributed::Vector<double> &vec)
  {
    fe_eval.read_dof_values(vec);
  }

  template <typename FEEval>
  inline void
  get_velocity_values_plain(FEEval &                                          fe_eval,
                            const LinearAlgebra::distributed::Vector<double> &vec)
  {
    fe_eval.read_dof_values(vec);
  }

  template <typename FEEval>
  inline void
  get_pressure_values(FEEval &                                               fe_eval,
                      const LinearAlgebra::distributed::BlockVector<double> &vec)
  {
    fe_eval.read_dof_values(vec.block(1));
  }

  template <typename FEEval>
  inline void
  get_pressure_values_plain(FEEval &fe_eval,
                            const LinearAlgebra::distributed::BlockVector<double> &vec)
  {
    fe_eval.read_dof_values_plain(vec.block(1));
  }

  template <typename FEEval>
  inline void
  get_pressure_values(FEEval &                                          fe_eval,
                      const LinearAlgebra::distributed::Vector<double> &vec)
  {
    fe_eval.read_dof_values(vec);
  }

  template <typename FEEval>
  inline void
  get_pressure_values_plain(FEEval &                                          fe_eval,
                            const LinearAlgebra::distributed::Vector<double> &vec)
  {
    fe_eval.read_dof_values(vec);
  }

  template <typename FEEval>
  inline void
  distribute_velocity_ltg(FEEval &                                         fe_eval,
                          LinearAlgebra::distributed::BlockVector<double> &vec)
  {
    fe_eval.distribute_local_to_global(vec.block(0));
  }

  template <typename FEEval>
  inline void
  distribute_velocity_ltg(FEEval &                                    fe_eval,
                          LinearAlgebra::distributed::Vector<double> &vec)
  {
    fe_eval.distribute_local_to_global(vec);
  }

  template <typename FEEval>
  inline void
  distribute_pressure_ltg(FEEval &                                         fe_eval,
                          LinearAlgebra::distributed::BlockVector<double> &vec)
  {
    fe_eval.distribute_local_to_global(vec.block(1));
  }

  template <typename FEEval>
  inline void
  distribute_pressure_ltg(FEEval &                                    fe_eval,
                          LinearAlgebra::distributed::Vector<double> &vec)
  {
    fe_eval.distribute_local_to_global(vec);
  }
} // namespace



// Here comes the implementation of local operations. It is templated in the
// degree of the pressure (velocity degree is one higher in this
// implementation), the vector type (allowing for BlockVector in vmult and a
// simple vector in velocity_vmult for the (0,0) subsystem), and the actual
// operation.

template <int dim>
template <int degree_p, typename VectorType, int LocalOps>
void
NavierStokesMatrix<dim>::local_operation(
  const MatrixFree<dim> &                      data,
  VectorType &                                 dst,
  const VectorType &                           src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  typedef VectorizedArray<double>                                            vector_t;
  FEEvaluation<dim, degree_p == -1 ? -1 : (degree_p + 1), degree_p + 2, dim> velocity(
    data, dof_index_u, quad_index_u);
  FEEvaluation<dim, degree_p, degree_p + 2, 1> pressure(data, dof_index_p, quad_index_u);

  FEEvaluation<dim, degree_p == -1 ? -1 : (degree_p + 1), degree_p + 2, dim> old(
    data, dof_index_u, quad_index_u);
  FEEvaluation<dim, degree_p == -1 ? -1 : (degree_p + 1), degree_p + 2, dim> old_old(
    data, dof_index_u, quad_index_u);

  // get variables for the time step
  const vector_t time_step_weight = make_vectorized_array<double>(
    parameters.physical_type == FlowParameters::incompressible ? time_stepping->weight() :
                                                                 0.);
  const vector_t time_step_weight_old =
    make_vectorized_array<double>(time_stepping->weight_old());
  const vector_t time_step_weight_old_old =
    make_vectorized_array<double>(time_stepping->weight_old_old());
  const vector_t tau1 = make_vectorized_array<double>(time_stepping->tau1());
  Assert(time_stepping->tau2() == 0., ExcNotImplemented());

  velocity_stored *linearized =
    linearized_velocities.size() > 0 ?
      &linearized_velocities[cell_range.first * velocity.n_q_points] :
      0;

  const bool      use_variable_coefficients = variable_densities.size() > 0;
  const vector_t *rho_values =
    use_variable_coefficients ? begin_densities(cell_range.first) : 0;
  const vector_t *mu_values =
    use_variable_coefficients ? begin_viscosities(cell_range.first) : 0;

  const bool need_extrapolated_velocity =
    parameters.linearization == FlowParameters::projection ||
    parameters.linearization == FlowParameters::coupled_velocity_semi_implicit ||
    parameters.linearization == FlowParameters::coupled_velocity_explicit;

  // 0:   NS in convective form
  // 0.5: NS in skew-symmetric form
  // 1:   NS in conservative form
  const vector_t conservative_form = make_vectorized_array<double>(0.5);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // get velocity part. for the residual computation, we do not want to
      // resolve constraints (in order to respect non-zero boundary
      // conditions), whereas we need to resolve them for matrix-vector
      // products

      velocity.reinit(cell);
      if (LocalOps == NavierStokesOps::residual)
        get_velocity_values_plain(velocity, src);
      else
        get_velocity_values(velocity, src);

      velocity.evaluate(parameters.physical_type != FlowParameters::stokes, true);

      if (LocalOps == NavierStokesOps::residual &&
          parameters.physical_type == FlowParameters::incompressible)
        {
          old.reinit(cell);
          old.read_dof_values_plain(solution_old.block(0));
          old.evaluate(true, need_extrapolated_velocity);
          old_old.reinit(cell);
          old_old.read_dof_values_plain(solution_old_old.block(0));
          old_old.evaluate(true, need_extrapolated_velocity);
        }

      // get pressure part
      pressure.reinit(cell);
      if (LocalOps == NavierStokesOps::residual || LocalOps == NavierStokesOps::vmult)
        {
          if (LocalOps == NavierStokesOps::residual)
            get_pressure_values_plain(pressure, src);
          else
            get_pressure_values(pressure, src);
          pressure.evaluate(true, false);
        }

      // loop over all quadrature points and implement the Navier-Stokes
      // operation

      for (unsigned int q = 0; q < velocity.n_q_points; ++q)
        {
          Tensor<2, dim, vector_t> grad_u     = velocity.get_gradient(q);
          vector_t                 divergence = trace(grad_u);

          if (parameters.physical_type != FlowParameters::stokes)
            {
              // variable parameters if present
              const vector_t rho = use_variable_coefficients ?
                                     rho_values[q] :
                                     make_vectorized_array<double>(parameters.density);
              Tensor<1, dim, vector_t> val_u = velocity.get_value(q);

              Tensor<1, dim, vector_t> conv = val_u * time_step_weight;

              // For the convective term, need to distinguish between residual
              // computations and matrix-vector products because of
              // linearization and various time stepping possibilities.

              // The convective part uses the skew-symmetric form:
              // u * nabla u + conservative_form u (div u).
              if (LocalOps == NavierStokesOps::residual)
                {
                  if (parameters.physical_type !=
                      FlowParameters::incompressible_stationary)
                    conv += old.get_value(q) * time_step_weight_old +
                            old_old.get_value(q) * time_step_weight_old_old;

                  // Also store the current velocity and velocity gradient (or
                  // divergence, depending on the linearization scheme). A
                  // complication is that we cannot always use the
                  // extrapolated velocity stored in the 'solution' vector
                  // that is used as a source due to boundary conditions, so
                  // need to extrapolate those velocities.
                  if (need_extrapolated_velocity)
                    {
                      Tensor<2, dim, vector_t> old_grad     = old.get_gradient(q);
                      Tensor<2, dim, vector_t> old_old_grad = old_old.get_gradient(q);
                      for (unsigned int d = 0; d < dim; ++d)
                        for (unsigned int e = 0; e < dim; ++e)
                          old_grad[d][e] = time_stepping->extrapolate(old_grad[d][e],
                                                                      old_old_grad[d][e]);
                      const vector_t           extrapol_divergence = trace(old_grad);
                      Tensor<1, dim, vector_t> old_val             = old.get_value(q);
                      Tensor<1, dim, vector_t> old_old_val         = old_old.get_value(q);
                      for (unsigned int d = 0; d < dim; ++d)
                        old_val[d] =
                          time_stepping->extrapolate(old_val[d], old_old_val[d]);
                      if (parameters.linearization ==
                          FlowParameters::coupled_velocity_explicit)
                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            vector_t res =
                              conservative_form * extrapol_divergence * old_val[d];
                            for (unsigned int e = 0; e < dim; ++e)
                              res += old_val[e] * old_grad[d][e];
                            conv[d] += tau1 * res;
                          }
                      else
                        {
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              vector_t res =
                                conservative_form * extrapol_divergence * val_u[d];
                              for (unsigned int e = 0; e < dim; ++e)
                                res += old_val[e] * grad_u[d][e];
                              conv[d] += tau1 * res;

                              linearized[q].first[d] = old_val[d];
                            }
                          linearized[q].second[0][0] = extrapol_divergence;
                        }
                    }
                  else
                    {
                      for (unsigned int d = 0; d < dim; ++d)
                        {
                          vector_t res = conservative_form * divergence * val_u[d];
                          for (unsigned int e = 0; e < dim; ++e)
                            res += val_u[e] * grad_u[d][e];
                          conv[d] += tau1 * res;

                          linearized[q].first[d] = val_u[d];
                        }
                      if (parameters.linearization ==
                          FlowParameters::coupled_implicit_newton)
                        linearized[q].second = grad_u;
                      else
                        linearized[q].second[0][0] = divergence;
                    }
                }
              // matrix-vector cases
              else if (parameters.linearization ==
                       FlowParameters::coupled_implicit_newton)
                {
                  vector_t factor1 = conservative_form * divergence;
                  vector_t factor2 = conservative_form * trace(linearized[q].second);
                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      vector_t res =
                        factor1 * linearized[q].first[d] + factor2 * val_u[d];
                      for (unsigned int e = 0; e < dim; ++e)
                        res += (linearized[q].first[e] * grad_u[d][e] +
                                val_u[e] * linearized[q].second[d][e]);
                      conv[d] += tau1 * res;
                    }
                }
              else if (parameters.linearization !=
                       FlowParameters::coupled_velocity_explicit)
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    vector_t res =
                      conservative_form * linearized[q].second[0][0] * val_u[d];
                    for (unsigned int e = 0; e < dim; ++e)
                      res += linearized[q].first[e] * grad_u[d][e];
                    conv[d] += tau1 * res;
                  }
              conv *= rho;
              velocity.submit_value(conv, q);
            }

          // symmetrize gradient and multiply by viscosity
          const vector_t tmu = (use_variable_coefficients ?
                                  mu_values[q] :
                                  make_vectorized_array<double>(parameters.viscosity)) *
                               tau1;
          const vector_t tmu_times_2 = 2. * tmu;


          // get divergence, extract pressure, integrate (p, -div (u)), which
          // writes into the pressure field
          vector_t pres;
          if (LocalOps != NavierStokesOps::vmult_velocity)
            {
              pres = pressure.get_value(q);

              // divergence, tested by pressure test function
              pressure.submit_value(-divergence, q);
            }

          vector_t sym;
          switch (dim)
            {
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
              case 3:
                sym          = tmu * (grad_u[0][2] + grad_u[2][0]);
                grad_u[0][2] = sym;
                grad_u[2][0] = sym;
                sym          = tmu * (grad_u[1][2] + grad_u[2][1]);
                grad_u[1][2] = sym;
                grad_u[2][1] = sym;
              // fall through to part that is also present also in 2D
              case 2:
                sym          = tmu * (grad_u[0][1] + grad_u[1][0]);
                grad_u[0][1] = sym;
                grad_u[1][0] = sym;
                break;
              default:
                Assert(false, ExcNotImplemented());
#pragma GCC diagnostic push
            }
          // add pressure
          for (unsigned int d = 0; d < dim; ++d)
            {
              grad_u[d][d] =
                tmu_times_2 * grad_u[d][d] + parameters.tau_grad_div * divergence;
              if (LocalOps == NavierStokesOps::vmult ||
                  LocalOps == NavierStokesOps::residual)
                grad_u[d][d] -= pres;
            }

          velocity.submit_gradient(grad_u, q);
        }

      // finally, integrate velocity and pressure and increase pointers to
      // linearization data and rho and mu values
      velocity.integrate(parameters.physical_type != FlowParameters::stokes, true);
      distribute_velocity_ltg(velocity, dst);
      if (LocalOps != NavierStokesOps::vmult_velocity &&
          parameters.linearization != FlowParameters::projection)
        {
          pressure.integrate(true, false);
          distribute_pressure_ltg(pressure, dst);
        }
      linearized += velocity.n_q_points;
      if (use_variable_coefficients)
        {
          rho_values += velocity.n_q_points;
          mu_values += velocity.n_q_points;
        }
    }
}



template <int dim>
template <int degree_p, const bool weight_by_viscosity>
void
NavierStokesMatrix<dim>::local_divergence(
  const MatrixFree<dim> &                           data,
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  FEEvaluation<dim, degree_p == -1 ? -1 : (degree_p + 1), degree_p + 2, dim> velocity(
    data, dof_index_u, quad_index_u);
  FEEvaluation<dim, degree_p, degree_p + 2, 1> pressure(data, dof_index_p, quad_index_u);

  const VectorizedArray<double> *mu_values =
    variable_viscosities.empty() ? 0 : begin_viscosities(cell_range.first);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      pressure.reinit(cell);
      velocity.reinit(cell);
      if (parameters.linearization == FlowParameters::projection)
        velocity.read_dof_values_plain(src);
      else
        velocity.read_dof_values(src);
      velocity.evaluate(false, true, false);

      for (unsigned int q = 0; q < velocity.n_q_points; ++q)
        {
          VectorizedArray<double> weight =
            weight_by_viscosity ?
              (variable_viscosities.empty() ?
                 make_vectorized_array<double>(-parameters.viscosity) :
                 -mu_values[q]) :
              make_vectorized_array(-1.);

          pressure.submit_value(weight * velocity.get_divergence(q), q);
        }

      pressure.integrate(true, false);
      pressure.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int degree_p>
void
NavierStokesMatrix<dim>::local_pressure_poisson(
  const MatrixFree<dim> &                           data,
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  typedef VectorizedArray<double> vector_t;
  const bool                      use_variable_coefficients =
    variable_densities.size() > 0 &&
    parameters.linearization != FlowParameters::projection;

  if (use_variable_coefficients &&
      parameters.physical_type != FlowParameters::incompressible_stationary)
    {
      FEEvaluation<dim, degree_p, degree_p + 2, 1> pressure(data,
                                                            dof_index_p,
                                                            quad_index_u);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          pressure.reinit(cell);
          get_pressure_values(pressure, src);
          pressure.evaluate(false, true, false);

          for (unsigned int q = 0; q < pressure.n_q_points; ++q)
            pressure.submit_gradient(pressure.get_gradient(q) *
                                       (1. / (time_stepping->weight() *
                                              begin_densities(cell)[q])),
                                     q);

          pressure.integrate(false, true);
          pressure.distribute_local_to_global(dst);
        }
    }
  else
    {
      FEEvaluation<dim, degree_p, degree_p + 1, 1> pressure(data,
                                                            dof_index_p,
                                                            quad_index_p);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          pressure.reinit(cell);
          get_pressure_values(pressure, src);
          pressure.evaluate(false, true, false);

          const vector_t rho_value =
            use_variable_coefficients ?
              begin_densities(cell)[data.get_n_q_points(0) / 2] :
              make_vectorized_array(
                std::min(parameters.density,
                         parameters.density + parameters.density_diff));
          const vector_t coefficient =
            parameters.physical_type == FlowParameters::incompressible_stationary ?
              make_vectorized_array<double>(1.) :
              1. / (time_stepping->weight() * rho_value);

          for (unsigned int q = 0; q < pressure.n_q_points; ++q)
            pressure.submit_gradient(pressure.get_gradient(q) * coefficient, q);

          pressure.integrate(false, true);
          pressure.distribute_local_to_global(dst);
        }
    }
}



template <int dim>
template <int degree_p>
void
NavierStokesMatrix<dim>::local_pressure_mass(
  const MatrixFree<dim> &                           data,
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  typedef VectorizedArray<double>              vector_t;
  FEEvaluation<dim, degree_p, degree_p + 1, 1> pressure(data, dof_index_p, quad_index_p);

  const bool use_variable_coefficients = variable_viscosities.size() > 0;

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      pressure.reinit(cell);
      get_pressure_values(pressure, src);
      pressure.evaluate(true, false, false);

      const vector_t mu_value = use_variable_coefficients ?
                                  begin_viscosities(cell)[data.get_n_q_points(0) / 2] :
                                  make_vectorized_array(parameters.viscosity);
      const vector_t coefficient =
        (parameters.linearization == FlowParameters::projection ||
         parameters.physical_type == FlowParameters::incompressible_stationary) ?
          make_vectorized_array<double>(1.) :
          1. / (mu_value + parameters.tau_grad_div);

      for (unsigned int q = 0; q < pressure.n_q_points; ++q)
        pressure.submit_value(pressure.get_value(q) * coefficient, q);

      pressure.integrate(true, false);
      pressure.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int degree_p>
void
NavierStokesMatrix<dim>::local_pressure_mass_weight(
  const MatrixFree<dim> &                     data,
  LinearAlgebra::distributed::Vector<double> &dst,
  const unsigned int &,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  FEEvaluation<dim, degree_p, degree_p + 1, 1> pressure(data, dof_index_p, quad_index_p);
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      pressure.reinit(cell);
      VectorizedArray<double> one;
      one = 1.;
      for (unsigned int q = 0; q < pressure.n_q_points; ++q)
        pressure.submit_value(one, q);
      pressure.integrate(true, false);
      pressure.distribute_local_to_global(dst);
    }
}



template <int dim>
template <int degree_p>
void
NavierStokesMatrix<dim>::local_pressure_convdiff(
  const MatrixFree<dim> &                           data,
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  typedef VectorizedArray<double>              vector_t;
  FEEvaluation<dim, degree_p, degree_p + 2, 1> pressure(data, dof_index_p, quad_index_u);

  Assert(linearized_velocities.size() > 0, ExcNotImplemented());

  const bool       use_variable_coefficients = variable_viscosities.size() > 0;
  velocity_stored *linearized =
    &linearized_velocities[cell_range.first * pressure.n_q_points];

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      pressure.reinit(cell);
      get_pressure_values(pressure, src);
      pressure.evaluate(false, true, false);

      const vector_t mu_value = use_variable_coefficients ?
                                  begin_viscosities(cell)[pressure.n_q_points / 2] :
                                  make_vectorized_array(parameters.viscosity);

      for (unsigned int q = 0; q < pressure.n_q_points; ++q)
        {
          Tensor<1, dim, vector_t> pres_grad = pressure.get_gradient(q);
          pressure.submit_gradient(pres_grad * mu_value, q);
          // Does not work properly yet...
          // pressure.submit_value(pres_grad*linearized[q].first, q);
        }

      pressure.integrate(false, true);
      distribute_pressure_ltg(pressure, dst);

      linearized += pressure.n_q_points;
    }
}



template <int dim>
void
NavierStokesMatrix<dim>::fix_linearization_point() const
{
  linearized_velocities_preconditioner = linearized_velocities;
  variable_densities_preconditioner    = variable_densities;
  variable_viscosities_preconditioner  = variable_viscosities;
}



template <int dim>
std::size_t
NavierStokesMatrix<dim>::memory_consumption() const
{
  std::size_t memory = linearized_velocities.memory_consumption();
  memory += variable_densities.size();
  memory += variable_viscosities.size();
  memory += linearized_velocities_preconditioner.size();
  memory += variable_densities_preconditioner.size();
  memory += variable_viscosities_preconditioner.size();
  return memory;
}



template <int dim>
void
NavierStokesMatrix<dim>::print_memory_consumption(std::ostream &stream) const
{
  stream << "| Linearized velocities: "
         << 1e-6 * double(linearized_velocities.memory_consumption() +
                          linearized_velocities_preconditioner.memory_consumption())
         << " MB\n";
  if (variable_viscosities.size() > 0)
    stream << "| Variable densities & viscosities: "
           << 1e-6 * double(variable_densities.memory_consumption() +
                            variable_viscosities.memory_consumption() +
                            variable_densities_preconditioner.memory_consumption() +
                            variable_viscosities_preconditioner.memory_consumption())
           << " MB\n";
}



template <int dim>
std::pair<Utilities::MPI::MinMaxAvg, unsigned int>
NavierStokesMatrix<dim>::get_matvec_statistics() const
{
  std::pair<Utilities::MPI::MinMaxAvg, unsigned int> minmax;
  minmax.second = matvec_timer.first;
  minmax.first =
    Utilities::MPI::min_max_avg(matvec_timer.second,
                                solution_old.block(0).get_mpi_communicator());
  matvec_timer.first  = 0;
  matvec_timer.second = 0;
  return minmax;
}


// explicit instantiations
template class NavierStokesMatrix<2>;
template class NavierStokesMatrix<3>;
