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

#ifndef __adaflo_level_set_base_h
#define __adaflo_level_set_base_h

#include <deal.II/fe/fe_values.h>

#include <adaflo/diagonal_preconditioner.h>
#include <adaflo/navier_stokes.h>
#include <adaflo/two_phase_base.h>


namespace adaflo
{
  using namespace dealii;

  template <int dim>
  class LevelSetBaseAlgorithm : public TwoPhaseBaseAlgorithm<dim>
  {
  public:
    LevelSetBaseAlgorithm(const FlowParameters &parameters,
                          Triangulation<dim>   &triangulation);
    virtual ~LevelSetBaseAlgorithm()
    {}

    virtual void
    setup_problem(const Function<dim> &initial_velocity_field,
                  const Function<dim> &initial_distance_function =
                    Functions::ZeroFunction<dim>()) override;
    virtual void
    initialize_data_structures() override;

    virtual std::pair<unsigned int, unsigned int>
    advance_time_step() override; // perform one time step

    // write whole solution to file
    virtual void
    output_solution(const std::string  output_name,
                    const unsigned int n_subdivisions = 0) const override;

    double old_residual;

  protected:
    virtual void
    evaluate_heaviside_function(FEValues<dim>               &fe_values,
                                std::vector<double>         &cell_heaviside,
                                std::vector<Tensor<1, dim>> &cell_delta) const override
    {
      for (unsigned int d = 0; d < dim; ++d)
        {
          // We extract the heaviside and the normal vector. This latter one
          // will be used to compute the area in 3D and perimeter in 2D.

          fe_values.get_function_values(normal_vector_field.block(d), cell_heaviside);
          for (unsigned int q = 0; q < cell_heaviside.size(); ++q)
            cell_delta[q][d] = cell_heaviside[q];
        }
      fe_values.get_function_values(heaviside, cell_heaviside);
    }

    // adaptive grid refinement around the interface (based on values of
    // concentration gradient)
    virtual bool
    mark_cells_for_refinement() override;

    // compute the force term and variable density/viscosity for the
    // Navier--Stokes equations
    virtual void
    compute_force() = 0;

    // compute the density on faces needed for the Navier-Stokes preconditioner
    // with FE_Q_DG0 elements
    void
    compute_density_on_faces();

    // advection step
    virtual void
    advance_concentration() = 0;

    virtual void
    compute_normal(const bool fast_computation) = 0;

    virtual void
    compute_curvature(const bool diffuse_large_values = false) = 0;

    // performs reinitialization
    virtual void
    reinitialize(const unsigned int stab_steps,
                 const unsigned int diff_steps                              = 0,
                 const bool         diffuse_cells_with_large_curvature_only = false) = 0;

    virtual void
    compute_heaviside() = 0;

    LinearAlgebra::distributed::Vector<double> heaviside;

    LinearAlgebra::distributed::BlockVector<double> normal_vector_field;
    LinearAlgebra::distributed::BlockVector<double> normal_vector_rhs;

    FullMatrix<double> interpolation_concentration_pressure;

    unsigned int last_smoothing_step, last_refine_step;
  };



  // this is the integral of the sqrt-formed discrete delta function described
  // in Peskin (The immersed boundary method, Acta Numerica 11:479--517, 2002)
  inline double
  discrete_heaviside(const double x)
  {
    if (x > 0)
      return 1. - discrete_heaviside(-x);
    else if (x < -2.)
      return 0;
    else if (x < -1.)
      {
        return (1. / 8. * (5. * x + x * x) +
                1. / 32. * (-3. - 2. * x) * std::sqrt(-7. - 12. * x - 4. * x * x) -
                1. / 16 * std::asin(std::sqrt(2.) * (x + 3. / 2.)) + 23. / 32. -
                numbers::PI / 64.);
      }
    else
      {
        return (1. / 8. * (3. * x + x * x) -
                1. / 32. * (-1. - 2. * x) * std::sqrt(1. - 4. * x - 4. * x * x) +
                1. / 16 * std::asin(std::sqrt(2.) * (x + 1. / 2.)) + 15. / 32. -
                numbers::PI / 64.);
      }
  }


  // and this is the actual delta function from Peskin
  inline double
  discrete_delta(const double x)
  {
    if (x > 0)
      return discrete_delta(-x);
    else if (x < -2.)
      return 0;
    else if (x < -1.)
      return 1. / 8. * (5. + 2. * x - std::sqrt(-7. - 12. * x - 4. * x * x));
    else
      return 1. / 8. * (3. + 2. * x + std::sqrt(1. - 4. * x - 4. * x * x));
  }


  // this is a cutoff function
  inline double
  cutoff_function(const double x)
  {
    const double beta = 5., gamma = 7.;
    const double abx = std::fabs(x);
    if (abx < beta)
      return 1;
    else if (abx < gamma)
      return (abx - gamma) * (abx - gamma) * (2 * abx + gamma - 3 * beta) /
             ((gamma - beta) * (gamma - beta) * (gamma - beta));
    else
      return 0;
  }
} // namespace adaflo

#endif
