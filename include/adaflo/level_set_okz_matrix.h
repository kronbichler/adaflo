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

#ifndef __adaflo_level_set_okz_matrix_h
#define __adaflo_level_set_okz_matrix_h

#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <adaflo/level_set_base.h>


namespace adaflo
{
  using namespace dealii;


  /**
   * A matrix-based level set solver based on the conservative level set method
   * by Olsson, Kreiss and Zahedi.
   */
  template <int dim>
  class LevelSetOKZMatrixSolver : public LevelSetBaseAlgorithm<dim>
  {
  public:
    LevelSetOKZMatrixSolver(const FlowParameters &parameters,
                            Triangulation<dim> &  triangulation);
    virtual ~LevelSetOKZMatrixSolver()
    {}

    virtual void
    initialize_data_structures() override;

    virtual void
    transform_distance_function(
      LinearAlgebra::distributed::Vector<double> &vector) const override;

  private:
    // compute the force term and variable density/viscosity for the
    // Navier--Stokes equations
    virtual void
    compute_force() override;

    // advection step
    virtual void
    advance_concentration() override;

    // computes normal direction vector by projection of level set gradient (not
    // scaled to have norm 1!)
    virtual void
    compute_normal(const bool fast_computation) override;

    // computes curvature by projecting the divergence of the normal vector
    // (scaled to norm 1 now)
    virtual void
    compute_curvature(const bool diffuse_large_values = false) override;

    // performs reinitialization
    virtual void
    reinitialize(const unsigned int stab_steps,
                 const unsigned int diff_steps                      = 0,
                 const bool diffuse_cells_with_large_curvature_only = false) override;

    virtual void
    compute_heaviside() override;

    bool   normal_calculated;
    double global_max_velocity;

    TrilinosWrappers::SparseMatrix system_matrix;
  };
} // namespace adaflo

#endif
