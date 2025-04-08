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

#ifndef __adaflo_two_phase_base_h_
#define __adaflo_two_phase_base_h_

#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/data_out.h>

#include <adaflo/flow_base_algorithm.h>
#include <adaflo/navier_stokes.h>
#include <adaflo/parameters.h>

#include <fstream>
#include <iostream>


namespace adaflo
{
  using namespace dealii;

  template <int dim>
  class TwoPhaseBaseAlgorithm : public FlowBaseAlgorithm<dim>
  {
  public:
    TwoPhaseBaseAlgorithm(const FlowParameters &                    parameters,
                          const std::shared_ptr<FiniteElement<dim>> fe,
                          Triangulation<dim> &                      triangulation,
                          TimerOutput *                             external_timer = 0);

    virtual ~TwoPhaseBaseAlgorithm();

    virtual void
    clear_data();

    virtual void
    setup_problem(const Function<dim> &initial_velocity_field,
                  const Function<dim> &initial_distance_function =
                    Functions::ZeroFunction<dim>()) override;

    virtual void
    distribute_dofs();
    virtual void
    initialize_data_structures();

    virtual void
    init_time_advance();

    // Adaptive grid refinement strategy based on concentration gradient. Can be
    // overriden by derived class
    virtual bool
    mark_cells_for_refinement();
    virtual void
    refine_grid();

    // Computes some bubble statistics such as the bubble diameter and perimeter
    // and the velocity of the bubble. In 2D, the zero contour of the interface
    // is computed explicitly from the level set function. The optional
    // parameter sub_refinements allows to specify how often to refine each
    // element for sub-element resolution; if set to -1, an automatic value will
    // be used. The optional argument @p interface_points collects the interface
    // position in a list of points, connecting the start and end point of a
    // segment within an element or part of an element.
    std::vector<double>
    compute_bubble_statistics(
      std::vector<Tensor<2, dim>> *interface_points = 0,
      const unsigned int           sub_refinements = numbers::invalid_unsigned_int) const;

    // Computes some bubble statistics such as the bubble diameter and perimeter
    // and the velocity of the bubble using immersed/cut functionality. In 2D,
    // this is relatively similar to compute_bubble_statistics. In 3D, this call
    // is considerably more accurate and detailed than the plain bubble
    // statistics which compute smeared-out quantities.
    std::vector<double>
    compute_bubble_statistics_immersed(
      std::vector<Tensor<2, dim>> *interface_points = 0) const;

    void
    set_adaptive_time_step(const double velocity_norm) const;

    double
    get_maximal_velocity() const;
    std::pair<double, double>
    get_concentration_range() const;

    // Prints the solution fields to a vtu file. For the output file name, the
    // base name of the file must be given, e.g. "sol". The routine will then
    // append a time indicator in terms of the currently active time step as
    // well as the processor number. The final file name then looks e.g. like
    // "sol-0001-0000.vtu" for time step 1 on processor 0. The routine also
    // writes a master record for pvtu and visit files to the same file.
    //
    // The optional argument @p n_subdivisions lets the user override the
    // default value (0, taking the minimum of velocity degree and possible
    // degree in level set) the sub-refinement used for representing higher
    // order solutions.
    virtual void
    output_solution(const std::string  output_base_name,
                    const unsigned int n_subdivisions = 0) const override;

    // A derived class can transform the initial signed distance function
    // prescribed by the input distance function in setup_problem, e.g. to have
    // a tanh shape like for the LevelSetOKZ or phase field solver
    virtual void
    transform_distance_function(LinearAlgebra::distributed::Vector<double> &) const
    {}

    const DoFHandler<dim> &
    get_dof_handler() const
    {
      return dof_handler;
    }

    const AffineConstraints<double> &
    get_constraints_concentration() const
    {
      return constraints;
    }

    AffineConstraints<double> &
    modify_constraints_concentration()
    {
      return constraints;
    }

    const AffineConstraints<double> &
    get_constraints_curvature() const
    {
      return constraints_curvature;
    }

    AffineConstraints<double> &
    modify_constraints_curvature()
    {
      return constraints_curvature;
    }

    const AffineConstraints<double> &
    get_constraints_normals() const
    {
      return constraints_normals;
    }

    AffineConstraints<double> &
    modify_constraints_normals()
    {
      return constraints_normals;
    }

    const TimeStepping &
    get_time_stepping() const
    {
      return navier_stokes.time_stepping;
    }

    const NavierStokes<dim> &
    get_navier_stokes() const
    {
      return navier_stokes;
    }

    NavierStokes<dim> &
    get_navier_stokes()
    {
      return navier_stokes;
    }

    LinearAlgebra::distributed::BlockVector<double> solution_update;
    LinearAlgebra::distributed::BlockVector<double> solution, solution_old,
      solution_old_old;

  protected:
    virtual void
    evaluate_heaviside_function(FEValues<dim> &,
                                std::vector<double> &,
                                std::vector<Tensor<1, dim>> &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    print_n_dofs() const;

    ConditionalOStream           pcout;
    std::shared_ptr<TimerOutput> timer;

    // Reference to externally defined triangulation
    Triangulation<dim> &                      triangulation;
    NavierStokes<dim>                         navier_stokes;
    MatrixFree<dim>                           matrix_free;
    const std::shared_ptr<FiniteElement<dim>> fe;

    DoFHandler<dim> dof_handler;

    // We use two sets of constraints for the concentration/level set variable
    // and a 'curvature' field (in phase field, this is the chemical potential),
    // but both use the same DoFHandler
    AffineConstraints<double> hanging_node_constraints;
    AffineConstraints<double> constraints;
    AffineConstraints<double> constraints_curvature;
    AffineConstraints<double> constraints_normals;

    LinearAlgebra::distributed::BlockVector<double> system_rhs;

    TimeStepping &        time_stepping;
    const FlowParameters &parameters;

    double                                 epsilon_used;
    double                                 minimal_edge_length;
    double                                 global_omega_diameter;
    mutable std::pair<double, double>      last_concentration_range;
    int                                    refine_lower_level_limit;
    AlignedVector<VectorizedArray<double>> cell_diameters;

    const Quadrature<dim> face_center_quadrature;

    std::string curvature_name;
  };
} // namespace adaflo

#endif
