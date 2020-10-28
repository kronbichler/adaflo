// --------------------------------------------------------------------------
//
// Copyright (C) 2013 - 2016 by the adaflo authors
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

#ifndef __adaflo_flow_base_algorithm_h
#define __adaflo_flow_base_algorithm_h

#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/data_out.h>
#include <adaflo/time_stepping.h>

#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <iomanip>


using namespace dealii;



namespace helpers
{
  /**
   * Internal structure that keeps all information about boundary
   * conditions. Necessary to enable different classes to share the boundary
   * conditions.
   */
  template <int dim>
  struct BoundaryDescriptor
  {
    BoundaryDescriptor();

    std::map<types::boundary_id,std::shared_ptr<Function<dim> > > dirichlet_conditions_u;
    std::map<types::boundary_id,std::shared_ptr<Function<dim> > > open_conditions_p;
    std::map<types::boundary_id,std::shared_ptr<Function<dim> > > pressure_fix;

    std::set<types::boundary_id> normal_flux;
    std::set<types::boundary_id> symmetry;
    std::set<types::boundary_id> no_slip;

    std::set<types::boundary_id> fluid_type_plus;
    std::set<types::boundary_id> fluid_type_minus;

    std::pair<types::boundary_id,types::boundary_id> periodic_boundaries[dim];
  };
}


/**
 * Base class for fluid flow problems (both single-phase and two-phase
 * problems) that implements boundary conditions and the basic steps of the
 * solution, setup, advance_one_time_step, and output). The actual action of
 * most of these is defined in derived classes.
 *
 * Boundary conditions are set at the beginning of the setup() function of the
 * individual fluid solvers using the stored internal boundary information. In
 * order for the program to recognize the conditions, they must be set before
 * the setup() call. The best way to impose the conditions is to do it as soon
 * as the triangulation is created. There is no need to re-impose boundary
 * conditions when the mesh is refined as boundary indicators do not change
 * (but you need to reset them when the boundary indicator changes, of
 * course).
 *
 * All of the set functions below are only allowed to be called
 * once. Otherwise, the conditions need to be cleared with this
 * clear_all_boundary_conditions() function.
 *
 * This class is a base class for all fluid classes and includes the standard
 * set of boundary conditions. It allows derived classes to use the methods
 * here as given or to override them when additional information is necessary,
 * e.g. in phase field methods where one might want to set the concentration
 * order function to a given value.
 */
template <int dim>
struct FlowBaseAlgorithm
{
  /**
   * Constructor.
   */
  FlowBaseAlgorithm();

  /**
   * Destructor.
   */
  virtual ~FlowBaseAlgorithm();

  /**
   * Setup of problem. Initializes the degrees of freedom and solver-related
   * variables (vectors, matrices, etc.)
   */
  virtual void setup_problem (const Function<dim> &initial_velocity_field,
                              const Function<dim> &initial_distance_function = Functions::ZeroFunction<dim>()) = 0;

  /**
   * Performs one complete time step of the problem, including the solution of
   * each associated field. Returns the number of accumulated linear
   * iterations during the time step.
   */
  virtual unsigned int advance_time_step () = 0;

  /**
   * Generic output interface. Allows to write the complete solution field to
   * a vtu file. Derived classes decide which variables need to be written and
   * how often this is about to happen, typically governed by the output
   * option 'output frequency'.
   *
   * The optional argument @p n_subdivisions lets the user override the
   * default value (0, taking the minimum of velocity degree and possible
   * degree in level set) the sub-refinement used for representing higher
   * order solutions.
   */
  virtual void output_solution (const std::string output_base_name,
                                const unsigned int n_subdivisions = 0) const = 0;

  /**
   * Deletes all stored boundary descriptions.
   */
  void clear_all_boundary_conditions();

  /*
   * Sets a Dirichlet condition for the fluid velocity on the boundary of the
   * domain with the given id. Note that you can specify time-dependent
   * functions. Remember to use the built-in "get_time()" within the function
   * to access the time and not define your own time variable, otherwise the
   * imposed conditions will not be correct.
   *
   * The optional argument fluid_type allows to set which fluid is flowing in
   * at the boundary: If it is set to zero (or a single-fluid problem is
   * solved), no explicit Dirichlet condition on the fluid indicator is
   * set. Otherwise, a value of +1 sets the fluid to the +1 fluid type,
   * whereas a value of -1 sets it to the other type. Other values are
   * currently not allowed.
   *
   * Prerequisite: The given function must consist of dim components.
   */
  void set_velocity_dirichlet_boundary (const types::boundary_id  boundary_id,
                                        const std::shared_ptr<Function<dim> > &velocity_function,
                                        const int inflow_fluid_type = 0);

  /*
   * Sets a pressure condition on the boundary of the domain with the given
   * id. Note that you can specify time-dependent functions. It is only
   * important that you use the built-in "get_time()" within the function to
   * access the time and not define your own time variable.
   *
   * You can only set pressure boundary conditions on boundaries where there
   * is no velocity Dirichlet conditions.
   *
   * The optional argument fluid_type allows to set which fluid is flowing in
   * at the boundary: If it is set to zero (or a single-fluid problem is
   * solved), no explicit Dirichlet condition on the fluid indicator is
   * set. Otherwise, a value of +1 sets the fluid to the +1 fluid type,
   * whereas a value of -1 sets it to the other type. Other values are
   * currently not allowed.
   *
   * Prerequisite: The given function(s) must be scalar.
   */
  void set_open_boundary (const types::boundary_id  boundary_id,
                          const std::shared_ptr<Function<dim> > &pressure_function
                          = std::shared_ptr<Function<dim> >(),
                          const int inflow_fluid_type = 0);

  /*
   * Sets a pressure condition on the boundary of the domain with the given
   * id. It forces the flow field to be normal to the boundary, i.e., the
   * tangential component of the flow will be constrained to zero. Note that
   * you can specify time-dependent functions. It is only important that you
   * use the built-in "get_time()" within the function to access the time and
   * not define your own time variable.
   *
   * You can only set this boundary condition on boundaries where no velocity
   * boundary condition and no other "open" boundary condition is set.
   *
   * The optional argument fluid_type allows to set which fluid is flowing in
   * at the boundary: If it is set to zero (or a single-fluid problem is
   * solved), no explicit Dirichlet condition on the fluid indicator is
   * set. Otherwise, a value of +1 sets the fluid to the +1 fluid type,
   * whereas a value of -1 sets it to the other type. Other values are
   * currently not allowed.
   *
   * Prerequisite: The given function(s) must be scalar.
   */
  void set_open_boundary_with_normal_flux (const types::boundary_id  boundary_id,
                                           const std::shared_ptr<Function<dim> > &pressure_function
                                           = std::shared_ptr<Function<dim> >(),
                                           const int inflow_fluid_type = 0);

  /*
   * Fix one boundary node to a value specified by the given function,
   * evaluating it on the smallest index of pressure on the given boundary
   * id. For specifying a zero function, you can skip the second argument.
   *
   * You can only set this conditions when Dirichlet boundary conditions are
   * set on the whole boundary.
   *
   * Prerequisite: The given function(s) must be scalar.
   */
  void fix_pressure_constant (const types::boundary_id  boundary_id,
                              const std::shared_ptr<Function<dim> > &pressure_function
                              = std::shared_ptr<Function<dim> >());

  /*
   * Sets symmetry boundary conditions on the given boundaries. A symmetry
   * condition sets the normal velocity on the boundary to zero but allows
   * tangential velocities. Symmetry boundary conditions can be set on both
   * straight boundaries and curved boundaries.
   */
  void set_symmetry_boundary (const types::boundary_id boundary_id);

  /*
   * Sets no-slip boundary conditions on the given side. This function sets
   * the velocity to zero along the boundary that corresponds to the given
   * boundary indicator.
   */
  void set_no_slip_boundary (const types::boundary_id boundary_id);

  /**
   * Sets a direction of the flow to be periodic in the given coordinate
   * direction. A prerequisite for this functionality is that the boundary id
   * of the incoming direction and the boundary id of the outgoing direction
   * have a different boundary indicator (that is not allowed to be used for
   * any other type of boundary condition).
   *
   * For using this function on a parallel distributed triangulation, you need
   * to perform the following steps on the triangulation:
   *
   * @code
   * std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
   *     periodic_faces;
   * GridTools::collect_periodic_faces(triangulation, incoming_boundary_id,
   *                                   outgoing_boundary_id, direction, periodic_faces);
   * // possibly other directions you might want to be periodic
   * triangulation.add_periodicity(periodic_faces);
   * @endcode
   */
  void set_periodic_direction (const unsigned int direction,
                               const types::boundary_id incoming_boundary_id,
                               const types::boundary_id outgoing_boundary_id);

  /*
   * Writes the data output, assuming that the user already has set up a
   * DataOut object and called build_patches on it. Mostly used internally,
   * but can be useful to user programs in case one wants to output additional
   * vectors that are not done by the standard solvers.
   */
  void write_data_output (const std::string  &output_base_name,
                          const TimeStepping &time_stepping,
                          const double        output_frequency,
                          DataOut<dim>       &data_out) const;

  std::shared_ptr<helpers::BoundaryDescriptor<dim> > boundary;

  MappingQ<dim> mapping;
};


#endif
