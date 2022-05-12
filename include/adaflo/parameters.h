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

#ifndef __adaflo_parameters_h
#define __adaflo_parameters_h

#include <deal.II/base/parameter_handler.h>

#include <adaflo/time_stepping.h>

#include <fstream>
#include <iostream>

using namespace dealii;

struct FlowParameters
{
  // allows to defer construction to a derived class. Handle with care, as
  // parameters are not default-initialized
  FlowParameters();

  FlowParameters(const std::string &parameter_filename);

  static void
  declare_parameters(ParameterHandler &prm);

  void
  parse_parameters(const std::string parameter_filename, ParameterHandler &prm);

  void
  parse_parameters(ParameterHandler &prm);

  void
  check_for_file(const std::string &parameter_filename, ParameterHandler &prm) const;

  unsigned int dimension;
  unsigned int global_refinements;
  unsigned int adaptive_refinements;
  bool         use_anisotropic_refinement;
  bool         use_simplex_mesh;

  enum PhysicalType
  {
    incompressible,
    incompressible_stationary,
    stokes
  } physical_type;

  enum ConstitutiveType
  {
    newtonian_compressible_stokes_hypothesis,
    newtonian_incompressible,
    user_defined
  } constitutive_type;

  std::map<std::string, double> get_beta_formulation_convective_term_momentum_balance = {
    {"conservative", 1.0},
    {"convective", 0.0},
    {"skew-symmetric", 0.5},
  };
  double       beta_convective_term_momentum_balance;
  unsigned int velocity_degree;
  bool         augmented_taylor_hood;
  double       viscosity;
  double       density;
  double       damping;
  double       tau_grad_div;
  unsigned int max_nl_iteration;
  double       tol_nl_iteration;
  enum Linearization
  {
    coupled_implicit_newton,
    coupled_implicit_picard,
    coupled_velocity_semi_implicit,
    coupled_velocity_explicit,
    projection
  } linearization;

  unsigned int max_lin_iteration;
  double       tol_lin_iteration;
  bool         rel_lin_iteration;
  enum PreconditionVelocity
  {
    u_ilu,
    u_ilu_scalar,
    u_amg_linear,
    u_amg
  } precondition_velocity;
  enum PreconditionPressure
  {
    p_mass_diag,
    p_mass_ilu
  } precondition_pressure;
  unsigned int iterations_before_inner_solvers;

  std::string  output_filename;
  unsigned int output_verbosity;
  double       output_frequency;
  unsigned int print_solution_fields;
  bool         output_wall_times;

  TimeSteppingParameters::Scheme time_step_scheme;
  double                         start_time;
  double                         end_time;
  double                         time_step_size_start;
  double                         time_stepping_cfl;
  double                         time_stepping_coef2;
  double                         time_step_tolerance;
  double                         time_step_size_max;
  double                         time_step_size_min;

  // Two-phase specific parameters
  double density_diff;
  double viscosity_diff;

  double surface_tension;
  double gravity;
  double epsilon;
  double diffusion_length; // only useful in Cahn-Hilliard
  double contact_angle;    // only useful in Cahn-Hilliard

  bool pressure_constraint;

  unsigned int concentration_subdivisions;
  unsigned int curvature_correction;           // only for level set
  bool         interpolate_grad_onto_pressure; // only for level set
  bool         surface_tension_from_heaviside; // only for level set
  bool         approximate_projections;        // only for level set
  bool         ch_do_newton;                   // only for Cahn-Hilliard
  bool         do_iteration;                   // only for Cahn-Hilliard
  unsigned int n_reinit_steps;                 // only for level set
  unsigned int n_initial_reinit_steps;         // only for level set
  bool         convection_stabilization;
};



#endif
