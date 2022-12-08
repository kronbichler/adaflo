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

#include <deal.II/base/mpi.h>

#include <adaflo/parameters.h>


FlowParameters::FlowParameters()
  : dimension(numbers::invalid_unsigned_int)
{
  // do nothing
}



FlowParameters::FlowParameters(const std::string &parameter_filename)
{
  ParameterHandler prm;
  FlowParameters::declare_parameters(prm);
  check_for_file(parameter_filename, prm);
  parse_parameters(parameter_filename, prm);
}



void
FlowParameters::check_for_file(const std::string &parameter_filename,
                               ParameterHandler & /*prm*/) const
{
  std::ifstream parameter_file(parameter_filename.c_str());

  if (!parameter_file)
    {
      parameter_file.close();

      std::ostringstream message;
      message << "Input parameter file <" << parameter_filename
              << "> not found. Please make sure the file exists!" << std::endl;

      AssertThrow(false, ExcMessage(message.str().c_str()));
    }
}


void
FlowParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Navier-Stokes");
  prm.declare_entry("dimension",
                    "2",
                    Patterns::Integer(),
                    "Defines the dimension of the problem. Not essential "
                    "to the Navier-Stokes class, but useful in many "
                    "applications.");
  prm.declare_entry("global refinements",
                    "1",
                    Patterns::Integer(),
                    "Defines the number of initial global refinements. "
                    "Not used in the Navier-Stokes class, but useful in "
                    "many applications.");
  prm.declare_entry("anisotropic refinement",
                    "0",
                    Patterns::Integer(),
                    "defines whether the mesh should be refined "
                    "anisotropically in normal direction to the interface, "
                    "0 means no anisotropy");
  prm.declare_entry("simplex mesh",
                    "0",
                    Patterns::Integer(),
                    "defines whether a simplex mesh has been provided, "
                    "0 means mesh with only quadrilaterals (2D) and hexahedra "
                    "(3D) has been provided");
  prm.declare_entry("adaptive refinements",
                    "0",
                    Patterns::Integer(),
                    "Defines the number of adaptive refinements. Not used "
                    "in the Navier-Stokes class, but useful in many "
                    "applications.");
  prm.declare_entry("velocity degree",
                    "2",
                    Patterns::Integer(),
                    "Sets the degree for velocity. Pressure degree is "
                    "velocity degree minus one. Currently implemented for "
                    "orders 2 to 6");
  prm.declare_entry("augmented Taylor-Hood elements",
                    "0",
                    Patterns::Integer(),
                    "Option to choose the pressure space FE_Q_DG0(p_degree) "
                    "instead of the standard space FE_Q(p_degree). This "
                    "adds a constant discontinuous part to the pressure "
                    "basis and gives element-wise divergence-free solutions. "
                    "It produces solutions that are in general better but "
                    "also a bit more expensive to compute.");
  prm.declare_entry("viscosity",
                    "1.",
                    Patterns::Double(),
                    "Defines the fluid dynamic viscosity");
  prm.declare_entry("density", "1.", Patterns::Double(), "Defines the fluid density");
  prm.declare_entry("damping", "0", Patterns::Double(), "Defines the fluid damping.");
  prm.declare_entry("physical type",
                    "incompressible",
                    Patterns::Selection(
                      "incompressible|incompressible stationary|stokes"),
                    "Sets the type of equations, Navier-Stokes or Stokes. For "
                    "Navier-Stokes, one can choose between a stationary and a "
                    "time-dependent variant. The time-dependent Navier-Stokes "
                    "equations are the default.");
  prm.declare_entry("formulation convective term momentum balance",
                    "skew-symmetric",
                    Patterns::Selection("skew-symmetric|convective|conservative"),
                    "Sets the formulation of the convective term in the "
                    "momentum balance of the Navier-Stokes equations, i.e. "
                    "∇·(u x u) =(u·∇)u + βu(∇·u). The parameter β will be "
                    "set to 1 for the conservative form, to 0 for the "
                    "convective form and to 0.5 for the skew-symmetric "
                    "form (default formulation).");

  prm.enter_subsection("Solver");
  prm.declare_entry("NL max iterations",
                    "10",
                    Patterns::Integer(),
                    "Defines the maximum number of nonlinear Newton "
                    "iterations.");
  prm.declare_entry("NL tolerance",
                    "1e-6",
                    Patterns::Double(),
                    "Defines the tolerance in the residual l2 norm in "
                    "the nonlinear Newton iteration.");
  prm.declare_entry(
    "linearization scheme",
    "coupled implicit Newton",
    Patterns::Selection(
      "coupled implicit Newton|coupled implicit Picard|coupled velocity semi-implicit|coupled velocity explicit|projection"),
    "Sets how to treat the coupled nonlinear Navier-Stokes "
    "system. The 'coupled' variants solve for the full block "
    "system, whereas 'projection' applies a fractional-step "
    "pressure correction method with the solution of a pressure "
    "Poisson matrix. "
    "The nonlinear convective term can be treated by a"
    "full Newton iteration, a Picard iteration (fixed-point "
    "like), a semi-implicit approach with the same term as in "
    "the fixed-point like iteration but velocity extrapolated "
    "from the old time, and an approach where the complete "
    "convective term is treated explicitly. "
    "For the projection scheme, only the semi-implicit "
    "velocity treatment is implemented because iterating "
    "out the nonlinearity makes no sense.");
  prm.declare_entry("tau grad div",
                    "0.",
                    Patterns::Double(),
                    "Adds the term (div(v), tau div(u))"
                    "to the weak form the momentum equation, which is "
                    "consistent with the Navier-Stokes equations but "
                    "penalizes the divergence more. This term is usually "
                    "referred to as grad-div stabilization. It simplifies "
                    "the solution of linear systems if tau is on the order "
                    "of unity but not too large (as the added term is "
                    "singular).");

  prm.declare_entry("lin max iterations",
                    "500",
                    Patterns::Integer(),
                    "Maximum number of linear iterations");
  prm.declare_entry("lin tolerance",
                    "1.e-3",
                    Patterns::Double(),
                    "Tolerance for the linear solver");
  prm.declare_entry("lin relative tolerance",
                    "1",
                    Patterns::Integer(),
                    "Sets whether the residual for the linear solver "
                    "should be measured relative to the nonlinear residual "
                    "(recommended option).");
  prm.declare_entry("lin velocity preconditioner",
                    "amg linear",
                    Patterns::Selection("ilu|ilu scalar|amg linear|amg"),
                    "Sets the preconditioner for approximating the inverse "
                    "of the velocity matrix in the Schur complement "
                    "preconditioner. 'amg linear' uses a matrix based on "
                    "subdividing FE_Q into several linear elements to "
                    "create a matrix hierarchy. This might decrease "
                    "interpolation quality, but AMG is typically much better "
                    "for linears, so it is recommended for more complex "
                    "problems with relatively large time steps or large "
                    "viscosities, otherwise ILU. The method 'ilu scalar' "
                    "is a simplified ILU that only constructs the ILU for "
                    "one velocity block and applies the same operator to "
                    "all components. It is cheaper to apply but approximates "
                    "somewhat worse.");
  prm.declare_entry("lin pressure mass preconditioner",
                    "ilu",
                    Patterns::Selection("ilu|diagonal"),
                    "Sets whether the pressure mass matrix in the Schur "
                    "complement should be represented by the diagonal only "
                    "or by an ILU based on the full pressure mass matrix.");
  prm.declare_entry("lin its before inner solvers",
                    "50",
                    Patterns::Integer(),
                    "The linear solver comes in two flavors. A simple "
                    "solver which uses only AMG V-cycles or ILUs as "
                    "preconditioner components in the Schur complement, "
                    "or a stronger solver with inner iterations. The "
                    "variant with inner solves is less efficient when "
                    "only a few iterations are needed, but much more "
                    "robust and more efficient for many iterations. This "
                    "option sets how many linear iterations with the cheap "
                    "preconditioners should be made before the stronger "
                    "version with more iterations starts.");
  prm.leave_subsection();
  prm.leave_subsection();

  prm.enter_subsection("Output options");
  prm.declare_entry("output filename",
                    "",
                    Patterns::Anything(),
                    "Sets the base name for the file output.");
  prm.declare_entry("output verbosity",
                    "2",
                    Patterns::Integer(),
                    "Sets the amount of information from the "
                    "Navier-Stokes solver that is printed to screen. "
                    "0 means no output at all, and larger numbers mean an "
                    "increasing amount of output (maximum value: 3). "
                    "A value of 3 not only includes solver iterations "
                    "but also details on solution time and some memory "
                    "statistics.");
  prm.declare_entry("output frequency",
                    "1",
                    Patterns::Double(),
                    "defines at with time interface the solution "
                    "should be written to file (in supported routines)");
  prm.declare_entry("output vtk files",
                    "0",
                    Patterns::Integer(),
                    "defines whether to output vtk files with the "
                    "whole solution field or just collected point data");
  prm.declare_entry("output wall times",
                    "0",
                    Patterns::Integer(),
                    "Defines whether to output wall times. 0 means no output.");
  prm.declare_entry("output memory",
                    "0",
                    Patterns::Integer(),
                    "Defines whether to output memory. 0 means no output.");
  prm.leave_subsection();

  prm.enter_subsection("Two phase");
  prm.declare_entry("density",
                    "-1.",
                    Patterns::Double(),
                    "Density of fluid 1 (negative region of level set function). "
                    "If given a positive value, overwrites density in "
                    "Navier-Stokes subsection.");
  prm.declare_entry("density difference",
                    "0.",
                    Patterns::Double(),
                    "absolute difference in density compared to fluid 1");
  prm.declare_entry("viscosity",
                    "-1.",
                    Patterns::Double(),
                    "Dynamic viscosity of fluid 1 (negative region of level "
                    "set function). If given a positive value, overwrites "
                    "density in Navier-Stokes subsection.");
  prm.declare_entry("viscosity difference",
                    "0.",
                    Patterns::Double(),
                    "absolute difference in viscosity compared to fluid 1");

  prm.declare_entry("surface tension",
                    "1.",
                    Patterns::Double(),
                    "surface tension coefficient");
  prm.declare_entry("epsilon",
                    "1",
                    Patterns::Double(),
                    "Width of diffuse interface, relative to mesh size "
                    "for Level-Set method, but absolute for Cahn-Hilliard.");
  prm.declare_entry("gravity", "0", Patterns::Double(), "");
  prm.declare_entry("diffusion length",
                    "0.1",
                    Patterns::Double(),
                    "Diffusion length scale in Cahn-Hilliard. Its square "
                    "equals the mobility and inverse Peclet number.");
  prm.declare_entry("contact angle",
                    "0",
                    Patterns::Double(),
                    "defines the contact angle at solid interfaces, "
                    "at boundaries with indicator 0 or 2");
  prm.declare_entry("pressure constraint",
                    "1",
                    Patterns::Integer(),
                    "Fixes value of pressure in one point to zero");
  prm.declare_entry("concentration subdivisions",
                    "2",
                    Patterns::Integer(),
                    "Number of subdivision of Q1 elements in smaller elements "
                    "to generate higher accuracy in level set/phase field");
  prm.declare_entry("curvature correction",
                    "0",
                    Patterns::Integer(),
                    "if 1, extend the curvature to the value "
                    "at the interface in normal direction");
  prm.declare_entry("grad pressure compatible",
                    "0",
                    Patterns::Integer(),
                    "if 1, the gradient in the surface tension force "
                    "is interpolated from the pressure gradient");
  prm.declare_entry("localize surface tension",
                    "1",
                    Patterns::Integer(),
                    "if 1, the surface tension is computed from a gradient "
                    "that is localized around the interface (from a "
                    "reconstructed distance function), otherwise it is "
                    "computed from the tanh profile (i.e., nonzero "
                    "everywhere)");
  prm.declare_entry("approximate projections",
                    "0",
                    Patterns::Integer(),
                    "if 0, the normal and curvature in the level set method "
                    "are computed by proper projection (full mass matrix "
                    "and little diffusion), otherwise with diagonal mass "
                    "matrix and time-dependent diffusion");
  prm.declare_entry("Cahn-Hilliard do Newton",
                    "1",
                    Patterns::Integer(),
                    "Sets whether a Newton iteration should be done on the "
                    "Cahn-Hilliard equation (if on that model). If 0 is "
                    "selected, use a convexity splitting as proposed by "
                    "Eyre.");
  prm.declare_entry("full nonlinear iteration",
                    "0",
                    Patterns::Integer(),
                    "iterates between Navier-Stokes and concentration "
                    "if enabled");
  prm.declare_entry("number reinit steps",
                    "2",
                    Patterns::Integer(),
                    "number of iterations in reinitialization");
  prm.declare_entry("number initial reinit steps",
                    "0",
                    Patterns::Integer(),
                    "reinitialization steps before starting the time "
                    "loop (for bad initial profiles)");
  prm.declare_entry("convection stabilization",
                    "0",
                    Patterns::Integer(),
                    "add stabilization terms to advection equation if "
                    "set to 1 (typically not necessary)");
  prm.leave_subsection();


  prm.enter_subsection("Time stepping");
  prm.declare_entry("start time",
                    "0.",
                    Patterns::Double(),
                    "Sets the start time for the simulation");
  prm.declare_entry("end time",
                    "1.",
                    Patterns::Double(),
                    "Sets the final time for the simulation");
  prm.declare_entry("step size",
                    "1.e-2",
                    Patterns::Double(),
                    "Sets the step size for time stepping. For non-uniform "
                    "time stepping, this sets the size of the first time "
                    "step.");
  prm.declare_entry("CFL number",
                    "0.8",
                    Patterns::Double(),
                    "Limits the time step size in terms of a condition "
                    "dt <= CFL * dx / |u|, where u is a characteristic velocity. "
                    "For two-phase flow, we typically take the velocity of "
                    "the bubble");
  prm.declare_entry("CFL number capillary",
                    "10",
                    Patterns::Double(),
                    "Limits the time step size in terms of a condition "
                    "dt <= CFL_cap * sqrt(rho/sigma) * dx^1.5, i.e., it "
                    "represents a capillarity time step limit.");
  prm.declare_entry("tolerance",
                    "1.e-2",
                    Patterns::Double(),
                    "Sets the tolerance for time step selection in "
                    "non-uniform time stepping strategies.");
  prm.declare_entry("max step size",
                    "1.",
                    Patterns::Double(),
                    "Defines the maximum time step size in non-uniform "
                    "strategies.");
  prm.declare_entry("min step size",
                    ".1",
                    Patterns::Double(),
                    "Defines the minimum time step size in non-uniform "
                    "strategies.");
  prm.declare_entry("scheme",
                    "bdf_2",
                    Patterns::Selection("explicit_euler|implicit_euler|"
                                        "crank_nicolson|bdf_2"),
                    "Sets the time stepping scheme. Allowed options are "
                    "explicit_euler, implicit_euler, crank_nicolson "
                    "fractional0, fractional1, new_variant, and bdf_2.");
  prm.leave_subsection();
}



void
FlowParameters::parse_parameters(const std::string parameter_file, ParameterHandler &prm)
{
  try
    {
      if (parameter_file.substr(parameter_file.find_last_of(".") + 1) == "json")
        {
          std::ifstream file;
          file.open(parameter_file);
          prm.parse_input_from_json(file, true);
        }
      else if (parameter_file.substr(parameter_file.find_last_of(".") + 1) == "prm")
        prm.parse_input(parameter_file);
      else
        AssertThrow(false, ExcMessage("Parameterhandler cannot handle current file"));
    }
  catch (...)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        prm.print_parameters(std::cout, ParameterHandler::Text);
      AssertThrow(false, ExcMessage("Invalid input parameter file."));
    }

  this->parse_parameters(prm);
}

void
FlowParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Navier-Stokes");

  dimension                  = prm.get_integer("dimension");
  global_refinements         = prm.get_integer("global refinements");
  adaptive_refinements       = prm.get_integer("adaptive refinements");
  use_anisotropic_refinement = prm.get_integer("anisotropic refinement") > 0;
  use_simplex_mesh           = prm.get_integer("simplex mesh") > 0;

  velocity_degree = prm.get_integer("velocity degree");
  AssertThrow(velocity_degree > 1, ExcNotImplemented());
  augmented_taylor_hood = prm.get_integer("augmented Taylor-Hood elements");
  viscosity             = prm.get_double("viscosity");
  density               = prm.get_double("density");
  damping               = -prm.get_double(
    "damping"); // sign(damping) = minus: damping; sign(damping) = plus: acceleration
  std::string type = prm.get("physical type");
  if (type == "stokes")
    physical_type = stokes;
  else if (type == "incompressible")
    physical_type = incompressible;
  else if (type == "incompressible stationary")
    physical_type = incompressible_stationary;
  else
    Assert(false, ExcNotImplemented());
  if (physical_type == stokes)
    density = 0;
  beta_convective_term_momentum_balance =
    get_beta_formulation_convective_term_momentum_balance[prm.get(
      "formulation convective term momentum balance")];

  prm.enter_subsection("Solver");
  max_nl_iteration   = prm.get_integer("NL max iterations");
  tol_nl_iteration   = prm.get_double("NL tolerance");
  std::string scheme = prm.get("linearization scheme");
  if (scheme == "coupled implicit Newton")
    linearization = coupled_implicit_newton;
  else if (scheme == "coupled implicit Picard")
    linearization = coupled_implicit_picard;
  else if (scheme == "coupled velocity semi-implicit")
    linearization = coupled_velocity_semi_implicit;
  else if (scheme == "coupled velocity explicit")
    linearization = coupled_velocity_explicit;
  else if (scheme == "projection")
    linearization = projection;
  else
    Assert(false, ExcMessage(("Linearization " + scheme + " not available").c_str()));

  if (physical_type == incompressible_stationary)
    Assert(linearization == coupled_implicit_newton,
           ExcMessage("Only coupled implicit Newton linearization available for "
                      "stationary equation"));

  tau_grad_div = prm.get_double("tau grad div");
  AssertThrow(tau_grad_div >= 0., ExcMessage("Invalid parameter value"));

  max_lin_iteration = prm.get_integer("lin max iterations");
  tol_lin_iteration = prm.get_double("lin tolerance");
  rel_lin_iteration = prm.get_integer("lin relative tolerance") > 0;
  std::string uprec = prm.get("lin velocity preconditioner");
  std::string pprec = prm.get("lin pressure mass preconditioner");
  if (uprec == "ilu")
    precondition_velocity = u_ilu;
  else if (uprec == "ilu scalar")
    precondition_velocity = u_ilu_scalar;
  else if (uprec == "amg linear")
    precondition_velocity = u_amg_linear;
  else if (uprec == "amg")
    precondition_velocity = u_amg;
  else
    AssertThrow(false, ExcMessage("Invalid name"));

  if (pprec == "ilu")
    precondition_pressure = p_mass_ilu;
  else
    precondition_pressure = p_mass_diag;

  iterations_before_inner_solvers = prm.get_integer("lin its before inner solvers");
  prm.leave_subsection();
  prm.leave_subsection();



  prm.enter_subsection("Output options");
  output_filename  = prm.get("output filename");
  output_verbosity = prm.get_integer("output verbosity");
  Assert(output_verbosity <= 3, ExcInternalError());
  output_frequency      = prm.get_double("output frequency");
  print_solution_fields = prm.get_integer("output vtk files");
  if (print_solution_fields > 2)
    print_solution_fields = 1;
  output_wall_times = prm.get_integer("output wall times") > 0;
  output_memory     = prm.get_integer("output memory") > 0;
  prm.leave_subsection();



  prm.enter_subsection("Two phase");
  if (prm.get_double("density") > 0.)
    density = prm.get_double("density");
  density_diff = prm.get_double("density difference");
  if (physical_type == stokes)
    density = density_diff = 0;

  if (prm.get_double("viscosity") > 0.)
    viscosity = prm.get_double("viscosity");
  viscosity_diff = prm.get_double("viscosity difference");

  surface_tension     = prm.get_double("surface tension");
  gravity             = prm.get_double("gravity");
  epsilon             = prm.get_double("epsilon");
  diffusion_length    = prm.get_double("diffusion length");
  contact_angle       = prm.get_double("contact angle");
  pressure_constraint = prm.get_integer("pressure constraint");

  AssertThrow(diffusion_length > 0, ExcMessage("Diffusion length must be positive"));
  AssertThrow(epsilon > 0, ExcMessage("Diffusion length must be positive"));

  concentration_subdivisions = prm.get_integer("concentration subdivisions");

  curvature_correction           = prm.get_integer("curvature correction");
  interpolate_grad_onto_pressure = prm.get_integer("grad pressure compatible");
  surface_tension_from_heaviside = prm.get_integer("localize surface tension");
  approximate_projections        = prm.get_integer("approximate projections");
  ch_do_newton                   = prm.get_integer("Cahn-Hilliard do Newton");
  do_iteration                   = prm.get_integer("full nonlinear iteration");
  n_reinit_steps                 = prm.get_integer("number reinit steps");
  n_initial_reinit_steps         = prm.get_integer("number initial reinit steps");
  convection_stabilization       = prm.get_integer("convection stabilization");

  prm.leave_subsection();



  prm.enter_subsection("Time stepping");
  start_time           = (prm.get_double("start time"));
  end_time             = (prm.get_double("end time"));
  time_step_size_start = (prm.get_double("step size"));
  time_stepping_cfl    = (prm.get_double("CFL number"));
  time_stepping_coef2  = (prm.get_double("CFL number capillary"));
  time_step_tolerance  = prm.get_double("tolerance");
  time_step_size_max   = prm.get_double("max step size");
  time_step_size_min   = prm.get_double("min step size");
  // no adaptive time stepping in case the start step was large
  if (time_step_size_min > time_step_size_start)
    time_step_size_max = time_step_size_min = time_step_size_start;

  const std::string schem = prm.get("scheme");
  if (schem == std::string("implicit_euler"))
    time_step_scheme = TimeSteppingParameters::Scheme::implicit_euler;
  else if (schem == std::string("explicit_euler"))
    time_step_scheme = TimeSteppingParameters::Scheme::explicit_euler;
  else if (schem == std::string("crank_nicolson"))
    time_step_scheme = TimeSteppingParameters::Scheme::crank_nicolson;
  else if (schem == std::string("bdf_2"))
    time_step_scheme = TimeSteppingParameters::Scheme::bdf_2;
  else
    // parameter handler should make sure that we
    // never end up here
    AssertThrow(false, ExcInternalError());

  prm.leave_subsection();
}
