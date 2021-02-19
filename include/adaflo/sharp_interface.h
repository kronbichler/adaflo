// --------------------------------------------------------------------------
//
// Copyright (C) 2021 by the adaflo authors
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

#ifndef __adaflo_block_sharp_inteface_h
#define __adaflo_block_sharp_inteface_h


#include <deal.II/fe/fe_point_evaluation.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/numerics/vector_tools.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <adaflo/level_set_okz_advance_concentration.h>
#include <adaflo/level_set_okz_compute_curvature.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/level_set_okz_preconditioner.h>
#include <adaflo/level_set_okz_reinitialization.h>
#include <adaflo/sharp_interface_util.h>
#include <adaflo/util.h>

#include <filesystem>

template <int dim>
class LevelSetSolver
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  static const unsigned int dof_index_ls        = 1;
  static const unsigned int dof_index_normal    = 2;
  static const unsigned int dof_index_curvature = 3;
  static const unsigned int dof_index_velocity  = 0;
  static const unsigned int quad_index          = 0;
  static const unsigned int quad_index_vel      = 1;

  LevelSetSolver(
    const Triangulation<dim> &tria,
    const Function<dim> &     initial_values_ls,
    const FlowParameters &    parameters,
    const TimeStepping &      time_stepping,
    VectorType &              velocity_solution,
    VectorType &              velocity_solution_old,
    VectorType &              velocity_solution_old_old,
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &fluid_type,
    const std::set<types::boundary_id> &                                symmetry)
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , parameters(parameters)
    , time_stepping(time_stepping)
    , last_concentration_range(-1, +1)
    , first_reinit_step(true)
    , dof_handler(tria)
    , dof_handler_dim(tria)
    , velocity_solution(velocity_solution)
    , velocity_solution_old(velocity_solution_old)
    , velocity_solution_old_old(velocity_solution_old_old)
    , normal_vector_field(dim)
    , normal_vector_rhs(dim)
  {
    {
      LevelSetOKZSolverComputeNormalParameter params;
      params.dof_index_ls            = dof_index_ls;
      params.dof_index_normal        = dof_index_normal;
      params.quad_index              = quad_index;
      params.epsilon                 = this->parameters.epsilon;
      params.approximate_projections = this->parameters.approximate_projections;

      normal_operator =
        std::make_unique<LevelSetOKZSolverComputeNormal<dim>>(normal_vector_field,
                                                              normal_vector_rhs,
                                                              ls_solution,
                                                              cell_diameters,
                                                              epsilon_used,
                                                              minimal_edge_length,
                                                              constraints_normals,
                                                              params,
                                                              matrix_free,
                                                              preconditioner,
                                                              projection_matrix,
                                                              ilu_projection_matrix);
    }

    {
      LevelSetOKZSolverReinitializationParameter params;
      params.dof_index_ls     = dof_index_ls;
      params.dof_index_normal = dof_index_normal;
      params.quad_index       = quad_index;
      params.do_iteration     = this->parameters.do_iteration;

      params.time.time_step_scheme     = this->parameters.time_step_scheme;
      params.time.start_time           = this->parameters.start_time;
      params.time.end_time             = this->parameters.end_time;
      params.time.time_step_size_start = this->parameters.time_step_size_start;
      params.time.time_stepping_cfl    = this->parameters.time_stepping_cfl;
      params.time.time_stepping_coef2  = this->parameters.time_stepping_coef2;
      params.time.time_step_tolerance  = this->parameters.time_step_tolerance;
      params.time.time_step_size_max   = this->parameters.time_step_size_max;
      params.time.time_step_size_min   = this->parameters.time_step_size_min;

      reinit =
        std::make_unique<LevelSetOKZSolverReinitialization<dim>>(normal_vector_field,
                                                                 cell_diameters,
                                                                 epsilon_used,
                                                                 minimal_edge_length,
                                                                 constraints,
                                                                 ls_update,
                                                                 ls_solution,
                                                                 ls_rhs,
                                                                 pcout,
                                                                 preconditioner,
                                                                 last_concentration_range,
                                                                 params,
                                                                 first_reinit_step,
                                                                 matrix_free);
    }

    {
      LevelSetOKZSolverComputeCurvatureParameter params;
      params.dof_index_ls            = dof_index_ls;
      params.dof_index_curvature     = dof_index_curvature;
      params.dof_index_normal        = dof_index_normal;
      params.quad_index              = quad_index;
      params.epsilon                 = this->parameters.epsilon;
      params.approximate_projections = this->parameters.approximate_projections;
      params.curvature_correction    = this->parameters.curvature_correction;

      curvature_operator =
        std::make_unique<LevelSetOKZSolverComputeCurvature<dim>>(cell_diameters,
                                                                 normal_vector_field,
                                                                 constraints_curvature,
                                                                 constraints,
                                                                 epsilon_used,
                                                                 curvature_rhs,
                                                                 params,
                                                                 curvature_solution,
                                                                 ls_solution,
                                                                 matrix_free,
                                                                 preconditioner,
                                                                 projection_matrix,
                                                                 ilu_projection_matrix);
    }

    {
      LevelSetOKZSolverAdvanceConcentrationParameter params;

      params.dof_index_ls             = dof_index_ls;
      params.dof_index_vel            = dof_index_velocity;
      params.quad_index               = quad_index;
      params.convection_stabilization = this->parameters.convection_stabilization;
      params.do_iteration             = this->parameters.do_iteration;
      params.tol_nl_iteration         = this->parameters.tol_nl_iteration;

      LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor<dim> bcs;

      bcs.dirichlet = fluid_type;
      bcs.symmetry  = symmetry;

      params.time.time_step_scheme     = this->parameters.time_step_scheme;
      params.time.start_time           = this->parameters.start_time;
      params.time.end_time             = this->parameters.end_time;
      params.time.time_step_size_start = this->parameters.time_step_size_start;
      params.time.time_stepping_cfl    = this->parameters.time_stepping_cfl;
      params.time.time_stepping_coef2  = this->parameters.time_stepping_coef2;
      params.time.time_step_tolerance  = this->parameters.time_step_tolerance;
      params.time.time_step_size_max   = this->parameters.time_step_size_max;
      params.time.time_step_size_min   = this->parameters.time_step_size_min;

      this->advection_operator =
        std::make_unique<LevelSetOKZSolverAdvanceConcentration<dim>>(
          ls_solution,
          ls_solution_old,
          ls_solution_old_old,
          ls_update,
          ls_rhs,
          velocity_solution,
          velocity_solution_old,
          velocity_solution_old_old,
          cell_diameters,
          this->constraints,
          this->pcout,
          bcs,
          this->matrix_free,
          params,
          this->preconditioner);
    }

    // MatrixFree
    {
      const FE_Q_iso_Q1<dim> fe(parameters.concentration_subdivisions);
      dof_handler.distribute_dofs(fe);

      pcout << "Number of level set degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

      typename MatrixFree<dim>::AdditionalData data;

      data.tasks_parallel_scheme =
        Utilities::MPI::n_mpi_processes(get_communicator(dof_handler)) > 1 ?
          MatrixFree<dim>::AdditionalData::none :
          MatrixFree<dim>::AdditionalData::partition_color;
      if (parameters.velocity_degree == 2)
        data.tasks_block_size = 16;
      else
        data.tasks_block_size = 2;
      data.store_plain_indices = true;

      const FESystem<dim> fe_dim(FE_Q<dim>(parameters.velocity_degree), dim);
      dof_handler_dim.distribute_dofs(fe_dim);

      const QIterated<dim> quad(QGauss<1>(2), fe.degree);
      const QGauss<dim>    quad_vel(parameters.velocity_degree + 1);

      for (const auto &i : fluid_type)
        VectorTools::interpolate_boundary_values(mapping,
                                                 dof_handler,
                                                 i.first,
                                                 Functions::ConstantFunction<dim>(0.0),
                                                 constraints);

      constraints.close();
      constraints_curvature.close();
      constraints_normals.close();
      constraints_force.close();
      hanging_node_constraints.close();

      const std::vector<const DoFHandler<dim> *> dof_handlers{&dof_handler_dim,
                                                              &dof_handler,
                                                              &dof_handler,
                                                              &dof_handler};

      const std::vector<const AffineConstraints<double> *> all_constraints{
        &constraints_force, &constraints, &constraints_normals, &constraints_curvature};

      const std::vector<Quadrature<dim>> quadratures{quad, quad_vel};

      matrix_free.reinit(mapping, dof_handlers, all_constraints, quadratures, data);
    }

    // Vectors
    {
      initialize_dof_vector(ls_solution, dof_index_ls);
      initialize_dof_vector(ls_solution_old, dof_index_ls);
      initialize_dof_vector(ls_solution_old_old, dof_index_ls);
      initialize_dof_vector(ls_update, dof_index_ls);
      initialize_dof_vector(ls_rhs, dof_index_ls);
      initialize_dof_vector(curvature_solution, dof_index_curvature);
      initialize_dof_vector(curvature_rhs, dof_index_curvature);

      for (unsigned int i = 0; i < dim; ++i)
        {
          initialize_dof_vector(normal_vector_field.block(i), dof_index_normal);
          initialize_dof_vector(normal_vector_rhs.block(i), dof_index_normal);
        }
    }

    // miscellaneous
    {
      compute_cell_diameters(
        matrix_free, dof_index_ls, cell_diameters, minimal_edge_length, epsilon_used);

      pcout << "Mesh size (largest/smallest element length at finest level): "
            << epsilon_used << " / " << minimal_edge_length << std::endl;

      epsilon_used =
        parameters.epsilon / parameters.concentration_subdivisions * epsilon_used;

      initialize_mass_matrix_diagonal(
        matrix_free, hanging_node_constraints, dof_index_ls, quad_index, preconditioner);

      projection_matrix     = std::make_shared<BlockMatrixExtension>();
      ilu_projection_matrix = std::make_shared<BlockILUExtension>();

      initialize_projection_matrix(matrix_free,
                                   constraints_normals,
                                   dof_index_ls,
                                   quad_index,
                                   epsilon_used,
                                   this->parameters.epsilon,
                                   cell_diameters,
                                   *projection_matrix,
                                   *ilu_projection_matrix);
    }

    VectorTools::interpolate(mapping, dof_handler, initial_values_ls, ls_solution);


    // transform_distance_function
    for (unsigned int i = 0; i < ls_solution.local_size(); i++)
      ls_solution.local_element(i) =
        -std::tanh(ls_solution.local_element(i) / (2. * epsilon_used));

    velocity_solution_old     = velocity_solution;
    velocity_solution_old_old = velocity_solution;
  }

  void
  initialize()
  {
    reinitialize(true);
  }

  void
  initialize_dof_vector(VectorType &vec, const unsigned int dof_index)
  {
    matrix_free.initialize_dof_vector(vec, dof_index);
  }

  void
  solve()
  {
    const double step_size     = time_stepping.step_size();
    const double step_size_old = time_stepping.old_step_size();
    ls_update                  = ls_solution;

    if (step_size_old > 0)
      ls_update.sadd((step_size + step_size_old) / step_size_old,
                     -step_size / step_size_old,
                     ls_solution_old);

    ls_solution_old_old = ls_solution_old;
    ls_solution_old     = ls_solution;
    ls_solution         = ls_update;

    ls_solution.update_ghost_values();
    ls_solution_old.update_ghost_values();
    ls_solution_old_old.update_ghost_values();

    this->advance_concentration();
    this->reinitialize();
  }

  const VectorType &
  get_level_set_vector()
  {
    return ls_solution;
  }

  const BlockVectorType &
  get_normal_vector()
  {
    return normal_vector_field;
  }

  const VectorType &
  get_curvature_vector()
  {
    return curvature_solution;
  }

  const DoFHandler<dim> &
  get_dof_handler() const
  {
    return dof_handler;
  }

  const DoFHandler<dim> &
  get_dof_handler_dim() const
  {
    return dof_handler_dim;
  }

  MatrixFree<dim, double> &
  get_matrix_free()
  {
    return matrix_free;
  }

private:
  void
  advance_concentration()
  {
    advection_operator->advance_concentration(this->time_stepping.step_size());
  }

  void
  reinitialize(const bool initialization = false)
  {
    const double       dt         = this->time_stepping.step_size();
    const unsigned int stab_steps = initialization ?
                                      this->parameters.n_initial_reinit_steps :
                                      this->parameters.n_reinit_steps;
    const unsigned int diff_steps = 0;

    reinit->reinitialize(dt, stab_steps, diff_steps, [this](const bool fast) {
      normal_operator->compute_normal(fast);
    });

    normal_operator->compute_normal(/*fast_computation*/ false);

    curvature_operator->compute_curvature(/*diffuse_large_values*/ false);
  }

  ConditionalOStream    pcout;
  const FlowParameters &parameters;
  const TimeStepping &  time_stepping;


  std::pair<double, double>              last_concentration_range;
  bool                                   first_reinit_step;
  AlignedVector<VectorizedArray<double>> cell_diameters;
  double                                 minimal_edge_length;
  double                                 epsilon_used;

  MappingQ1<dim> mapping;

  // DoFHandlers
  DoFHandler<dim> dof_handler;     // for ls, normal, curvature
  DoFHandler<dim> dof_handler_dim; // for velocity field

  // Constraints
  AffineConstraints<double> constraints;
  AffineConstraints<double> constraints_normals;
  AffineConstraints<double> hanging_node_constraints;
  AffineConstraints<double> constraints_curvature;
  AffineConstraints<double> constraints_force;

  // MatrixFree
  MatrixFree<dim, double> matrix_free;

  // vectors: velocity (external)
  VectorType &velocity_solution;
  VectorType &velocity_solution_old;
  VectorType &velocity_solution_old_old;

  // ... level set
  VectorType ls_solution;
  VectorType ls_solution_old;
  VectorType ls_solution_old_old;
  VectorType ls_update;
  VectorType ls_rhs;

  // ... normal
  BlockVectorType normal_vector_field;
  BlockVectorType normal_vector_rhs;

  // ... curvature
  VectorType curvature_solution;
  VectorType curvature_rhs;

  // Preconditioners
  DiagonalPreconditioner<double>        preconditioner;
  std::shared_ptr<BlockMatrixExtension> projection_matrix;
  std::shared_ptr<BlockILUExtension>    ilu_projection_matrix;

  // Operators
  std::unique_ptr<LevelSetOKZSolverComputeNormal<dim>>    normal_operator;
  std::unique_ptr<LevelSetOKZSolverReinitialization<dim>> reinit;
  std::unique_ptr<LevelSetOKZSolverComputeCurvature<dim>> curvature_operator;

public:
  std::unique_ptr<LevelSetOKZSolverAdvanceConcentration<dim>> advection_operator;
};



class SharpInterfaceSolver
{
public:
  virtual void
  advance_time_step() = 0;

  virtual void
  output_solution(const std::string &output_filename) = 0;
};



template <int dim>
class FrontTrackingSolver : public SharpInterfaceSolver
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  FrontTrackingSolver(NavierStokes<dim> &          navier_stokes_solver,
                      Triangulation<dim - 1, dim> &surface_mesh)
    : navier_stokes_solver(navier_stokes_solver)
    , surface_dofhandler_dim(surface_mesh)
    , surface_dofhandler(surface_mesh)
  {
    const unsigned int fe_degree      = 3;
    const unsigned int mapping_degree = fe_degree;

    FESystem<dim - 1, dim> euler_fe(FE_Q<dim - 1, dim>(fe_degree), dim);
    surface_dofhandler_dim.distribute_dofs(euler_fe);

    surface_dofhandler.distribute_dofs(FE_Q<dim - 1, dim>(fe_degree));

    surface_coordinates_vector.reinit(surface_dofhandler_dim.n_dofs());
    VectorTools::get_position_vector(surface_dofhandler_dim,
                                     surface_coordinates_vector,
                                     MappingQGeneric<dim - 1, dim>(mapping_degree));

    euler_mapping = std::make_shared<MappingFEField<dim - 1, dim, VectorType>>(
      surface_dofhandler_dim, surface_coordinates_vector);

    this->update_phases();
    this->update_gravity_force();
    this->update_surface_tension();
  }

  void
  advance_time_step() override
  {
    this->move_surface_mesh();
    this->update_phases();
    this->update_gravity_force();
    this->update_surface_tension();

    navier_stokes_solver.get_constraints_u().set_zero(
      navier_stokes_solver.user_rhs.block(0));
    navier_stokes_solver.advance_time_step();
  }

  void
  output_solution(const std::string &output_filename) override
  {
    // background mesh
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);

      navier_stokes_solver.solution.update_ghost_values();

      data_out.add_data_vector(navier_stokes_solver.get_dof_handler_u(),
                               navier_stokes_solver.solution.block(0),
                               std::vector<std::string>(dim, "velocity"),
                               vector_component_interpretation);

      data_out.add_data_vector(navier_stokes_solver.get_dof_handler_u(),
                               navier_stokes_solver.user_rhs.block(0),
                               std::vector<std::string>(dim, "user_rhs"),
                               vector_component_interpretation);

      data_out.add_data_vector(navier_stokes_solver.get_dof_handler_p(),
                               navier_stokes_solver.solution.block(1),
                               "pressure");

      data_out.build_patches(navier_stokes_solver.mapping,
                             navier_stokes_solver.get_dof_handler_u().get_fe().degree +
                               1);

      navier_stokes_solver.write_data_output(
        output_filename,
        navier_stokes_solver.time_stepping,
        navier_stokes_solver.get_parameters().output_frequency,
        navier_stokes_solver.get_dof_handler_u().get_triangulation(),
        data_out);
    }

    // surface mesh
    {
      DataOutBase::VtkFlags flags;

      DataOut<dim - 1, DoFHandler<dim - 1, dim>> data_out;
      data_out.set_flags(flags);
      data_out.add_data_vector(surface_dofhandler, curvature_vector, "curvature");
      data_out.add_data_vector(surface_dofhandler_dim, normal_vector, "normal");

      data_out.build_patches(
        *euler_mapping,
        surface_dofhandler_dim.get_fe().degree + 1,
        DataOut<dim - 1, DoFHandler<dim - 1, dim>>::CurvedCellRegion::curved_inner_cells);

      std::filesystem::path path(output_filename + "_surface");

      data_out.write_vtu_with_pvtu_record(path.parent_path().string() + "/",
                                          path.filename(),
                                          navier_stokes_solver.time_stepping.step_no(),
                                          MPI_COMM_WORLD);
    }
  }

private:
  void
  move_surface_mesh()
  {
    VectorTools::update_position_vector(navier_stokes_solver.time_stepping.step_size(),
                                        navier_stokes_solver.get_dof_handler_u(),
                                        navier_stokes_solver.mapping,
                                        navier_stokes_solver.solution.block(0),
                                        surface_dofhandler_dim,
                                        *euler_mapping,
                                        surface_coordinates_vector);
  }

  void
  update_phases()
  {
    const auto density        = navier_stokes_solver.get_parameters().density;
    const auto density_diff   = navier_stokes_solver.get_parameters().density_diff;
    const auto viscosity      = navier_stokes_solver.get_parameters().viscosity;
    const auto viscosity_diff = navier_stokes_solver.get_parameters().viscosity_diff;

    if (density_diff == 0.0 && viscosity_diff == 0.0)
      return; // nothing to do

    boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double>> polygon;
    GridTools::construct_polygon(*euler_mapping, surface_dofhandler_dim, polygon);

    double dummy;

    navier_stokes_solver.matrix_free->template cell_loop<double, double>(
      [&](const auto &matrix_free, auto &, const auto &, auto macro_cells) {
        FEEvaluation<dim, -1, 0, 1, double> phi(matrix_free, 0, 0);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            phi.reinit(cell);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                const auto indicator =
                  GridTools::within(polygon, phi.quadrature_point(q));

                navier_stokes_solver.get_matrix().begin_densities(cell)[q] =
                  density + density_diff * indicator;
                navier_stokes_solver.get_matrix().begin_viscosities(cell)[q] =
                  viscosity + viscosity_diff * indicator;
              }
          }
      },
      dummy,
      dummy);
  }

  void
  update_surface_tension()
  {
    // return; // TODO: not working

    normal_vector.reinit(surface_dofhandler_dim.n_dofs());
    curvature_vector.reinit(surface_dofhandler.n_dofs());

    compute_normal(*euler_mapping, surface_dofhandler_dim, normal_vector);
    compute_curvature(*euler_mapping,
                      surface_dofhandler_dim,
                      surface_dofhandler,
                      QGaussLobatto<dim - 1>(surface_dofhandler.get_fe().degree + 1),
                      normal_vector,
                      curvature_vector);

    compute_force_vector_sharp_interface(
      *euler_mapping,
      surface_dofhandler,
      surface_dofhandler_dim,
      QGauss<dim - 1>(surface_dofhandler_dim.get_fe().degree + 1),
      navier_stokes_solver.mapping,
      navier_stokes_solver.get_dof_handler_u(),
      navier_stokes_solver.get_parameters().surface_tension,
      normal_vector,
      curvature_vector,
      navier_stokes_solver.user_rhs.block(0));
  }

  void
  update_gravity_force()
  {
    const auto gravity = navier_stokes_solver.get_parameters().gravity;

    const auto density      = navier_stokes_solver.get_parameters().density;
    const auto density_diff = navier_stokes_solver.get_parameters().density_diff;

    const bool zero_out = true;

    navier_stokes_solver.matrix_free->template cell_loop<VectorType, std::nullptr_t>(
      [&](const auto &matrix_free, auto &vec, const auto &, auto macro_cells) {
        FEEvaluation<dim, -1, 0, dim, double> phi(matrix_free, 0, 0);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            phi.reinit(cell);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                Tensor<1, dim, VectorizedArray<double>> force;

                force[dim - 1] -=
                  gravity *
                  (density_diff == 0.0 ?
                     VectorizedArray<double>(density) :
                     navier_stokes_solver.get_matrix().begin_densities(cell)[q]);
                phi.submit_value(force, q);
              }
            phi.integrate_scatter(true, false, vec);
          }
      },
      navier_stokes_solver.user_rhs.block(0),
      nullptr,
      zero_out);
  }

  // background mesh
  NavierStokes<dim> &navier_stokes_solver;

  // surface mesh
  DoFHandler<dim - 1, dim>               surface_dofhandler_dim;
  DoFHandler<dim - 1, dim>               surface_dofhandler;
  VectorType                             surface_coordinates_vector;
  std::shared_ptr<Mapping<dim - 1, dim>> euler_mapping;

  VectorType normal_vector;
  VectorType curvature_vector;
};



template <int dim>
class MixedLevelSetSolver : public SharpInterfaceSolver
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  MixedLevelSetSolver(NavierStokes<dim> &          navier_stokes_solver,
                      Triangulation<dim - 1, dim> &surface_mesh,
                      const Function<dim> &        initial_values_ls)
    : use_auxiliary_surface_mesh(true)
    , use_sharp_interface(true)
    , decoupled_meshes(false)
    , navier_stokes_solver(navier_stokes_solver)
    , level_set_solver(navier_stokes_solver.get_dof_handler_u().get_triangulation(),
                       initial_values_ls,
                       navier_stokes_solver.get_parameters(),
                       navier_stokes_solver.time_stepping,
                       navier_stokes_solver.solution.block(0),
                       navier_stokes_solver.solution_old.block(0),
                       navier_stokes_solver.solution_old_old.block(0),
                       navier_stokes_solver.boundary->fluid_type,
                       navier_stokes_solver.boundary->symmetry)
    , euler_dofhandler(surface_mesh)
  {
    level_set_solver.initialize();

    const unsigned int fe_degree = 1;

    FESystem<dim - 1, dim> surface_fe_dim(FE_Q<dim - 1, dim>(fe_degree), dim);
    euler_dofhandler.distribute_dofs(surface_fe_dim);

    euler_vector.reinit(euler_dofhandler.n_dofs());
    VectorTools::
      get_position_vector(euler_dofhandler, euler_vector, MappingQGeneric<dim - 1, dim>(4 /*TODO: this is a high number to well represent curved surfaces, the actual values is not that relevant*/));
    euler_mapping =
      std::make_shared<MappingFEField<dim - 1, dim, VectorType>>(euler_dofhandler,
                                                                 euler_vector);

    // initialize
    this->update_phases();
    this->update_gravity_force();
    this->update_surface_tension();
  }

  MixedLevelSetSolver(NavierStokes<dim> &  navier_stokes_solver,
                      const Function<dim> &initial_values_ls,
                      const bool           use_sharp_interface = true)
    : use_auxiliary_surface_mesh(false)
    , use_sharp_interface(use_sharp_interface)
    , decoupled_meshes(false)
    , navier_stokes_solver(navier_stokes_solver)
    , level_set_solver(navier_stokes_solver.get_dof_handler_u().get_triangulation(),
                       initial_values_ls,
                       navier_stokes_solver.get_parameters(),
                       navier_stokes_solver.time_stepping,
                       navier_stokes_solver.solution.block(0),
                       navier_stokes_solver.solution_old.block(0),
                       navier_stokes_solver.solution_old_old.block(0),
                       navier_stokes_solver.boundary->fluid_type,
                       navier_stokes_solver.boundary->symmetry)
  {
    level_set_solver.initialize();

    this->update_phases();
    this->update_gravity_force();
    this->update_surface_tension();
  }

  MixedLevelSetSolver(NavierStokes<dim> &                        navier_stokes_solver,
                      parallel::distributed::Triangulation<dim> &tria,
                      const Function<dim> &                      initial_values_ls)
    : use_auxiliary_surface_mesh(false)
    , use_sharp_interface(false)
    , decoupled_meshes(true)
    , navier_stokes_solver(navier_stokes_solver)
    , level_set_solver(tria,
                       initial_values_ls,
                       navier_stokes_solver.get_parameters(),
                       navier_stokes_solver.time_stepping,
                       navier_stokes_solver.solution.block(0),
                       navier_stokes_solver.solution_old.block(0),
                       navier_stokes_solver.solution_old_old.block(0),
                       navier_stokes_solver.boundary->fluid_type,
                       navier_stokes_solver.boundary->symmetry)
  {
    // initialize
    evaluate_velocity_at_quadrature_points();
    evaluate_level_set_and_curvature_at_quadrature_points(); // TODO: to init data
                                                             // structures

    level_set_solver.initialize();

    this->update_phases();
    this->update_gravity_force();
    this->update_surface_tension();
  }

  void
  advance_time_step() override
  {
    level_set_solver.solve();

    if (decoupled_meshes)
      evaluate_level_set_and_curvature_at_quadrature_points();

    if (use_auxiliary_surface_mesh)
      this->move_surface_mesh();
    this->update_phases();
    this->update_gravity_force();
    this->update_surface_tension();

    navier_stokes_solver.get_constraints_u().set_zero(
      navier_stokes_solver.user_rhs.block(0));
    navier_stokes_solver.advance_time_step();

    if (decoupled_meshes)
      evaluate_velocity_at_quadrature_points();
  }

  void
  evaluate_level_set_and_curvature_at_quadrature_points()
  {
    Assert(decoupled_meshes, ExcNotImplemented());

    const unsigned int n_q_points = level_set_solver.get_matrix_free()
                                      .get_quadrature(LevelSetSolver<dim>::quad_index_vel)
                                      .size();
    const unsigned int n_q_points_total =
      n_q_points * level_set_solver.get_matrix_free().n_cell_batches();
    level_set_values.resize(n_q_points_total);
    level_set_gradients.resize(n_q_points_total);
    curvature_values.resize(n_q_points_total);

    const auto &matrix_free = *navier_stokes_solver.matrix_free;

    FEEvaluation<dim, -1, 0, dim, double> vel_values(matrix_free,
                                                     navier_stokes_solver.dof_index_u,
                                                     navier_stokes_solver.quad_index_u);

    std::vector<Point<dim>> evaluation_points;

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        vel_values.reinit(cell);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const auto points = vel_values.quadrature_point(q);
            for (unsigned int v = 0;
                 v < matrix_free.n_active_entries_per_cell_batch(cell);
                 ++v)
              {
                Point<dim> point;

                for (int i = 0; i < dim; ++i)
                  point[i] = points[i][v];

                evaluation_points.push_back(point);
              }
          }
      }

    const auto evaluation_point_results = [&] {
      Utilities::MPI::RemotePointEvaluation<dim, dim> eval;

      eval.reinit(evaluation_points,
                  level_set_solver.get_dof_handler().get_triangulation(),
                  navier_stokes_solver.mapping /*TODO*/);

      FEPointEvaluation<1, dim> evaluator(navier_stokes_solver.mapping /*TODO*/,
                                          level_set_solver.get_dof_handler().get_fe());

      FEPointEvaluation<1, dim> evaluator_curvature(
        navier_stokes_solver.mapping /*TODO*/,
        level_set_solver.get_dof_handler().get_fe());

      std::vector<double> solution_values;
      std::vector<double> solution_values_curvature;

      const auto fu = [&](auto &values, const auto &quadrature_points) {
        unsigned int i = 0;

        for (const auto &cells_and_n : std::get<0>(quadrature_points))
          {
            typename DoFHandler<dim>::active_cell_iterator cell = {
              &navier_stokes_solver.get_dof_handler_u().get_triangulation(),
              cells_and_n.first.first,
              cells_and_n.first.second,
              &navier_stokes_solver.get_dof_handler_u()};

            const ArrayView<const Point<dim>> unit_points(
              std::get<1>(quadrature_points).data() + i, cells_and_n.second);
            solution_values.resize(cell->get_fe().n_dofs_per_cell());
            solution_values_curvature.resize(cell->get_fe().n_dofs_per_cell());

            cell->get_dof_values(level_set_solver.get_level_set_vector(),
                                 solution_values.begin(),
                                 solution_values.end());

            evaluator.evaluate(cell,
                               unit_points,
                               solution_values,
                               EvaluationFlags::values | EvaluationFlags::gradients);

            cell->get_dof_values(level_set_solver.get_level_set_vector(),
                                 solution_values_curvature.begin(),
                                 solution_values_curvature.end());

            evaluator_curvature.evaluate(cell,
                                         unit_points,
                                         solution_values_curvature,
                                         EvaluationFlags::values);

            for (unsigned int q = 0; q < unit_points.size(); ++q, ++i)
              values[std::get<2>(quadrature_points)[i]] = {evaluator.get_value(q),
                                                           evaluator.get_gradient(q),
                                                           evaluator_curvature.get_value(
                                                             q)};
          }
      };

      using T = std::tuple<double, Tensor<1, dim>, double>;

      std::vector<T> evaluation_point_results;
      std::vector<T> buffer;

      eval.template process<T>(evaluation_point_results, buffer, fu);

      return evaluation_point_results;
    }();


    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        vel_values.reinit(cell);

        for (unsigned int q = 0, c = 0; q < n_q_points; ++q)
          for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell);
               ++v, ++c)
            {
              level_set_values[n_q_points * cell + q][v] =
                std::get<0>(evaluation_point_results[c]);
              for (unsigned int d = 0; d < dim; ++d)
                level_set_gradients[n_q_points * cell + q][d][v] =
                  std::get<1>(evaluation_point_results[c])[d];
              curvature_values[n_q_points * cell + q][v] =
                std::get<2>(evaluation_point_results[c]);
            }
      }
  }

  void
  evaluate_velocity_at_quadrature_points()
  {
    Assert(decoupled_meshes, ExcNotImplemented());

    auto &op = *level_set_solver.advection_operator;

    op.velocity_at_quadrature_points_given = true;

    const unsigned int n_q_points =
      navier_stokes_solver.matrix_free->get_quadrature(navier_stokes_solver.quad_index_u)
        .size();
    const unsigned int n_q_points_total =
      n_q_points * navier_stokes_solver.matrix_free->n_cell_batches();

    op.evaluated_vel.resize(n_q_points_total);
    op.evaluated_vel_old.resize(n_q_points_total);
    op.evaluated_vel_old_old.resize(n_q_points_total);

    const auto &matrix_free = *navier_stokes_solver.matrix_free;

    FEEvaluation<dim, -1, 0, dim, double> vel_values(matrix_free,
                                                     navier_stokes_solver.dof_index_u,
                                                     navier_stokes_solver.quad_index_u);

    std::vector<Point<dim>> evaluation_points;

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        vel_values.reinit(cell);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const auto points = vel_values.quadrature_point(q);
            for (unsigned int v = 0;
                 v < matrix_free.n_active_entries_per_cell_batch(cell);
                 ++v)
              {
                Point<dim> point;

                for (int i = 0; i < dim; ++i)
                  point[i] = points[i][v];

                evaluation_points.push_back(point);
              }
          }
      }

    const auto evaluation_point_results = [&] {
      Utilities::MPI::RemotePointEvaluation<dim, dim> eval;

      eval.reinit(evaluation_points,
                  navier_stokes_solver.get_dof_handler_u().get_triangulation(),
                  navier_stokes_solver.mapping);

      FEPointEvaluation<dim, dim> evaluator(
        navier_stokes_solver.mapping, navier_stokes_solver.get_dof_handler_u().get_fe());

      std::vector<double> solution_values;

      std::array<VectorType *, 3> vectors = {
        {&navier_stokes_solver.solution.block(0),
         &navier_stokes_solver.solution_old.block(0),
         &navier_stokes_solver.solution_old_old.block(0)}};

      const auto fu = [&](auto &values, const auto &quadrature_points) {
        unsigned int i = 0;

        for (const auto &cells_and_n : std::get<0>(quadrature_points))
          {
            typename DoFHandler<dim>::active_cell_iterator cell = {
              &navier_stokes_solver.get_dof_handler_u().get_triangulation(),
              cells_and_n.first.first,
              cells_and_n.first.second,
              &navier_stokes_solver.get_dof_handler_u()};

            const ArrayView<const Point<dim>> unit_points(
              std::get<1>(quadrature_points).data() + i, cells_and_n.second);
            solution_values.resize(cell->get_fe().n_dofs_per_cell());

            for (unsigned int v = 0; v < vectors.size(); ++v)
              {
                cell->get_dof_values(*vectors[v],
                                     solution_values.begin(),
                                     solution_values.end());

                evaluator.evaluate(cell,
                                   unit_points,
                                   solution_values,
                                   EvaluationFlags::values);

                for (unsigned int q = 0; q < unit_points.size(); ++q)
                  values[std::get<2>(quadrature_points)[i]][v] = evaluator.get_value(q);
              }
            i += unit_points.size();
          }
      };

      using T = std::array<Tensor<1, dim>, 3>;

      std::vector<T> evaluation_point_results;
      std::vector<T> buffer;

      eval.template process<T>(evaluation_point_results, buffer, fu);

      return evaluation_point_results;
    }();


    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        vel_values.reinit(cell);

        for (unsigned int q = 0, c = 0; q < n_q_points; ++q)
          for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell);
               ++v, ++c)
            for (unsigned int d = 0; d < dim; ++d)
              {
                op.evaluated_vel[n_q_points * cell + q][d][v] =
                  evaluation_point_results[c][0][d];
                op.evaluated_vel_old[n_q_points * cell + q][d][v] =
                  evaluation_point_results[c][1][d];
                op.evaluated_vel_old_old[n_q_points * cell + q][d][v] =
                  evaluation_point_results[c][2][d];
              }
      }
  }

  void
  output_solution(const std::string &output_filename) override
  {
    // background mesh
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);

      navier_stokes_solver.solution.update_ghost_values();

      data_out.add_data_vector(navier_stokes_solver.get_dof_handler_u(),
                               navier_stokes_solver.solution.block(0),
                               std::vector<std::string>(dim, "velocity"),
                               vector_component_interpretation);

      data_out.add_data_vector(navier_stokes_solver.get_dof_handler_u(),
                               navier_stokes_solver.user_rhs.block(0),
                               std::vector<std::string>(dim, "user_rhs"),
                               vector_component_interpretation);

      data_out.add_data_vector(navier_stokes_solver.get_dof_handler_p(),
                               navier_stokes_solver.solution.block(1),
                               "pressure");

      data_out.add_data_vector(level_set_solver.get_dof_handler(),
                               level_set_solver.get_level_set_vector(),
                               "level_set");

      data_out.add_data_vector(level_set_solver.get_dof_handler(),
                               level_set_solver.get_curvature_vector(),
                               "curvature");

      for (unsigned int i = 0; i < dim; ++i)
        data_out.add_data_vector(level_set_solver.get_dof_handler(),
                                 level_set_solver.get_normal_vector().block(i),
                                 "normal_" + std::to_string(i));

      data_out.build_patches(navier_stokes_solver.mapping,
                             navier_stokes_solver.get_dof_handler_u().get_fe().degree +
                               1);

      navier_stokes_solver.write_data_output(
        output_filename,
        navier_stokes_solver.time_stepping,
        navier_stokes_solver.get_parameters().output_frequency,
        navier_stokes_solver.get_dof_handler_u().get_triangulation(),
        data_out);
    }

    // surface mesh
    if (use_auxiliary_surface_mesh)
      {
        DataOutBase::VtkFlags flags;

        DataOut<dim - 1, DoFHandler<dim - 1, dim>> data_out;
        data_out.set_flags(flags);
        data_out.attach_dof_handler(euler_dofhandler);

        data_out.build_patches(
          *euler_mapping,
          euler_dofhandler.get_fe().degree + 1,
          DataOut<dim - 1,
                  DoFHandler<dim - 1, dim>>::CurvedCellRegion::curved_inner_cells);

        std::filesystem::path path(output_filename + "_surface");

        data_out.write_vtu_with_pvtu_record(path.parent_path().string() + "/",
                                            path.filename(),
                                            navier_stokes_solver.time_stepping.step_no(),
                                            MPI_COMM_WORLD);
      }
  }

private:
  void
  move_surface_mesh()
  {
    Assert(use_auxiliary_surface_mesh, ExcNotImplemented());

    VectorTools::update_position_vector(navier_stokes_solver.time_stepping.step_size(),
                                        navier_stokes_solver.get_dof_handler_u(),
                                        navier_stokes_solver.mapping,
                                        navier_stokes_solver.solution.block(0),
                                        euler_dofhandler,
                                        *euler_mapping,
                                        euler_vector);
  }

  void
  update_phases()
  {
    const auto density        = navier_stokes_solver.get_parameters().density;
    const auto density_diff   = navier_stokes_solver.get_parameters().density_diff;
    const auto viscosity      = navier_stokes_solver.get_parameters().viscosity;
    const auto viscosity_diff = navier_stokes_solver.get_parameters().viscosity_diff;

    if (density_diff == 0.0 && viscosity_diff == 0.0)
      return; // nothing to do

    if (!decoupled_meshes)
      {
        double dummy;

        // TODO: select proper MatrixFree object and set right dof/quad index
        level_set_solver.get_matrix_free().template cell_loop<double, VectorType>(
          [&](const auto &matrix_free, auto &, const auto &src, auto macro_cells) {
            FEEvaluation<dim, -1, 0, 1, double> phi(matrix_free,
                                                    LevelSetSolver<dim>::dof_index_ls,
                                                    LevelSetSolver<dim>::quad_index_vel);

            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                phi.reinit(cell);
                phi.gather_evaluate(src, EvaluationFlags::values);

                for (unsigned int q = 0; q < phi.n_q_points; ++q)
                  {
                    const auto indicator =
                      (phi.get_value(q) + 1.0) / 2.0; // TODO: fix indicator -> Heaviside

                    navier_stokes_solver.get_matrix().begin_densities(cell)[q] =
                      density + density_diff * indicator;
                    navier_stokes_solver.get_matrix().begin_viscosities(cell)[q] =
                      viscosity + viscosity_diff * indicator;
                  }
              }
          },
          dummy,
          level_set_solver.get_level_set_vector());
      }
    else
      {
        double dummy;

        navier_stokes_solver.matrix_free->template cell_loop<double, VectorType>(
          [&](const auto &matrix_free, auto &, const auto &, auto macro_cells) {
            const unsigned int n_q_points =
              matrix_free.get_quadrature(navier_stokes_solver.quad_index_u).size();

            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const auto indicator = level_set_values[n_q_points * cell + q];

                    navier_stokes_solver.get_matrix().begin_densities(cell)[q] =
                      density + density_diff * indicator;
                    navier_stokes_solver.get_matrix().begin_viscosities(cell)[q] =
                      viscosity + viscosity_diff * indicator;
                  }
              }
          },
          dummy,
          dummy);
      }
  }

  void
  update_surface_tension()
  {
    if (use_auxiliary_surface_mesh && use_sharp_interface)
      compute_force_vector_sharp_interface<dim>(euler_dofhandler.get_triangulation(),
                                                *euler_mapping,
                                                euler_dofhandler.get_fe().base_element(0),
                                                QGauss<dim - 1>(
                                                  euler_dofhandler.get_fe().degree + 1),
                                                navier_stokes_solver.mapping,
                                                level_set_solver.get_dof_handler(),
                                                navier_stokes_solver.get_dof_handler_u(),
                                                level_set_solver.get_normal_vector(),
                                                level_set_solver.get_curvature_vector(),
                                                navier_stokes_solver.user_rhs.block(0));
    else if (!use_auxiliary_surface_mesh && use_sharp_interface)
      compute_force_vector_sharp_interface(QGauss<dim - 1>(2 /*TODO*/),
                                           navier_stokes_solver.mapping,
                                           level_set_solver.get_dof_handler(),
                                           navier_stokes_solver.get_dof_handler_u(),
                                           level_set_solver.get_normal_vector(),
                                           level_set_solver.get_curvature_vector(),
                                           level_set_solver.get_level_set_vector(),
                                           navier_stokes_solver.user_rhs.block(0));
    else if (!use_auxiliary_surface_mesh && !use_sharp_interface && !decoupled_meshes)
      compute_force_vector_regularized(level_set_solver.get_matrix_free(),
                                       level_set_solver.get_level_set_vector(),
                                       level_set_solver.get_curvature_vector(),
                                       navier_stokes_solver.user_rhs.block(0),
                                       LevelSetSolver<dim>::dof_index_ls,
                                       LevelSetSolver<dim>::dof_index_curvature,
                                       LevelSetSolver<dim>::dof_index_velocity,
                                       LevelSetSolver<dim>::quad_index_vel);
    else if (!use_auxiliary_surface_mesh && !use_sharp_interface && decoupled_meshes)
      compute_force_vector_regularized(*navier_stokes_solver.matrix_free,
                                       level_set_gradients,
                                       curvature_values,
                                       navier_stokes_solver.user_rhs.block(0),
                                       navier_stokes_solver.dof_index_u,
                                       navier_stokes_solver.quad_index_u);
    else
      AssertThrow(false, ExcNotImplemented());
  }

  void
  update_gravity_force()
  {
    const auto gravity = navier_stokes_solver.get_parameters().gravity;

    const auto density      = navier_stokes_solver.get_parameters().density;
    const auto density_diff = navier_stokes_solver.get_parameters().density_diff;

    const bool zero_out = true;

    navier_stokes_solver.matrix_free->template cell_loop<VectorType, std::nullptr_t>(
      [&](const auto &matrix_free, auto &vec, const auto &, auto macro_cells) {
        FEEvaluation<dim, -1, 0, dim, double> phi(matrix_free, 0, 0);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            phi.reinit(cell);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                Tensor<1, dim, VectorizedArray<double>> force;

                force[dim - 1] -=
                  gravity *
                  (density_diff == 0.0 ?
                     VectorizedArray<double>(density) :
                     navier_stokes_solver.get_matrix().begin_densities(cell)[q]);
                phi.submit_value(force, q);
              }
            phi.integrate_scatter(true, false, vec);
          }
      },
      navier_stokes_solver.user_rhs.block(0),
      nullptr,
      zero_out);
  }


  AlignedVector<VectorizedArray<double>>                 level_set_values;
  AlignedVector<Tensor<1, dim, VectorizedArray<double>>> level_set_gradients;
  AlignedVector<VectorizedArray<double>>                 curvature_values;

  const bool use_auxiliary_surface_mesh;
  const bool use_sharp_interface;
  const bool decoupled_meshes;

  // background mesh
  NavierStokes<dim> & navier_stokes_solver;
  LevelSetSolver<dim> level_set_solver;

  // surface mesh
  DoFHandler<dim - 1, dim>               euler_dofhandler;
  VectorType                             euler_vector;
  std::shared_ptr<Mapping<dim - 1, dim>> euler_mapping;
};

#endif
