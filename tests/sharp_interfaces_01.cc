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

#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_point_evaluation.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <adaflo/level_set_okz_compute_curvature.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/level_set_okz_preconditioner.h>
#include <adaflo/level_set_okz_reinitialization.h>
#include <adaflo/util.h>

#include "sharp_interfaces_util.h"

using VectorType      = LinearAlgebra::distributed::Vector<double>;
using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

static const unsigned int dof_index_ls        = 0;
static const unsigned int dof_index_normal    = 1;
static const unsigned int dof_index_curvature = 2;
static const unsigned int dof_index_force     = 3;
static const unsigned int quad_index          = 0;


template <int dim>
class VelocityFunction : public Function<dim>
{
public:
  VelocityFunction()
    : Function<dim>(dim)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const
  {
    return component == 0 ? p[1] : -p[0];
  }
};

template <int dim>
class InitialValuesLS : public Function<dim>
{
public:
  InitialValuesLS()
    : Function<dim>(1, 0)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const
  {
    (void)component;
    AssertDimension(component, 0);

    const double radius = 0.25;
    Point<dim>   origin(0.0, 0.5);
    return (radius - p.distance(origin) > 0.0) ? +1.0 : -1.0;
  }
};

template <int dim, int spacedim>
void
create_surface_mesh(Triangulation<dim, spacedim> &tria)
{
  GridGenerator::hyper_sphere(tria, Point<spacedim>(0, 0.5), 0.25);
  tria.refine_global(5);
}

template <int dim>
void
compute_ls_normal_curvature(const MatrixFree<dim, double> &  matrix_free,
                            const AffineConstraints<double> &constraints,
                            const AffineConstraints<double> &constraints_normals,
                            const AffineConstraints<double> &hanging_node_constraints,
                            const AffineConstraints<double> &constraints_curvature,
                            BlockVectorType &                normal_vector_field,
                            VectorType &                     ls_solution,
                            VectorType &                     curvature_solution)
{
  //
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  // vectors
  BlockVectorType normal_vector_rhs(dim);
  VectorType      ls_solution_update;
  VectorType      ls_system_rhs;
  VectorType      curvature_rhs;

  matrix_free.initialize_dof_vector(ls_solution_update, dof_index_ls);
  matrix_free.initialize_dof_vector(ls_system_rhs, dof_index_ls);
  matrix_free.initialize_dof_vector(curvature_rhs, dof_index_curvature);

  for (unsigned int i = 0; i < dim; ++i)
    matrix_free.initialize_dof_vector(normal_vector_rhs.block(i), dof_index_normal);

  // TODO
  const double              dt                         = 0.01;
  const unsigned int        stab_steps                 = 20;
  std::pair<double, double> last_concentration_range   = {-1, +1};
  bool                      first_reinit_step          = true;
  double                    epsilon                    = 1.5;
  const unsigned int        concentration_subdivisions = 1;

  AlignedVector<VectorizedArray<double>> cell_diameters;
  double                                 minimal_edge_length;
  double                                 epsilon_used;
  compute_cell_diameters(
    matrix_free, dof_index_ls, cell_diameters, minimal_edge_length, epsilon_used);

  pcout << "Mesh size (largest/smallest element length at finest level): " << epsilon_used
        << " / " << minimal_edge_length << std::endl;

  epsilon_used = epsilon / concentration_subdivisions * epsilon_used;

  // preconditioner
  DiagonalPreconditioner<double> preconditioner;

  initialize_mass_matrix_diagonal(
    matrix_free, hanging_node_constraints, dof_index_ls, quad_index, preconditioner);

  auto projection_matrix     = std::make_shared<BlockMatrixExtension>();
  auto ilu_projection_matrix = std::make_shared<BlockILUExtension>();

  initialize_projection_matrix(matrix_free,
                               constraints_normals,
                               dof_index_ls,
                               quad_index,
                               epsilon_used,
                               epsilon,
                               cell_diameters,
                               *projection_matrix,
                               *ilu_projection_matrix);

  // normal operator
  LevelSetOKZSolverComputeNormalParameter nomral_parameter;
  nomral_parameter.dof_index_ls            = dof_index_ls;
  nomral_parameter.dof_index_normal        = dof_index_normal;
  nomral_parameter.quad_index              = quad_index;
  nomral_parameter.epsilon                 = epsilon;
  nomral_parameter.approximate_projections = false;

  LevelSetOKZSolverComputeNormal<dim> normal_operator(normal_vector_field,
                                                      normal_vector_rhs,
                                                      ls_solution,
                                                      cell_diameters,
                                                      epsilon_used,
                                                      minimal_edge_length,
                                                      constraints_normals,
                                                      nomral_parameter,
                                                      matrix_free,
                                                      preconditioner,
                                                      projection_matrix,
                                                      ilu_projection_matrix);

  // reinitialization operator
  LevelSetOKZSolverReinitializationParameter reinit_parameters;
  reinit_parameters.dof_index_ls     = dof_index_ls;
  reinit_parameters.dof_index_normal = dof_index_normal;
  reinit_parameters.quad_index       = quad_index;
  reinit_parameters.do_iteration     = true;

  reinit_parameters.time.time_step_scheme     = TimeSteppingParameters::Scheme::bdf_2;
  reinit_parameters.time.start_time           = 0.0;
  reinit_parameters.time.end_time             = dt;
  reinit_parameters.time.time_step_size_start = dt;
  reinit_parameters.time.time_stepping_cfl    = 1.0;
  reinit_parameters.time.time_stepping_coef2  = 10;
  reinit_parameters.time.time_step_tolerance  = 1.e-2;
  reinit_parameters.time.time_step_size_max   = dt;
  reinit_parameters.time.time_step_size_min   = dt;

  LevelSetOKZSolverReinitialization<dim> reinit(normal_vector_field,
                                                cell_diameters,
                                                epsilon_used,
                                                minimal_edge_length,
                                                constraints,
                                                ls_solution_update,
                                                ls_solution,
                                                ls_system_rhs,
                                                pcout,
                                                preconditioner,
                                                last_concentration_range,
                                                reinit_parameters,
                                                first_reinit_step,
                                                matrix_free);

  // curvature operator
  LevelSetOKZSolverComputeCurvatureParameter parameters_curvature;
  parameters_curvature.dof_index_ls            = dof_index_ls;
  parameters_curvature.dof_index_curvature     = dof_index_curvature;
  parameters_curvature.dof_index_normal        = dof_index_normal;
  parameters_curvature.quad_index              = quad_index;
  parameters_curvature.epsilon                 = epsilon;
  parameters_curvature.approximate_projections = false;
  parameters_curvature.curvature_correction    = true;

  LevelSetOKZSolverComputeCurvature<dim> curvature_operator(cell_diameters,
                                                            normal_vector_field,
                                                            constraints_curvature,
                                                            constraints,
                                                            epsilon_used,
                                                            curvature_rhs,
                                                            parameters_curvature,
                                                            curvature_solution,
                                                            ls_solution,
                                                            matrix_free,
                                                            preconditioner,
                                                            projection_matrix,
                                                            ilu_projection_matrix);

  // perform reinitialization
  constraints.set_zero(ls_solution);
  reinit.reinitialize(dt, stab_steps, 0, [&normal_operator](const bool fast) {
    normal_operator.compute_normal(fast);
  });

  // compute normal vectors
  normal_operator.compute_normal(false);

  // compute curvature
  curvature_operator.compute_curvature();

  constraints.distribute(ls_solution);
  for (unsigned int i = 0; i < dim; ++i)
    constraints_normals.distribute(normal_vector_field.block(i));
  constraints_curvature.distribute(curvature_solution);
}



template <int dim>
void
test(const std::string &parameter_filename)
{
  const unsigned int n_global_refinements = 6;
  const unsigned int fe_degree            = 1;

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, 2, -1.0, +1.0);

  for (const auto &cell : tria.cell_iterators())
    for (const auto &face : cell->face_iterators())
      if ((face->at_boundary()))
        {
          if (face->center()[0] == -1.0 && face->center()[1] > 0.0)
            face->set_boundary_id(0);
          else if (face->center()[0] == +1.0 && face->center()[1] < 0.0)
            face->set_boundary_id(0);
          else if (face->center()[1] == -1.0 && face->center()[0] > 0.0)
            face->set_boundary_id(0);
          else if (face->center()[1] == +1.0 && face->center()[0] < 0.0)
            face->set_boundary_id(0);
          else
            face->set_boundary_id(1);
        }

  tria.refine_global(n_global_refinements - 1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));

  DoFHandler<dim> dof_handler_dim(tria);
  dof_handler_dim.distribute_dofs(FESystem<dim>(FE_Q<dim>(fe_degree), dim));

  MappingQ1<dim> mapping;

  AffineConstraints<double> constraints, constraints_normals, hanging_node_constraints,
    constraints_curvature, constraints_force;

  /** not needed: why?
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ConstantFunction<dim>(-1.0), constraints);

  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ConstantFunction<dim>(0.0), constraints_normals);

  VectorTools::interpolate_boundary_values(mapping,
                                           dof_handler,
                                           0,
                                           Functions::ConstantFunction<dim>(0.0),
                                           constraints_curvature);
   */

  constraints.close();
  constraints_curvature.close();
  constraints_normals.close();
  constraints_force.close();
  hanging_node_constraints.close();

  QGauss<1> quad(fe_degree + 1);

  MatrixFree<dim, double> matrix_free;

  const std::vector<const DoFHandler<dim> *> dof_handlers{&dof_handler,
                                                          &dof_handler,
                                                          &dof_handler,
                                                          &dof_handler_dim};

  const std::vector<const AffineConstraints<double> *> all_constraints{
    &constraints, &constraints_normals, &constraints_curvature, &constraints_force};

  const std::vector<Quadrature<1>> quadratures{quad};

  matrix_free.reinit(mapping, dof_handlers, all_constraints, quadratures);

  // vectors
  BlockVectorType normal_vector_field(dim);
  VectorType      ls_solution;
  VectorType      curvature_solution;
  BlockVectorType force_vector_regularized(dim);
  VectorType      force_vector_sharp_interface;

  matrix_free.initialize_dof_vector(ls_solution, dof_index_ls);
  matrix_free.initialize_dof_vector(curvature_solution, dof_index_curvature);
  matrix_free.initialize_dof_vector(force_vector_sharp_interface, dof_index_force);

  for (unsigned int i = 0; i < dim; ++i)
    {
      matrix_free.initialize_dof_vector(normal_vector_field.block(i), dof_index_normal);
      matrix_free.initialize_dof_vector(force_vector_regularized.block(i),
                                        dof_index_normal);
    }

  // initialize level-set
  VectorTools::interpolate(mapping, dof_handler, InitialValuesLS<dim>(), ls_solution);

  // compute level-set, normal-vector, and curvature field
  compute_ls_normal_curvature(matrix_free,
                              constraints,
                              constraints_normals,
                              hanging_node_constraints,
                              constraints_curvature,
                              normal_vector_field,
                              ls_solution,
                              curvature_solution);

  //  compute force vector with a regularized approach
  compute_force_vector_regularized(matrix_free,
                                   dof_index_ls,
                                   dof_index_curvature,
                                   dof_index_normal,
                                   quad_index,
                                   1.0, /*TODO*/
                                   ls_solution,
                                   curvature_solution,
                                   force_vector_regularized);

  std::cout << force_vector_regularized.l2_norm() << std::endl;

  //  compute force vector with a sharp-interface approach
  Triangulation<dim - 1, dim> surface_mesh;
  create_surface_mesh(surface_mesh);

  FESystem<dim - 1, dim>   surface_fe_dim(FE_Q<dim - 1, dim>(fe_degree), dim);
  DoFHandler<dim - 1, dim> surface_dof_handler_dim(surface_mesh);
  surface_dof_handler_dim.distribute_dofs(surface_fe_dim);

  VectorType euler_vector(surface_dof_handler_dim.n_dofs());
  VectorTools::get_position_vector(surface_dof_handler_dim,
                                   euler_vector,
                                   MappingQGeneric<dim - 1, dim>(4));
  MappingFEField<dim - 1, dim, VectorType> surface_mapping(surface_dof_handler_dim,
                                                           euler_vector);

  FE_Q<dim - 1, dim> surface_fe(fe_degree);
  QGauss<dim - 1>    surface_quad(fe_degree + 1);
  compute_force_vector_sharp_interface<dim>(surface_mesh,
                                            surface_mapping,
                                            surface_fe,
                                            surface_quad,
                                            mapping,
                                            dof_handler,
                                            dof_handler_dim,
                                            1.0, /*TODO*/
                                            normal_vector_field,
                                            curvature_solution,
                                            force_vector_sharp_interface);

  std::cout << force_vector_sharp_interface.l2_norm() << std::endl;
  force_vector_sharp_interface = 0.0;

  // TODO: write computed vectors to Paraview
  {
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;

    DataOut<dim> data_out;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(dof_handler, ls_solution, "ls");
    data_out.add_data_vector(dof_handler, curvature_solution, "curvature");

    for (unsigned int i = 0; i < dim; ++i)
      data_out.add_data_vector(dof_handler,
                               normal_vector_field.block(i),
                               "normal_" + std::to_string(i));

    for (unsigned int i = 0; i < dim; ++i)
      data_out.add_data_vector(dof_handler,
                               force_vector_regularized.block(i),
                               "force_re_" + std::to_string(i));

    data_out.add_data_vector(dof_handler_dim, force_vector_sharp_interface, "force_si_");

    data_out.build_patches(mapping, fe_degree + 1);
    data_out.write_vtu_with_pvtu_record("./",
                                        "output-sharp_interfaces_01/sharp_interface_01",
                                        0,
                                        MPI_COMM_WORLD);
  }

  {
    FlowParameters parameters;

    ParameterHandler prm;
    parameters.declare_parameters(prm);
    parameters.check_for_file(parameter_filename, prm);
    parameters.parse_parameters(parameter_filename, prm);

    TimeStepping time_stepping(parameters);

    std::set<types::boundary_id>                                 symmetry;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> fluid_type;
    fluid_type[0] = std::make_shared<Functions::ConstantFunction<dim>>(-1.0);


    VectorType velocity_solution, velocity_solution_old, velocity_solution_old_old;

    LevelSetSolver<dim> level_set_solver(tria,
                                         InitialValuesLS<dim>(),
                                         parameters,
                                         time_stepping,
                                         velocity_solution,
                                         velocity_solution_old,
                                         velocity_solution_old_old,
                                         fluid_type,
                                         symmetry);

    level_set_solver.initialize_dof_vector(velocity_solution,
                                           LevelSetSolver<dim>::dof_index_velocity);

    VectorTools::interpolate(mapping,
                             level_set_solver.get_dof_handler_dim(),
                             VelocityFunction<dim>(),
                             velocity_solution);

    velocity_solution_old     = velocity_solution;
    velocity_solution_old_old = velocity_solution;

    const auto post_process = [&](const unsigned int i) {
      // regularized
      VectorType force_vector_regularized;

      level_set_solver.get_matrix_free().initialize_dof_vector(
        force_vector_regularized, LevelSetSolver<dim>::dof_index_velocity);

      compute_force_vector_regularized(level_set_solver.get_matrix_free(),
                                       LevelSetSolver<dim>::dof_index_ls,
                                       LevelSetSolver<dim>::dof_index_curvature,
                                       LevelSetSolver<dim>::dof_index_velocity,
                                       LevelSetSolver<dim>::quad_index,
                                       1.0, /*TODO*/
                                       level_set_solver.get_level_set_vector(),
                                       level_set_solver.get_curvature_vector(),
                                       force_vector_regularized);

      // sharp interface
      VectorType force_vector_sharp_interface;
      level_set_solver.initialize_dof_vector(force_vector_sharp_interface,
                                             LevelSetSolver<dim>::dof_index_velocity);

      compute_force_vector_sharp_interface<dim>(surface_mesh,
                                                surface_mapping,
                                                surface_fe,
                                                surface_quad,
                                                mapping,
                                                level_set_solver.get_dof_handler(),
                                                level_set_solver.get_dof_handler_dim(),
                                                1.0, /*TODO*/
                                                level_set_solver.get_normal_vector(),
                                                level_set_solver.get_curvature_vector(),
                                                force_vector_sharp_interface);

      {
        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;

        DataOut<dim> data_out;
        data_out.set_flags(flags);
        data_out.add_data_vector(level_set_solver.get_dof_handler(),
                                 level_set_solver.get_level_set_vector(),
                                 "ls");
        data_out.add_data_vector(level_set_solver.get_dof_handler(),
                                 level_set_solver.get_curvature_vector(),
                                 "curvature");

        for (unsigned int i = 0; i < dim; ++i)
          data_out.add_data_vector(level_set_solver.get_dof_handler(),
                                   level_set_solver.get_normal_vector().block(i),
                                   "normal_" + std::to_string(i));

        data_out.add_data_vector(level_set_solver.get_dof_handler_dim(),
                                 force_vector_regularized,
                                 "force_re_");

        data_out.add_data_vector(level_set_solver.get_dof_handler_dim(),
                                 force_vector_sharp_interface,
                                 "force_si_");

        data_out.add_data_vector(level_set_solver.get_dof_handler_dim(),
                                 velocity_solution,
                                 "velocity");

        data_out.build_patches(mapping, fe_degree + 1);
        data_out.write_vtu_with_pvtu_record("./",
                                            parameters.output_filename + "_temp",
                                            i,
                                            MPI_COMM_WORLD);
      }

      {
        DataOutBase::VtkFlags flags;

        DataOut<dim - 1, DoFHandler<dim - 1, dim>> data_out;
        data_out.set_flags(flags);
        data_out.attach_dof_handler(surface_dof_handler_dim);

        data_out.build_patches(
          surface_mapping,
          3,
          DataOut<dim - 1,
                  DoFHandler<dim - 1, dim>>::CurvedCellRegion::curved_inner_cells);
        data_out.write_vtu_with_pvtu_record("./",
                                            parameters.output_filename + "_surf",
                                            i,
                                            MPI_COMM_WORLD);
      }
    };

    post_process(0);

    for (unsigned int i = 1; i <= 10; ++i)
      {
        level_set_solver.solve();

        VectorTools::update_position_vector(time_stepping.step_size(),
                                            level_set_solver.get_dof_handler_dim(),
                                            mapping,
                                            velocity_solution,
                                            surface_dof_handler_dim,
                                            surface_mapping,
                                            euler_vector);

        post_process(i);

        time_stepping.next();
      }
  }
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertDimension(argc, 2)

    test<2>(argv[1]);
}
