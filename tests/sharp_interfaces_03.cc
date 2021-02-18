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

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_point_evaluation.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q_generic.h>

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

using VectorType = LinearAlgebra::distributed::Vector<double>;

template <int dim>
void
test()
{
  // surface mesh
  const unsigned int spacedim       = dim + 1;
  const unsigned int fe_degree      = 3;
  const unsigned int mapping_degree = fe_degree;
  const unsigned int n_refinements  = 5;

  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_sphere(tria, Point<spacedim>(), 0.5);
  tria.refine_global(n_refinements);

  FE_Q<dim, spacedim>       fe(fe_degree);
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  FESystem<dim, spacedim>   fe_dim(fe, spacedim);
  DoFHandler<dim, spacedim> dof_handler_dim(tria);
  dof_handler_dim.distribute_dofs(fe_dim);

  Vector<double> euler_vector(dof_handler_dim.n_dofs());
  VectorTools::get_position_vector(dof_handler_dim,
                                   euler_vector,
                                   MappingQGeneric<dim, spacedim>(mapping_degree));
  MappingFEField<dim, spacedim> mapping(dof_handler_dim, euler_vector);

  // background mesh
  const unsigned int background_n_global_refinements = 5;
  const unsigned int background_fe_degree            = 2;

  Triangulation<spacedim> background_tria;
  GridGenerator::hyper_cube(background_tria, -1.0, +1.0);
  background_tria.refine_global(background_n_global_refinements);

  MappingQ1<spacedim>  background_mapping;
  FE_Q<spacedim>       background_fe(background_fe_degree);
  DoFHandler<spacedim> background_dof_handler(background_tria);
  background_dof_handler.distribute_dofs(background_fe);

  // determine if quadrature points of background mesh are within codim-1 mesh
  VectorType quad_within_surface_mesh(background_dof_handler.n_dofs());
  GridTools::within(mapping,
                    dof_handler,
                    background_mapping,
                    background_dof_handler,
                    quad_within_surface_mesh);

  // print result
  {
    DataOutBase::VtkFlags flags;

    DataOut<dim, DoFHandler<dim, spacedim>> data_out;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler);

    data_out.build_patches(
      mapping,
      fe_degree + 1,
      DataOut<dim, DoFHandler<dim, spacedim>>::CurvedCellRegion::curved_inner_cells);
    data_out.write_vtu_with_pvtu_record("./",
                                        "output-sharp_interfaces_03/data_surface",
                                        0,
                                        MPI_COMM_WORLD);
  }
  {
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;

    DataOut<spacedim> data_out;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(background_dof_handler);
    data_out.add_data_vector(background_dof_handler,
                             quad_within_surface_mesh,
                             "quad_point_is_within_surface_mesh");

    data_out.build_patches(background_mapping, background_fe_degree + 1);
    data_out.write_vtu_with_pvtu_record("./",
                                        "output-sharp_interfaces_03/data_background",
                                        0,
                                        MPI_COMM_WORLD);
  }
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test<1>();
}
