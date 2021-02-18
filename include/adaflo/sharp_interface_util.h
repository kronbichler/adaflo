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

#ifndef __adaflo_block_sharp_inteface_util_h
#define __adaflo_block_sharp_inteface_util_h

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi_consensus_algorithms.h>
#include <deal.II/base/mpi_consensus_algorithms.templates.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_point_evaluation.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/vector_tools.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <adaflo/level_set_okz_advance_concentration.h>
#include <adaflo/level_set_okz_compute_curvature.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/level_set_okz_preconditioner.h>
#include <adaflo/level_set_okz_reinitialization.h>
#include <adaflo/util.h>

#include <filesystem>

namespace dealii
{
  namespace VectorTools
  {
    template <int dim, int spacedim, typename VectorType>
    void
    get_position_vector(const DoFHandler<dim, spacedim> &dof_handler_dim,
                        VectorType &                     euler_coordinates_vector,
                        const Mapping<dim, spacedim> &   mapping)
    {
      FEValues<dim, spacedim> fe_eval(
        mapping,
        dof_handler_dim.get_fe(),
        Quadrature<dim>(dof_handler_dim.get_fe().get_unit_support_points()),
        update_quadrature_points);

      Vector<double> temp;

      for (const auto &cell : dof_handler_dim.active_cell_iterators())
        {
          fe_eval.reinit(cell);

          temp.reinit(fe_eval.dofs_per_cell);

          for (const auto q : fe_eval.quadrature_point_indices())
            {
              const auto point = fe_eval.quadrature_point(q);

              const unsigned int comp =
                dof_handler_dim.get_fe().system_to_component_index(q).first;

              temp[q] = point[comp];
            }

          cell->set_dof_values(temp, euler_coordinates_vector);
        }
    }

    template <int dim, int spacedim, typename VectorType>
    void
    update_position_vector(const double                          dt,
                           const DoFHandler<spacedim, spacedim> &background_dofhandler,
                           const Mapping<spacedim, spacedim> &   background_mapping,
                           const VectorType &                    velocity_vector,
                           const DoFHandler<dim, spacedim> &     euler_dofhandler,
                           const Mapping<dim, spacedim> &        euler_mapping,
                           VectorType &                          euler_coordinates_vector)
    {
      FEValues<dim, spacedim> fe_eval(
        euler_mapping,
        euler_dofhandler.get_fe(),
        Quadrature<dim>(euler_dofhandler.get_fe().get_unit_support_points()),
        update_quadrature_points);

      Vector<double>                       temp;
      std::vector<types::global_dof_index> temp_dof_indices;

      auto euler_coordinates_vector_temp = euler_coordinates_vector;
      auto euler_coordinates_vector_bool = euler_coordinates_vector;
      euler_coordinates_vector_bool      = 0.0;

      const std::vector<bool>                    marked_vertices;
      const GridTools::Cache<spacedim, spacedim> cache(
        background_dofhandler.get_triangulation(), background_mapping);
      const double tolerance = 1e-10;
      auto         cell_hint = background_dofhandler.get_triangulation().begin_active();

      FEPointEvaluation<spacedim, spacedim> phi_velocity(background_mapping,
                                                         background_dofhandler.get_fe());

      for (const auto &cell : euler_dofhandler.active_cell_iterators())
        {
          fe_eval.reinit(cell);

          temp.reinit(fe_eval.dofs_per_cell);
          temp_dof_indices.resize(fe_eval.dofs_per_cell);

          cell->get_dof_indices(temp_dof_indices);
          cell->get_dof_values(euler_coordinates_vector, temp);

          for (const auto q : fe_eval.quadrature_point_indices())
            {
              // if (euler_coordinates_vector_bool[temp_dof_indices[q]] == 1.0)
              //  continue;

              euler_coordinates_vector_bool[temp_dof_indices[q]] = 1.0;

              const auto cell_and_reference_coordinate =
                GridTools::find_active_cell_around_point(cache,
                                                         fe_eval.quadrature_point(q),
                                                         cell_hint,
                                                         marked_vertices,
                                                         tolerance);

              std::vector<double> buffer(
                background_dofhandler.get_fe().n_dofs_per_cell());

              typename DoFHandler<spacedim, spacedim>::active_cell_iterator
                background_cell(&background_dofhandler.get_triangulation(),
                                cell_and_reference_coordinate.first->level(),
                                cell_and_reference_coordinate.first->index(),
                                &background_dofhandler);
              background_cell->get_dof_values(velocity_vector,
                                              buffer.begin(),
                                              buffer.end());

              const ArrayView<const Point<spacedim>> unit_points(
                &cell_and_reference_coordinate.second, 1);

              phi_velocity.evaluate(cell_and_reference_coordinate.first,
                                    unit_points,
                                    buffer,
                                    EvaluationFlags::values);

              const auto velocity = phi_velocity.get_value(0);

              const unsigned int comp =
                euler_dofhandler.get_fe().system_to_component_index(q).first;

              // temp[q] += dt * velocity[comp];
              temp[q] = fe_eval.quadrature_point(q)[comp] + dt * velocity[comp];
            }

          cell->set_dof_values(temp, euler_coordinates_vector_temp);
        }

      euler_coordinates_vector = euler_coordinates_vector_temp;
    }
  } // namespace VectorTools

  namespace GridTools
  {
    template <int dim, int spacedim>
    void
    construct_polygon(
      const Mapping<dim, spacedim> &   mapping,
      const DoFHandler<dim, spacedim> &dof_handler,
      boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double>> &poly)
    {
      typedef boost::geometry::model::d2::point_xy<double> point_type;
      typedef boost::geometry::model::polygon<
        boost::geometry::model::d2::point_xy<double>>
        polygon_type;

      std::vector<boost::geometry::model::d2::point_xy<double>> points;
      {
        FEValues<dim, spacedim> fe_eval(mapping,
                                        dof_handler.get_fe(),
                                        QGauss<dim>(dof_handler.get_fe().degree + 1),
                                        update_quadrature_points);

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            fe_eval.reinit(cell);

            for (const auto q : fe_eval.quadrature_point_indices())
              {
                const auto point = fe_eval.quadrature_point(q);
                points.emplace_back(point[0], point[1]);
              }
          }
      }

      points.push_back(points.front());

      boost::geometry::assign_points(poly, points);
    }

    template <int dim>
    double
    within(
      const boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double>>
        &               polygon,
      const Point<dim> &point)
    {
      boost::geometry::model::d2::point_xy<double> p(point[0], point[1]);
      return boost::geometry::within(p, polygon);
    }

    template <int dim>
    VectorizedArray<double>
    within(
      const boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double>>
        &                                        polygon,
      const Point<dim, VectorizedArray<double>> &points)
    {
      VectorizedArray<double> result;

      for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
        {
          boost::geometry::model::d2::point_xy<double> p(points[0][v], points[1][v]);
          result[v] = boost::geometry::within(p, polygon);
        }

      return result;
    }

    template <int dim, int spacedim, typename VectorType>
    void
    within(const MappingFEField<dim, spacedim> &mapping,
           const DoFHandler<dim, spacedim> &    dof_handler,
           const Mapping<spacedim> &            background_mapping,
           const DoFHandler<spacedim> &         background_dof_handler,
           VectorType &                         force_vector_sharp_interface)
    {
      typedef boost::geometry::model::d2::point_xy<double> point_type;
      typedef boost::geometry::model::polygon<
        boost::geometry::model::d2::point_xy<double>>
        polygon_type;

      polygon_type poly;

      construct_polygon(mapping, dof_handler, poly);

      {
        FEValues<spacedim> fe_eval(
          background_mapping,
          background_dof_handler.get_fe(),
          background_dof_handler.get_fe().get_unit_support_points(),
          update_quadrature_points);

        for (const auto &cell : background_dof_handler.active_cell_iterators())
          {
            fe_eval.reinit(cell);

            Vector<double> vec(fe_eval.n_quadrature_points);

            for (const auto q : fe_eval.quadrature_point_indices())
              {
                vec[q] = static_cast<double>(within(poly, fe_eval.quadrature_point(q)));
              }

            cell->set_dof_values(vec, force_vector_sharp_interface);
          }
      }
    }
  } // namespace GridTools

  namespace GridGenerator
  {
    template <int dim>
    class MarchingCubeAlgorithm
    {
    public:
      MarchingCubeAlgorithm(const Mapping<dim, dim> &      mapping,
                            const FiniteElement<dim, dim> &fe,
                            const unsigned int             n_subdivisions)
        : n_subdivisions(n_subdivisions)
        , fe_values(mapping,
                    fe,
                    create_qudrature_rule(n_subdivisions),
                    update_values | update_quadrature_points)
        , ls_values(fe_values.n_quadrature_points)
      {
        AssertDimension(dim, 2);
      }

      template <typename CellType, typename VectorType>
      void
      process_cell(const CellType &                  cell,
                   const VectorType &                ls_vector,
                   std::vector<Point<dim>> &         vertices,
                   std::vector<::CellData<dim - 1>> &cells)
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(ls_vector, ls_values);

        for (unsigned int j = 0; j < n_subdivisions; ++j)
          for (unsigned int i = 0; i < n_subdivisions; ++i)
            {
              std::vector<unsigned int> mask{(n_subdivisions + 1) * (j + 0) + (i + 0),
                                             (n_subdivisions + 1) * (j + 0) + (i + 1),
                                             (n_subdivisions + 1) * (j + 1) + (i + 1),
                                             (n_subdivisions + 1) * (j + 1) + (i + 0),
                                             (n_subdivisions + 1) * (n_subdivisions + 1) +
                                               (n_subdivisions * j + i)};

              process_sub_cell(
                ls_values, fe_values.get_quadrature_points(), mask, vertices, cells);
            }
      }

    private:
      static Quadrature<dim>
      create_qudrature_rule(const unsigned int n_subdivisions)
      {
        /** Example: n_subdivisions = 2
         *
         *  x,y in [0,1]x[0,1]
         *
         *      ^
         *    y |
         *
         *     (6)   (7)    (8)
         *      +_____+_____+
         *      | (11) (12) |
         *      |  * (4) *  |
         *   (3)+     +     +(5)
         *      | (9)  (10) |
         *      |  *     *  |
         *      +_____+_____+ --> x
         *      (0)   (1)  (2)
         */

        std::vector<Point<dim>> quadrature_points;

        for (unsigned int j = 0; j <= n_subdivisions; ++j)
          for (unsigned int i = 0; i <= n_subdivisions; ++i)
            quadrature_points.emplace_back(1.0 / n_subdivisions * i,
                                           1.0 / n_subdivisions * j);

        for (unsigned int j = 0; j < n_subdivisions; ++j)
          for (unsigned int i = 0; i < n_subdivisions; ++i)
            quadrature_points.emplace_back(1.0 / n_subdivisions * (i + 0.5),
                                           1.0 / n_subdivisions * (j + 0.5));

        return {quadrature_points};
      }

      static void
      process_sub_cell(const std::vector<double> &       ls_values,
                       const std::vector<Point<dim>> &   points,
                       const std::vector<unsigned int>   mask,
                       std::vector<Point<dim>> &         vertices,
                       std::vector<::CellData<dim - 1>> &cells)
      {
        unsigned int c = 0;

        for (unsigned int i = 0, scale = 1; i < 4; ++i, scale *= 2)
          c += (ls_values[mask[i]] > 0) * scale;

        if (c == 0 || c == 15)
          return; // nothing to do since the level set function is constant within the
                  // sub_cell

        const auto process_points = [&](const auto &lines) {
          const double w0 = std::abs(ls_values[mask[lines[0]]]);
          const double w1 = std::abs(ls_values[mask[lines[1]]]);

          return points[mask[lines[0]]] * (w1 / (w0 + w1)) +
                 points[mask[lines[1]]] * (w0 / (w0 + w1));
        };

        const auto process_lines = [&](const auto &lines) {
          std::array<std::array<unsigned int, 2>, 4> table{
            {{{0, 3}}, {{1, 2}}, {{0, 1}}, {{3, 2}}}};

          const auto p0 = process_points(table[lines[0]]);
          const auto p1 = process_points(table[lines[1]]);

          cells.resize(cells.size() + 1);
          cells.back().vertices[0] = vertices.size();
          cells.back().vertices[1] = vertices.size() + 1;

          vertices.emplace_back(p0);
          vertices.emplace_back(p1);
        };

        // Check if the isoline for level set values larger than zero is the element's
        // diagonal and level set values on both sides from the diagonal are smaller than
        // zero. In this case, the level set would be a "hat"-function which does not make
        // sense.
        if (c == 5 || c == 10)
          {
            Assert(false, ExcNotImplemented());
            return;
          }

        static const unsigned int X = -1;

        std::array<std::array<unsigned int, 2>, 16> table{{
          {{X, X}},
          {{0, 2}},
          {{1, 2}},
          {{0, 1}}, //  c=0-3
          {{1, 3}},
          {{X, X}},
          {{2, 3}},
          {{0, 3}}, //  c=4-7
          {{0, 3}},
          {{2, 3}},
          {{X, X}},
          {{1, 3}}, //  c=8-11
          {{0, 1}},
          {{2, 1}},
          {{0, 2}},
          {{X, X}} //   c=12-15
        }};

        process_lines(table[c]);
      }

      const unsigned int  n_subdivisions;
      FEValues<dim>       fe_values;
      std::vector<double> ls_values;
    };
  } // namespace GridGenerator

} // namespace dealii



template <int spacedim>
std::tuple<std::vector<std::pair<int, int>>,
           std::vector<unsigned int>,
           std::vector<Tensor<1, spacedim, double>>,
           std::vector<Point<spacedim>>>
collect_integration_points(
  const Triangulation<spacedim, spacedim> &       tria,
  const Mapping<spacedim, spacedim> &             mapping,
  const std::vector<Point<spacedim>> &            integration_points,
  const std::vector<Tensor<1, spacedim, double>> &integration_values)
{
  std::vector<std::pair<Point<spacedim>, Tensor<1, spacedim, double>>>
    locally_owned_surface_points;

  for (unsigned int i = 0; i < integration_points.size(); ++i)
    locally_owned_surface_points.emplace_back(integration_points[i],
                                              integration_values[i]);

  std::vector<
    std::tuple<Point<spacedim>, Tensor<1, spacedim, double>, std::pair<int, int>>>
    info;

  const std::vector<bool>                    marked_vertices;
  const GridTools::Cache<spacedim, spacedim> cache(tria, mapping);
  const double                               tolerance = 1e-10;
  auto                                       cell_hint = tria.begin_active();

  for (const auto &point_and_weight : locally_owned_surface_points)
    {
      try
        {
          const auto first_cell = GridTools::find_active_cell_around_point(
            cache, point_and_weight.first, cell_hint, marked_vertices, tolerance);

          cell_hint = first_cell.first;

          const auto active_cells_around_point =
            GridTools::find_all_active_cells_around_point(
              mapping, tria, point_and_weight.first, tolerance, first_cell);

          for (const auto &cell_and_reference_coordinate : active_cells_around_point)
            info.emplace_back(
              cell_and_reference_coordinate.second,
              point_and_weight.second,
              std::pair<int, int>(cell_and_reference_coordinate.first->level(),
                                  cell_and_reference_coordinate.first->index()));
        }
      catch (...)
        {}
    }

  // step 4: compress data structures
  std::sort(info.begin(), info.end(), [](const auto &a, const auto &b) {
    return std::get<2>(a) < std::get<2>(b);
  });

  std::vector<std::pair<int, int>>         cells;
  std::vector<unsigned int>                ptrs;
  std::vector<Tensor<1, spacedim, double>> weights;
  std::vector<Point<spacedim>>             points;

  std::pair<int, int> dummy{-1, -1};

  for (const auto &i : info)
    {
      if (dummy != std::get<2>(i))
        {
          dummy = std::get<2>(i);
          cells.push_back(std::get<2>(i));
          ptrs.push_back(weights.size());
        }
      weights.push_back(std::get<1>(i));
      points.push_back(std::get<0>(i));
    }
  ptrs.push_back(weights.size());

  return {cells, ptrs, weights, points};
}



template <int dim, int spacedim, typename VectorType>
void
compute_force_vector_sharp_interface(
  const Mapping<dim, spacedim> &   surface_mapping,
  const DoFHandler<dim, spacedim> &surface_dofhandler,
  const DoFHandler<dim, spacedim> &surface_dofhandler_dim,
  const Quadrature<dim> &          surface_quadrature,
  const Mapping<spacedim> &        mapping,
  const DoFHandler<spacedim> &     dof_handler,
  const double                     surface_tension,
  const VectorType &               normal_vector,
  const VectorType &               curvature_vector,
  VectorType &                     force_vector)
{
  std::vector<Point<spacedim>>             integration_points;
  std::vector<Tensor<1, spacedim, double>> integration_values;

  {
    FEValues<dim, spacedim> fe_eval(surface_mapping,
                                    surface_dofhandler.get_fe(),
                                    surface_quadrature,
                                    update_values | update_quadrature_points |
                                      update_JxW_values);
    FEValues<dim, spacedim> fe_eval_dim(surface_mapping,
                                        surface_dofhandler_dim.get_fe(),
                                        surface_quadrature,
                                        update_values);

    const auto &tria_surface = surface_dofhandler.get_triangulation();

    for (const auto &cell : tria_surface.active_cell_iterators())
      {
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell(&tria_surface,
                                                                     cell->level(),
                                                                     cell->index(),
                                                                     &surface_dofhandler);
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
          &tria_surface, cell->level(), cell->index(), &surface_dofhandler_dim);

        fe_eval.reinit(dof_cell);
        fe_eval_dim.reinit(dof_cell_dim);

        std::vector<double>         curvature_values(fe_eval.dofs_per_cell);
        std::vector<Vector<double>> normal_values(fe_eval.dofs_per_cell,
                                                  Vector<double>(spacedim));

        fe_eval.get_function_values(curvature_vector, curvature_values);
        fe_eval_dim.get_function_values(normal_vector, normal_values);

        for (const auto q : fe_eval_dim.quadrature_point_indices())
          {
            Tensor<1, spacedim, double> result;
            for (unsigned int i = 0; i < spacedim; ++i)
              result[i] = -curvature_values[q] * normal_values[q][i] * fe_eval.JxW(q) *
                          surface_tension;

            integration_points.push_back(fe_eval.quadrature_point(q));
            integration_values.push_back(result);
          }
      }
  }

  const auto [cells, ptrs, weights, points] = collect_integration_points(
    dof_handler.get_triangulation(), mapping, integration_points, integration_values);

  AffineConstraints<double> constraints; // TODO: use the right ones

  FEPointEvaluation<spacedim, spacedim> phi_normal_force(mapping, dof_handler.get_fe());

  std::vector<double>                  buffer;
  std::vector<types::global_dof_index> local_dof_indices;

  for (unsigned int i = 0; i < cells.size(); ++i)
    {
      typename DoFHandler<spacedim>::active_cell_iterator cell(
        &dof_handler.get_triangulation(), cells[i].first, cells[i].second, &dof_handler);

      const unsigned int n_dofs_per_cell = cell->get_fe().n_dofs_per_cell();

      local_dof_indices.resize(n_dofs_per_cell);
      buffer.resize(n_dofs_per_cell);

      cell->get_dof_indices(local_dof_indices);

      const unsigned int n_points = ptrs[i + 1] - ptrs[i];

      const ArrayView<const Point<spacedim>> unit_points(points.data() + ptrs[i],
                                                         n_points);
      const ArrayView<const Tensor<1, spacedim, double>> JxW(weights.data() + ptrs[i],
                                                             n_points);

      for (unsigned int q = 0; q < n_points; ++q)
        phi_normal_force.submit_value(JxW[q], q);

      phi_normal_force.integrate(cell, unit_points, buffer, EvaluationFlags::values);

      constraints.distribute_local_to_global(buffer, local_dof_indices, force_vector);
    }
}



template <int dim, int spacedim, typename VectorType>
void
compute_normal(const Mapping<dim, spacedim> &   mapping,
               const DoFHandler<dim, spacedim> &dof_handler_dim,
               VectorType &                     normal_vector)
{
  FEValues<dim, spacedim> fe_eval_dim(mapping,
                                      dof_handler_dim.get_fe(),
                                      dof_handler_dim.get_fe().get_unit_support_points(),
                                      update_normal_vectors | update_gradients);

  Vector<double> normal_temp;

  for (const auto &cell : dof_handler_dim.active_cell_iterators())
    {
      fe_eval_dim.reinit(cell);

      normal_temp.reinit(fe_eval_dim.dofs_per_cell);
      normal_temp = 0.0;

      for (const auto q : fe_eval_dim.quadrature_point_indices())
        {
          const auto normal = fe_eval_dim.normal_vector(q);

          const unsigned int comp =
            dof_handler_dim.get_fe().system_to_component_index(q).first;

          normal_temp[q] = normal[comp];
        }

      cell->set_dof_values(normal_temp, normal_vector);
    }
}



template <int dim, int spacedim, typename VectorType>
void
compute_curvature(const Mapping<dim, spacedim> &   mapping,
                  const DoFHandler<dim, spacedim> &dof_handler_dim,
                  const DoFHandler<dim, spacedim> &dof_handler,
                  const Quadrature<dim>            quadrature,
                  const VectorType &               normal_vector,
                  VectorType &                     curvature_vector)
{
  FEValues<dim, spacedim> fe_eval(mapping,
                                  dof_handler.get_fe(),
                                  quadrature,
                                  update_gradients);
  FEValues<dim, spacedim> fe_eval_dim(mapping,
                                      dof_handler_dim.get_fe(),
                                      quadrature,
                                      update_gradients);

  Vector<double> curvature_temp;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
        &dof_handler_dim.get_triangulation(),
        cell->level(),
        cell->index(),
        &dof_handler_dim);

      fe_eval.reinit(cell);
      fe_eval_dim.reinit(dof_cell_dim);

      curvature_temp.reinit(quadrature.size());

      std::vector<std::vector<Tensor<1, spacedim, double>>> normal_gradients(
        quadrature.size(), std::vector<Tensor<1, spacedim, double>>(spacedim));

      fe_eval_dim.get_function_gradients(normal_vector, normal_gradients);

      for (const auto q : fe_eval_dim.quadrature_point_indices())
        {
          double curvature = 0.0;

          for (unsigned c = 0; c < spacedim; ++c)
            curvature += normal_gradients[q][c][c];

          curvature_temp[q] = curvature;
        }

      cell->set_dof_values(curvature_temp, curvature_vector);
    }
}



template <int dim, int spacedim>
std::tuple<std::vector<std::pair<int, int>>,
           std::vector<unsigned int>,
           std::vector<double>,
           std::vector<Point<spacedim>>>
collect_evaluation_points(const Triangulation<dim, spacedim> &     surface_mesh,
                          const Mapping<dim, spacedim> &           surface_mapping,
                          const FiniteElement<dim, spacedim> &     surface_fe,
                          const Quadrature<dim> &                  surface_quad,
                          const Triangulation<spacedim, spacedim> &tria,
                          const Mapping<spacedim, spacedim> &      mapping)
{
  // step 1: determine quadrature points in real coordinate system and quadrature weight
  std::vector<std::pair<Point<spacedim>, double>> locally_owned_surface_points;

  FEValues<dim, spacedim> fe_eval(surface_mapping,
                                  surface_fe,
                                  surface_quad,
                                  update_quadrature_points | update_JxW_values);

  for (const auto &cell : surface_mesh.active_cell_iterators())
    {
      if (cell->is_locally_owned() == false)
        continue;

      fe_eval.reinit(cell);

      for (const auto q : fe_eval.quadrature_point_indices())
        locally_owned_surface_points.emplace_back(fe_eval.quadrature_point(q),
                                                  fe_eval.JxW(q));
    }

  // step 2 communicate (TODO)

  // step 3: convert quadrature points to a pair of cells and reference cell quadrature
  // point
  std::vector<std::tuple<Point<spacedim>, double, std::pair<int, int>>> info;

  const std::vector<bool>                    marked_vertices;
  const GridTools::Cache<spacedim, spacedim> cache(tria, mapping);
  const double                               tolerance = 1e-10;
  auto                                       cell_hint = tria.begin_active();

  for (const auto &point_and_weight : locally_owned_surface_points)
    {
      const auto cell_and_reference_coordinate = GridTools::find_active_cell_around_point(
        cache, point_and_weight.first, cell_hint, marked_vertices, tolerance);

      cell_hint = cell_and_reference_coordinate.first;

      info.emplace_back(
        cell_and_reference_coordinate.second,
        point_and_weight.second,
        std::pair<int, int>(cell_and_reference_coordinate.first->level(),
                            cell_and_reference_coordinate.first->index()));
    }

  // step 4: compress data structures
  std::sort(info.begin(), info.end(), [](const auto &a, const auto &b) {
    return std::get<2>(a) < std::get<2>(b);
  });

  std::vector<std::pair<int, int>> cells;
  std::vector<unsigned int>        ptrs;
  std::vector<double>              weights;
  std::vector<Point<spacedim>>     points;

  std::pair<int, int> dummy{-1, -1};

  for (const auto &i : info)
    {
      if (dummy != std::get<2>(i))
        {
          dummy = std::get<2>(i);
          cells.push_back(std::get<2>(i));
          ptrs.push_back(weights.size());
        }
      weights.push_back(std::get<1>(i));
      points.push_back(std::get<0>(i));
    }
  ptrs.push_back(weights.size());

  return {cells, ptrs, weights, points};
}



template <int dim, typename VectorType, typename BlockVectorType>
void
compute_force_vector_sharp_interface(const Triangulation<dim - 1, dim> &surface_mesh,
                                     const Mapping<dim - 1, dim> &      surface_mapping,
                                     const FiniteElement<dim - 1, dim> &surface_fe,
                                     const Quadrature<dim - 1> &        surface_quad,
                                     const Mapping<dim> &               mapping,
                                     const DoFHandler<dim> &            dof_handler,
                                     const DoFHandler<dim> &            dof_handler_dim,
                                     const BlockVectorType &normal_vector_field,
                                     const VectorType &     curvature_solution,
                                     VectorType &           force_vector)
{
  // step 1) collect all locally-relevant surface quadrature points (cell,
  // reference-cell position,
  //  quadrature weight)
  const auto [cells, ptrs, weights, points] =
    collect_evaluation_points(surface_mesh,
                              surface_mapping,
                              surface_fe,
                              surface_quad,
                              dof_handler.get_triangulation(),
                              mapping);

  // step 2) loop over all cells and evaluate curvature and normal in the cell-local
  // quadrature points
  //   and test with all test functions of the cell

  AffineConstraints<double> constraints; // TODO: use the right ones

  FEPointEvaluation<1, dim> phi_curvature(mapping, dof_handler.get_fe());

  FESystem<dim>               fe_dim(dof_handler.get_fe(), dim);
  FEPointEvaluation<dim, dim> phi_normal(mapping, fe_dim);
  FEPointEvaluation<dim, dim> phi_force(mapping, dof_handler_dim.get_fe());

  std::vector<double>                  buffer;
  std::vector<double>                  buffer_dim;
  std::vector<types::global_dof_index> local_dof_indices;

  for (unsigned int i = 0; i < cells.size(); ++i)
    {
      typename DoFHandler<dim>::active_cell_iterator cell = {
        &dof_handler.get_triangulation(), cells[i].first, cells[i].second, &dof_handler};

      typename DoFHandler<dim>::active_cell_iterator cell_dim = {
        &dof_handler.get_triangulation(),
        cells[i].first,
        cells[i].second,
        &dof_handler_dim};

      local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
      buffer.resize(cell->get_fe().n_dofs_per_cell());
      buffer_dim.resize(cell->get_fe().n_dofs_per_cell() * dim);

      cell->get_dof_indices(local_dof_indices);

      const unsigned int n_points = ptrs[i + 1] - ptrs[i];

      const ArrayView<const Point<dim>> unit_points(points.data() + ptrs[i], n_points);
      const ArrayView<const double>     JxW(weights.data() + ptrs[i], n_points);

      // gather curvature
      constraints.get_dof_values(curvature_solution,
                                 local_dof_indices.begin(),
                                 buffer.begin(),
                                 buffer.end());

      // evaluate curvature
      phi_curvature.evaluate(cell,
                             unit_points,
                             make_array_view(buffer),
                             EvaluationFlags::values);

      // gather normal
      for (int i = 0; i < dim; ++i)
        {
          constraints.get_dof_values(normal_vector_field.block(i),
                                     local_dof_indices.begin(),
                                     buffer.begin(),
                                     buffer.end());
          for (unsigned int c = 0; c < cell->get_fe().n_dofs_per_cell(); ++c)
            buffer_dim[fe_dim.component_to_system_index(i, c)] = buffer[c];
        }

      // evaluate normal
      phi_normal.evaluate(cell, unit_points, buffer_dim, EvaluationFlags::values);

      // quadrature loop
      for (unsigned int q = 0; q < n_points; ++q)
        {
          Assert(phi_normal.get_value(q).norm() > 0, ExcNotImplemented());
          const auto normal = phi_normal.get_value(q) / phi_normal.get_value(q).norm();
          phi_force.submit_value(normal * phi_curvature.get_value(q) * JxW[q], q);
        }

      buffer_dim.resize(dof_handler_dim.get_fe().n_dofs_per_cell());
      local_dof_indices.resize(dof_handler_dim.get_fe().n_dofs_per_cell());

      // integrate force
      phi_force.integrate(cell, unit_points, buffer_dim, EvaluationFlags::values);

      cell_dim->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(buffer_dim, local_dof_indices, force_vector);
    }
}



template <int dim, typename VectorType, typename BlockVectorType>
void
compute_force_vector_sharp_interface(const Quadrature<dim - 1> &surface_quad,
                                     const Mapping<dim> &       mapping,
                                     const DoFHandler<dim> &    dof_handler,
                                     const DoFHandler<dim> &    dof_handler_dim,
                                     const BlockVectorType &    normal_vector_field,
                                     const VectorType &         curvature_solution,
                                     const VectorType &         ls_vector,
                                     VectorType &               force_vector)
{
  const unsigned int                        n_subdivisions = 3;
  GridGenerator::MarchingCubeAlgorithm<dim> mc(mapping,
                                               dof_handler.get_fe(),
                                               n_subdivisions);

  AffineConstraints<double> constraints; // TODO: use the right ones

  FEPointEvaluation<1, dim> phi_curvature(mapping, dof_handler.get_fe());

  FESystem<dim>               fe_dim(dof_handler.get_fe(), dim);
  FEPointEvaluation<dim, dim> phi_normal(mapping, fe_dim);
  FEPointEvaluation<dim, dim> phi_force(mapping, dof_handler_dim.get_fe());

  std::vector<double>                  buffer;
  std::vector<double>                  buffer_dim;
  std::vector<types::global_dof_index> local_dof_indices;

  // loop over all cells
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // determine if cell is cut by the interface and if yes, determine the quadrature
      // point location and weight
      const auto [points, weights] =
        [&]() -> std::tuple<std::vector<Point<dim>>, std::vector<double>> {
        // determine points and cells of aux surface triangulation
        std::vector<Point<dim>>          surface_vertices;
        std::vector<::CellData<dim - 1>> surface_cells;

        // run square/cube marching algorithm
        mc.process_cell(cell, ls_vector, surface_vertices, surface_cells);

        if (surface_vertices.size() == 0)
          return {}; // cell is not cut by interface -> no quadrature points have the be
                     // determined

        std::vector<Point<dim>> points;
        std::vector<double>     weights;

        // create aux triangulation of subcells
        Triangulation<dim - 1, dim> surface_triangulation;
        surface_triangulation.create_triangulation(surface_vertices, surface_cells, {});

        FE_Nothing<dim - 1, dim> fe;
        FEValues<dim - 1, dim>   fe_eval(fe,
                                       surface_quad,
                                       update_quadrature_points | update_JxW_values);

        // loop over all cells ...
        for (const auto &sub_cell : surface_triangulation.active_cell_iterators())
          {
            fe_eval.reinit(sub_cell);

            // ... and collect quadrature points and weights
            for (const auto q : fe_eval.quadrature_point_indices())
              {
                points.emplace_back(
                  mapping.transform_real_to_unit_cell(cell, fe_eval.quadrature_point(q)));
                weights.emplace_back(fe_eval.JxW(q));
              }
          }
        return {points, weights};
      }();

      if (points.size() == 0)
        continue; // cell is not cut but the interface -> nothing to do

      // proceed as usual

      typename DoFHandler<dim>::active_cell_iterator cell_dim = {
        &dof_handler.get_triangulation(), cell->level(), cell->index(), &dof_handler_dim};

      local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
      buffer.resize(cell->get_fe().n_dofs_per_cell());
      buffer_dim.resize(cell->get_fe().n_dofs_per_cell() * dim);

      cell->get_dof_indices(local_dof_indices);

      const unsigned int n_points = points.size();

      const ArrayView<const Point<dim>> unit_points(points.data(), n_points);
      const ArrayView<const double>     JxW(weights.data(), n_points);

      // gather curvature
      constraints.get_dof_values(curvature_solution,
                                 local_dof_indices.begin(),
                                 buffer.begin(),
                                 buffer.end());

      // evaluate curvature
      phi_curvature.evaluate(cell,
                             unit_points,
                             make_array_view(buffer),
                             EvaluationFlags::values);

      // gather normal
      for (int i = 0; i < dim; ++i)
        {
          constraints.get_dof_values(normal_vector_field.block(i),
                                     local_dof_indices.begin(),
                                     buffer.begin(),
                                     buffer.end());
          for (unsigned int c = 0; c < cell->get_fe().n_dofs_per_cell(); ++c)
            buffer_dim[fe_dim.component_to_system_index(i, c)] = buffer[c];
        }

      // evaluate normal
      phi_normal.evaluate(cell, unit_points, buffer_dim, EvaluationFlags::values);

      // quadrature loop
      for (unsigned int q = 0; q < n_points; ++q)
        {
          Assert(phi_normal.get_value(q).norm() > 0, ExcNotImplemented());
          const auto normal = phi_normal.get_value(q) / phi_normal.get_value(q).norm();
          phi_force.submit_value(normal * phi_curvature.get_value(q) * JxW[q], q);
        }

      buffer_dim.resize(dof_handler_dim.get_fe().n_dofs_per_cell());
      local_dof_indices.resize(dof_handler_dim.get_fe().n_dofs_per_cell());

      // integrate force
      phi_force.integrate(cell, unit_points, buffer_dim, EvaluationFlags::values);

      cell_dim->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(buffer_dim, local_dof_indices, force_vector);
    }
}



template <int dim, typename VectorType1, typename VectorType2>
void
compute_force_vector_regularized(const MatrixFree<dim, double> &matrix_free,
                                 const VectorType1 &            ls_solution,
                                 const VectorType1 &            curvature_solution,
                                 VectorType2 &                  force_rhs,
                                 const unsigned int             dof_index_ls,
                                 const unsigned int             dof_index_curvature,
                                 const unsigned int             dof_index_normal,
                                 const unsigned int             quad_index)
{
  (void)matrix_free;
  (void)ls_solution;
  (void)curvature_solution;

  auto level_set_as_heaviside = ls_solution;
  level_set_as_heaviside.add(1.0);
  level_set_as_heaviside *= 0.5;

  const double surface_tension_coefficient = 1.0;

  matrix_free.template cell_loop<VectorType2, VectorType1>(
    [&](const auto &matrix_free,
        auto &      force_rhs,
        const auto &level_set_as_heaviside,
        auto        macro_cells) {
      FEEvaluation<dim, -1, 0, 1, double> level_set(matrix_free,
                                                    dof_index_ls,
                                                    quad_index);

      FEEvaluation<dim, -1, 0, 1, double> curvature(matrix_free,
                                                    dof_index_curvature,
                                                    quad_index);

      FEEvaluation<dim, -1, 0, dim, double> surface_tension(matrix_free,
                                                            dof_index_normal,
                                                            quad_index);

      for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
        {
          level_set.reinit(cell);
          level_set.read_dof_values_plain(level_set_as_heaviside);
          level_set.evaluate(false, true);

          surface_tension.reinit(cell);

          curvature.reinit(cell);
          curvature.read_dof_values_plain(curvature_solution);
          curvature.evaluate(true, false);

          for (unsigned int q_index = 0; q_index < surface_tension.n_q_points; ++q_index)
            {
              surface_tension.submit_value(surface_tension_coefficient *
                                             level_set.get_gradient(q_index) *
                                             curvature.get_value(q_index),
                                           q_index);
            }
          surface_tension.integrate_scatter(true, false, force_rhs);
        }
    },
    force_rhs,
    level_set_as_heaviside,
    true);
}



template <int dim, typename VectorType2>
void
compute_force_vector_regularized(
  const MatrixFree<dim, double> &                               matrix_free,
  const AlignedVector<Tensor<1, dim, VectorizedArray<double>>> &level_set_gradients,
  const AlignedVector<VectorizedArray<double>> &                curvature_values,
  VectorType2 &                                                 force_rhs,
  const unsigned int                                            dof_index_normal,
  const unsigned int                                            quad_index)
{
  const double surface_tension_coefficient = 1.0;

  int dummy;

  matrix_free.template cell_loop<VectorType2, int>(
    [&](const auto &matrix_free, auto &force_rhs, const auto &, auto macro_cells) {
      FEEvaluation<dim, -1, 0, dim, double> surface_tension(matrix_free,
                                                            dof_index_normal,
                                                            quad_index);

      for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
        {
          surface_tension.reinit(cell);

          for (unsigned int q_index = 0; q_index < surface_tension.n_q_points; ++q_index)
            {
              surface_tension.submit_value(
                surface_tension_coefficient *
                  level_set_gradients[surface_tension.n_q_points * cell + q_index] *
                  curvature_values[surface_tension.n_q_points * cell + q_index],
                q_index);
            }
          surface_tension.integrate_scatter(true, false, force_rhs);
        }
    },
    force_rhs,
    dummy,
    true);
}


namespace dealii
{
  template <typename MeshType>
  MPI_Comm
  get_mpi_comm(const MeshType &mesh)
  {
    const auto *tria_parallel =
      dynamic_cast<const parallel::TriangulationBase<MeshType::dimension,
                                                     MeshType::space_dimension> *>(
        &(mesh.get_triangulation()));

    return tria_parallel != nullptr ? tria_parallel->get_communicator() : MPI_COMM_SELF;
  }

  namespace Utilities
  {
    namespace MPI
    {
      template <int dim, int spacedim = dim>
      class RemotePointEvaluation
      {
      public:
        RemotePointEvaluation(const double tolerance = 1e-10)
          : tolerance(tolerance)
        {}

        void
        reinit(const std::vector<Point<spacedim>>  quadrature_points,
               const Triangulation<dim, spacedim> &tria,
               const Mapping<dim, spacedim> &      mapping)
        {
          comm = get_mpi_comm(tria);

          // create bounding boxed of local active cells
          std::vector<BoundingBox<spacedim>> local_boxes;
          for (const auto cell : tria.active_cell_iterators())
            if (cell->is_locally_owned())
              local_boxes.push_back(mapping.get_bounding_box(cell));

          // create r-tree of bounding boxes
          const auto local_tree = pack_rtree(local_boxes);

          // compress r-tree to a minimal set of bounding boxes
          const auto local_reduced_box = extract_rtree_level(local_tree, 0);

          // gather bounding boxes of other processes
          const auto global_bounding_boxes =
            Utilities::MPI::all_gather(comm, local_reduced_box);

          // determine ranks which might poses quadrature point
          auto potentially_relevant_points_per_process =
            std::vector<std::vector<Point<spacedim>>>(global_bounding_boxes.size());

          auto potentially_relevant_points_per_process_id =
            std::vector<std::vector<unsigned int>>(global_bounding_boxes.size());

          for (unsigned int i = 0; i < quadrature_points.size(); ++i)
            {
              const auto &point = quadrature_points[i];
              for (unsigned rank = 0; rank < global_bounding_boxes.size(); ++rank)
                for (const auto &box : global_bounding_boxes[rank])
                  if (box.point_inside(point))
                    {
                      potentially_relevant_points_per_process[rank].emplace_back(point);
                      potentially_relevant_points_per_process_id[rank].emplace_back(i);
                      break;
                    }
            }

          // TODO
          std::map<unsigned int,
                   std::vector<std::pair<std::pair<int, int>, Point<spacedim>>>>
            relevant_remote_points_per_process;

          // only communicate with processes that might have a quadrature point
          std::vector<unsigned int> targets;

          for (unsigned int i = 0; i < potentially_relevant_points_per_process.size();
               ++i)
            if (potentially_relevant_points_per_process[i].size() > 0 &&
                i != Utilities::MPI::this_mpi_process(comm))
              targets.emplace_back(i);


          std::map<unsigned int, std::vector<unsigned int>>
            relevant_points_per_process_offset;
          std::map<unsigned int, std::vector<unsigned int>>
            relevant_points_per_process_count;



          const std::vector<bool>          marked_vertices;
          const GridTools::Cache<dim, dim> cache(tria, mapping);
          auto                             cell_hint = tria.begin_active();

          const auto find_all_locally_owned_active_cells_around_point =
            [&](const Point<dim> &point) {
              std::vector<
                std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>>
                locally_owned_active_cells_around_point;

              try
                {
                  const auto first_cell = GridTools::find_active_cell_around_point(
                    cache, point, cell_hint, marked_vertices, tolerance);

                  cell_hint = first_cell.first;

                  const auto active_cells_around_point =
                    GridTools::find_all_active_cells_around_point(
                      mapping, tria, point, tolerance, first_cell);

                  locally_owned_active_cells_around_point.reserve(
                    active_cells_around_point.size());

                  for (const auto &cell : active_cells_around_point)
                    if (cell.first->is_locally_owned())
                      locally_owned_active_cells_around_point.push_back(cell);
                }
              catch (...)
                {}

              return locally_owned_active_cells_around_point;
            };


          // for local quadrature points no communication is needed...
          const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

          {
            const auto &potentially_relevant_points =
              potentially_relevant_points_per_process[my_rank];
            const auto &potentially_relevant_points_offset =
              potentially_relevant_points_per_process_id[my_rank];

            std::vector<std::pair<std::pair<int, int>, Point<spacedim>>> points;
            std::vector<unsigned int>                                    point_ids;
            std::vector<unsigned int>                                    point_counts;

            for (unsigned int j = 0; j < potentially_relevant_points.size(); ++j)
              {
                const auto adjacent_cells =
                  find_all_locally_owned_active_cells_around_point(
                    potentially_relevant_points[j]);

                if (adjacent_cells.size() > 0)
                  {
                    for (auto cell : adjacent_cells)
                      points.emplace_back(std::make_pair(cell.first->level(),
                                                         cell.first->index()),
                                          cell.second);

                    point_ids.push_back(potentially_relevant_points_offset[j]);
                    point_counts.push_back(adjacent_cells.size());
                  }
              }


            if (points.size() > 0)
              {
                // to send
                relevant_remote_points_per_process[my_rank] = points;

                // to recv
                relevant_points_per_process_offset[my_rank] = point_ids;
                relevant_points_per_process_count[my_rank]  = point_counts;
              }
          }

          // send to remote ranks the requested quadrature points and eliminate
          // not needed ones (note: currently, we cannot communicate points ->
          // switch to doubles here)
          Utilities::MPI::ConsensusAlgorithms::AnonymousProcess<double, unsigned int>
            process(
              [&]() { return targets; },
              [&](const unsigned int other_rank, std::vector<double> &send_buffer) {
                // send requested points
                for (auto point : potentially_relevant_points_per_process[other_rank])
                  for (unsigned int i = 0; i < spacedim; ++i)
                    send_buffer.emplace_back(point[i]);
              },
              [&](const unsigned int &       other_rank,
                  const std::vector<double> &recv_buffer,
                  std::vector<unsigned int> &request_buffer) {
                // received points, determine if point is actually possessed,
                // and send the result back

                std::vector<std::pair<std::pair<int, int>, Point<spacedim>>>
                  relevant_remote_points;

                request_buffer.clear();
                request_buffer.resize(recv_buffer.size() / spacedim);

                for (unsigned int i = 0, j = 0; i < recv_buffer.size();
                     i += spacedim, ++j)
                  {
                    Point<spacedim> point;
                    for (unsigned int j = 0; j < spacedim; ++j)
                      point[j] = recv_buffer[i + j];

                    const auto adjacent_cells =
                      find_all_locally_owned_active_cells_around_point(point);

                    request_buffer[j] = adjacent_cells.size();

                    if (adjacent_cells.size() > 0)
                      {
                        for (auto cell : adjacent_cells)
                          relevant_remote_points.emplace_back(
                            std::make_pair(cell.first->level(), cell.first->index()),
                            cell.second);
                      }
                  }

                if (relevant_remote_points.size() > 0)
                  relevant_remote_points_per_process[other_rank] = relevant_remote_points;
              },
              [&](const unsigned int other_rank, std::vector<unsigned int> &recv_buffer) {
                // prepare buffer
                recv_buffer.resize(
                  potentially_relevant_points_per_process[other_rank].size());
              },
              [&](const unsigned int               other_rank,
                  const std::vector<unsigned int> &recv_buffer) {
                // store recv_buffer -> make the algorithm deterministic

                const auto &potentially_relevant_points =
                  potentially_relevant_points_per_process[other_rank];
                const auto &potentially_relevant_points_offset =
                  potentially_relevant_points_per_process_id[other_rank];

                AssertDimension(potentially_relevant_points.size(), recv_buffer.size());
                AssertDimension(potentially_relevant_points_offset.size(),
                                recv_buffer.size());

                std::vector<unsigned int> point_ids;
                std::vector<unsigned int> point_counts;

                for (unsigned int i = 0; i < recv_buffer.size(); ++i)
                  if (recv_buffer[i] > 0)
                    {
                      point_ids.push_back(potentially_relevant_points_offset[i]);
                      point_counts.push_back(recv_buffer[i]);
                    }

                if (point_ids.size() > 0)
                  {
                    relevant_points_per_process_offset[other_rank] = point_ids;
                    relevant_points_per_process_count[other_rank]  = point_counts;

                    recv_ranks.push_back(other_rank);
                  }
              });

          Utilities::MPI::ConsensusAlgorithms::Selector<double, unsigned int>(process,
                                                                              comm)
            .run();

          std::vector<unsigned int> quadrature_points_count(quadrature_points.size(), 0);
          std::vector<std::pair<unsigned int, unsigned int>> indices_temp;

          for (const auto &i : relevant_points_per_process_offset)
            {
              const unsigned int rank = i.first;


              const auto &relevant_points_offset =
                relevant_points_per_process_offset[rank];
              const auto &relevant_points_count = relevant_points_per_process_count[rank];

              for (unsigned int j = 0; j < relevant_points_offset.size(); ++j)
                for (unsigned int k = 0; k < relevant_points_count[j]; ++k)
                  {
                    auto &qp_counter = quadrature_points_count[relevant_points_offset[j]];
                    indices_temp.emplace_back(relevant_points_offset[j], qp_counter);

                    ++qp_counter;
                  }
            }

          quadrature_points_ptr = {0};

          this->unique_mapping = true;
          for (const auto &i : quadrature_points_count)
            {
              this->unique_mapping &= (i == 1);
              quadrature_points_ptr.push_back(quadrature_points_ptr.back() + i);
            }

          indices = {};
          for (const auto i : indices_temp)
            indices.push_back(quadrature_points_ptr[i.first] + i.second);

          this->relevant_remote_points_per_process = {};

          send_ranks = {};
          send_ptr   = {0};

          for (const auto &i : relevant_remote_points_per_process)
            {
              auto &temp = this->relevant_remote_points_per_process;

              std::vector<std::tuple<std::pair<int, int>, unsigned int, Point<spacedim>>>
                full;
              full.reserve(i.second.size());

              for (const auto &j : i.second)
                full.emplace_back(j.first, full.size(), j.second);

              std::sort(full.begin(), full.end(), [](const auto &a, const auto &b) {
                if (std::get<0>(a) != std::get<0>(b))
                  return std::get<0>(a) < std::get<0>(b);
                else
                  return std::get<1>(a) < std::get<1>(b);
              });

              std::pair<int, int> current = {-1, -1};

              for (const auto j : full)
                {
                  if (current != std::get<0>(j))
                    {
                      current = std::get<0>(j);
                      std::get<0>(temp).emplace_back(current, 0);
                    }

                  std::get<0>(temp).back().second++;

                  std::get<1>(temp).push_back(std::get<2>(j));
                  std::get<2>(temp).push_back(std::get<1>(j) + send_ptr.back());
                }

              send_ranks.push_back(i.first);
              send_ptr.push_back(std::get<2>(temp).size());
            }
        }

        /**
         * Evaluate function @p fu in the requested quadrature points. The result
         * is sorted according to rank.
         */
        template <typename T>
        void
        process(
          std::vector<T> &                                   output,
          std::vector<T> &                                   input,
          const std::function<void(
            std::vector<T> &,
            const std::tuple<std::vector<std::pair<std::pair<int, int>, unsigned int>>,
                             std::vector<Point<spacedim>>,
                             std::vector<unsigned int>> &)> &fu) const
        {
          output.resize(quadrature_points_ptr.back());
          input.resize((std::get<1>(this->relevant_remote_points_per_process).size()));

          fu(input, relevant_remote_points_per_process);

          // process remote quadrature points and send them away
          std::map<unsigned int, std::vector<char>> temp_map;

          std::vector<MPI_Request> requests;
          requests.reserve(send_ranks.size());

          const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

          // receive result

          std::map<unsigned int, std::vector<T>> temp_recv_map;

          for (unsigned int i = 0; i < send_ranks.size(); ++i)
            {
              if (send_ranks[i] == my_rank)
                {
                  // process locally-owned values
                  temp_recv_map[my_rank] =
                    std::vector<T>(input.begin() + send_ptr[i],
                                   input.begin() + send_ptr[i + 1]);
                  continue;
                }

              temp_map[send_ranks[i]] =
                Utilities::pack(std::vector<T>(input.begin() + send_ptr[i],
                                               input.begin() + send_ptr[i + 1]));

              auto &buffer = temp_map[send_ranks[i]];

              requests.resize(requests.size() + 1);

              MPI_Isend(buffer.data(),
                        buffer.size(),
                        MPI_CHAR,
                        send_ranks[i],
                        11,
                        comm,
                        &requests.back());
            }

          for (const auto &vec : recv_ranks)
            {
              MPI_Status status;
              MPI_Probe(MPI_ANY_SOURCE, 11, comm, &status);

              int message_length;
              MPI_Get_count(&status, MPI_CHAR, &message_length);

              std::vector<char> buffer(message_length);

              MPI_Recv(buffer.data(),
                       buffer.size(),
                       MPI_CHAR,
                       status.MPI_SOURCE,
                       11,
                       comm,
                       MPI_STATUS_IGNORE);

              temp_recv_map[status.MPI_SOURCE] =
                Utilities::unpack<std::vector<T>>(buffer);
            }

          MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

          auto it = indices.begin();
          for (const auto &j : temp_recv_map)
            for (const auto &i : j.second)
              output[*(it++)] = i;
        }


        const std::vector<unsigned int> &
        get_quadrature_points_ptr() const
        {
          return quadrature_points_ptr;
        }

        bool
        is_unique_mapping()
        {
          return unique_mapping;
        }

      private:
        const double tolerance;

        MPI_Comm comm;

        bool unique_mapping;

        // receiver side
        std::vector<unsigned int> quadrature_points_ptr;
        std::vector<unsigned int> indices;

        std::vector<unsigned int> recv_ranks;

        // sender side (TODO: merge)
        std::tuple<std::vector<std::pair<std::pair<int, int>, unsigned int>>,
                   std::vector<Point<spacedim>>,
                   std::vector<unsigned int>>
          relevant_remote_points_per_process;

        std::vector<unsigned int> send_ranks;
        std::vector<unsigned int> send_ptr;
      };



    } // namespace MPI
  }   // namespace Utilities
} // namespace dealii

#endif
