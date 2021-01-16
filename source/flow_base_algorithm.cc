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

#include <deal.II/distributed/tria.h>

#include <adaflo/flow_base_algorithm.h>
#include <adaflo/util.h>

#include <filesystem>

using namespace dealii;



template <int dim>
helpers::BoundaryDescriptor<dim>::BoundaryDescriptor()
{
  for (unsigned int d = 0; d < dim; ++d)
    periodic_boundaries[d] = std::pair<types::boundary_id, types::boundary_id>(-1, -1);
}



template <int dim>
FlowBaseAlgorithm<dim>::FlowBaseAlgorithm(
  const std::shared_ptr<Mapping<dim>> mapping_data)
  : boundary(new helpers::BoundaryDescriptor<dim>())
  , mapping_data(mapping_data)
  , mapping(*mapping_data)
{}



template <int dim>
FlowBaseAlgorithm<dim>::FlowBaseAlgorithm()
  : FlowBaseAlgorithm(std::make_shared<MappingQ<dim>>(3))
{}



template <int dim>
FlowBaseAlgorithm<dim>::~FlowBaseAlgorithm()
{}



template <int dim>
void
FlowBaseAlgorithm<dim>::clear_all_boundary_conditions()
{
  boundary->dirichlet_conditions_u.clear();
  boundary->open_conditions_p.clear();
  boundary->pressure_fix.clear();
  boundary->normal_flux.clear();
  boundary->symmetry.clear();
  boundary->no_slip.clear();
}



template <int dim>
void
FlowBaseAlgorithm<dim>::set_velocity_dirichlet_boundary(
  const types::boundary_id              boundary_id,
  const std::shared_ptr<Function<dim>> &velocity_function,
  const int                             inflow_fluid_type)
{
  if (velocity_function.get() == 0)
    return set_no_slip_boundary(boundary_id);
  AssertThrow(velocity_function->n_components == dim,
              ExcMessage("Velocity boundary function need to have dim components."));
  boundary->dirichlet_conditions_u[boundary_id] = velocity_function;

  switch (inflow_fluid_type)
    {
      case 0:
        break;
      case 1:
        boundary->fluid_type[boundary_id] =
          std::make_shared<Functions::ConstantFunction<dim>>(1, 1);
        break;
      case -1:
        boundary->fluid_type[boundary_id] =
          std::make_shared<Functions::ConstantFunction<dim>>(-1, 1);
        break;
      default:
        AssertThrow(false, ExcMessage("Unknown fluid type"));
    }
}



template <int dim>
void
FlowBaseAlgorithm<dim>::set_open_boundary(
  const types::boundary_id              boundary_id,
  const std::shared_ptr<Function<dim>> &pressure_function,
  const int                             inflow_fluid_type)
{
  if (pressure_function.get() == 0)
    boundary->open_conditions_p[boundary_id] =
      std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(1));
  else
    {
      AssertThrow(pressure_function->n_components == 1,
                  ExcMessage("Pressure boundary function needs to be scalar."));
      boundary->open_conditions_p[boundary_id] = pressure_function;
    }

  switch (inflow_fluid_type)
    {
      case 0:
        break;
      case 1:
        boundary->fluid_type[boundary_id] =
          std::make_shared<Functions::ConstantFunction<dim>>(1, 1);
        break;
      case -1:
        boundary->fluid_type[boundary_id] =
          std::make_shared<Functions::ConstantFunction<dim>>(-1, 1);
        break;
      default:
        AssertThrow(false, ExcMessage("Unknown fluid type"));
    }
}



template <int dim>
void
FlowBaseAlgorithm<dim>::set_open_boundary_with_normal_flux(
  const types::boundary_id              boundary_id,
  const std::shared_ptr<Function<dim>> &pressure_function,
  const int                             inflow_fluid_type)
{
  if (pressure_function.get() == 0)
    boundary->open_conditions_p[boundary_id] =
      std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(1));
  else
    {
      AssertThrow(pressure_function->n_components == 1,
                  ExcMessage("Pressure boundary function needs to be scalar."));
      boundary->open_conditions_p[boundary_id] = pressure_function;
    }
  boundary->normal_flux.insert(boundary_id);

  switch (inflow_fluid_type)
    {
      case 0:
        break;
      case 1:
        boundary->fluid_type[boundary_id] =
          std::make_shared<Functions::ConstantFunction<dim>>(1, 1);
        break;
      case -1:
        boundary->fluid_type[boundary_id] =
          std::make_shared<Functions::ConstantFunction<dim>>(-1, 1);
        break;
      default:
        AssertThrow(false, ExcMessage("Unknown fluid type"));
    }
}



template <int dim>
void
FlowBaseAlgorithm<dim>::fix_pressure_constant(
  const types::boundary_id              boundary_id,
  const std::shared_ptr<Function<dim>> &pressure_function)
{
  AssertThrow(pressure_function.get() == 0 || pressure_function->n_components == 1,
              ExcMessage("Pressure boundary function need to be scalar."));
  boundary->pressure_fix[boundary_id] = pressure_function;
}



template <int dim>
void
FlowBaseAlgorithm<dim>::set_symmetry_boundary(const types::boundary_id boundary_id)
{
  boundary->symmetry.insert(boundary_id);
}



template <int dim>
void
FlowBaseAlgorithm<dim>::set_no_slip_boundary(const types::boundary_id boundary_id)
{
  boundary->no_slip.insert(boundary_id);
}



template <int dim>
void
FlowBaseAlgorithm<dim>::set_periodic_direction(
  const unsigned int       direction,
  const types::boundary_id incoming_boundary_id,
  const types::boundary_id outgoing_boundary_id)
{
  AssertThrow(direction < dim,
              ExcMessage("Coordinate direction must be between 0 and the dim"));
  boundary->periodic_boundaries[direction] =
    std::make_pair(incoming_boundary_id, outgoing_boundary_id);
}



template <int dim>
void
FlowBaseAlgorithm<dim>::write_data_output(const std::string & output_name,
                                          const TimeStepping &time_stepping,
                                          const double        output_frequency,
                                          const Triangulation<dim> &,
                                          DataOut<dim> &data_out) const
{
  // append time step and processor count to given output base name

  const unsigned int no_time_steps =
    (time_stepping.final() - time_stepping.start()) / output_frequency + 1;
  const unsigned int digits_steps = std::log10((double)no_time_steps) + 1;

  const unsigned int cycle = time_stepping.now() / output_frequency + 0.51;
  data_out.set_flags(DataOutBase::VtkFlags(time_stepping.now(), cycle));

  std::filesystem::path path(output_name);

  data_out.write_vtu_with_pvtu_record(
    path.relative_path(), path.filename(), cycle, MPI_COMM_WORLD, digits_steps, 1);
}


template struct FlowBaseAlgorithm<2>;
template struct FlowBaseAlgorithm<3>;
