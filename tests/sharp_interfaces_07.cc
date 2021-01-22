#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <adaflo/sharp_interface_util.h>

using namespace dealii;

template <int dim>
class LSFunction : public Function<dim>
{
public:
  LSFunction()
    : Function<dim>(1, 0)
  {}

  double
  value(const Point<dim> &p, const unsigned int) const
  {
    return p.distance(Point<dim>()) - 0.5;
  }
};

template <int dim>
void
test()
{
  const unsigned int n_refinements  = 2;
  const unsigned int fe_degree      = 2;
  const unsigned int n_subdivisions = 3;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>{fe_degree});

  Vector<double> ls_vector(dof_handler.n_dofs());

  MappingQGeneric<dim> mapping(1);

  VectorTools::interpolate(mapping, dof_handler, LSFunction<dim>(), ls_vector);

  std::vector<Point<dim>>          vertices;
  std::vector<::CellData<dim - 1>> cells;

  GridGenerator::MarchingCubeAlgorithm<dim> mc(mapping,
                                               dof_handler.get_fe(),
                                               n_subdivisions);

  for (const auto &cell : dof_handler.active_cell_iterators())
    mc.process_cell(cell, ls_vector, vertices, cells);

  Triangulation<dim - 1, dim> tria_interface;
  tria_interface.create_triangulation(vertices, cells, {});

  {
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;

    DataOut<dim> data_out;
    data_out.set_flags(flags);

    data_out.add_data_vector(dof_handler, ls_vector, "ls");
    data_out.build_patches(mapping, 4);

    std::ofstream out("sharp_interfaces_07_a.vtk");
    data_out.write_vtk(out);
  }

  {
    std::ofstream out("sharp_interfaces_07_b.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria_interface, out);
  }
}

int
main()
{
  test<2>();
}