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


#include <adaflo/parameters.h>
#include <adaflo/navier_stokes.h>
#include <adaflo/level_set_okz.h>
#include <adaflo/level_set_okz_matrix.h>


#include <deal.II/base/multithread_info.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>


using namespace dealii;

struct TwoPhaseParameters : public FlowParameters
{
  TwoPhaseParameters (const std::string &parameter_filename)
  {
    ParameterHandler prm;
    FlowParameters::declare_parameters (prm);
    prm.enter_subsection ("Problem-specific");
    prm.declare_entry ("two-phase method","level set okz",
                       Patterns::Selection("level set okz|level set okz matrix|phase field"),
                       "Defines the two-phase method to be used");
    prm.leave_subsection();
    check_for_file(parameter_filename, prm);
    parse_parameters (parameter_filename, prm);
    prm.enter_subsection ("Problem-specific");
    solver_method = prm.get("two-phase method");
    prm.leave_subsection();
  }

  std::string solver_method;
};


//Initial position of the particle

template <int dim>
class InitialValuesLS : public Function<dim>
{
public:
  InitialValuesLS ()
    :
    Function<dim>(1, 0)
  {}

  double value (const Point<dim> &p,
                const unsigned int /*component*/) const
  {
    const double radius = 0.25;
    Point<dim> distance_from_origin = p;
    for (unsigned int i=0; i<dim; ++i)
      distance_from_origin[i] = 0.5;
    return p.distance(distance_from_origin) - radius;
  }
};




//None Homogenous boundary conditions
//Here we impose the velocity at some part part of the boundary
//Depending of the location  of the boundary one of the component
//is 0. We use a switch to see which part we are dealing with
//This will be used at the boundary with id "boudary_ind_values"

template <int dim>
class BoundaryValuesVelocity : public Function<dim>
{
public:
  BoundaryValuesVelocity (const int boundary_ind=1);
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual  void vector_value (const Point<dim> &p,
                              Vector<double>   &value) const;
  const int  boundary_ind_values;
};

template <int dim>
BoundaryValuesVelocity<dim>::BoundaryValuesVelocity (const int boundary_ind)
  :
  Function<dim>(dim),
  boundary_ind_values(boundary_ind)
{}

template <int dim>
double
BoundaryValuesVelocity<dim>::value (const Point<dim>  &p,
                                    const unsigned int component) const
{
  (void)p;
  (void)component;
  double p_values=0.1;
  Assert (component < this->n_components,
          ExcIndexRange (component, 0, this->n_components));
  switch (boundary_ind_values)
    {
    case 1:
      if (component == 0)
        p_values=0.5;
      else
        p_values=0;
      break;
    case 2:
      if (component == 0)
        p_values=0;
      else
        p_values= 1;
      break;
    case 3:
      if (component == 0)
        p_values=-10;
      else
        p_values=0;
      break;
    default:
      AssertThrow (false, ExcMessage("Unknown boundary type"));
      break;
    }
  if (this->get_time() <= 0)
    p_values = 0;
  else if (this->get_time() < 0.01)
    p_values *= std::sin(numbers::PI*this->get_time()/(0.02));
  return p_values;
}

template <int dim>
void
BoundaryValuesVelocity<dim>::vector_value (const Point<dim> &p,
                                           Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values(c) = BoundaryValuesVelocity<dim>::value (p,  c);
}



namespace MicroFluidic
{





// @sect3{The <code>TwoPhaseFlowProblem</code> class template}
//All the class member and function are defined in TwoPhase.h

  /*
   *  Here start the main program.  We define  a class MicroFluidicProblem
   *  and its members. We will define the geometry in create_geometry_and_partition_boundary ()
   *  and define the boundary indicator. We use these latter ones
   *  to specify different boundary conditions.
   *  Through the member functions set_all_boundary_conditions ()
   *  we impose the boundary conditions. We remind that they are
   *  defined in the file boundary_condition.h.
   *  In  member function run (), we will call functions defined in TwoPhase.h
   *  for the resolution.
   *
   * */



  // @sect3{The <code>MicroFluidicProblem</code> class template}
  template <int dim>
  class MicroFluidicProblem
  {

  public:
    MicroFluidicProblem (const TwoPhaseParameters &parameters);
    void run (const TwoPhaseParameters &parameters);


  private:

    MPI_Comm mpi_communicator;
    ConditionalOStream  pcout;
    TwoPhaseParameters        parameters;
    parallel::distributed::Triangulation<dim>   triangulation;
    std::shared_ptr<TwoPhaseBaseAlgorithm<dim> > solver;

    void set_all_boundary_conditions (const TwoPhaseParameters &parameters);

  };


  // @sect1{MicroFluidicProblem::MicroFluidicProblem constructor}
  template <int dim>
  MicroFluidicProblem<dim>::MicroFluidicProblem (const TwoPhaseParameters &parameters)
    :
    mpi_communicator(MPI_COMM_WORLD),

    pcout (std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),

    parameters (parameters),
    triangulation (mpi_communicator)
  {
    if (parameters.solver_method == "level set okz")
      solver.reset(new LevelSetOKZSolver<dim>(parameters, triangulation));
    else if (parameters.solver_method == "level set okz matrix")
      solver.reset(new LevelSetOKZMatrixSolver<dim>(parameters, triangulation));
    // else if (parameters.solver_method == "phase field")
    //   solver.reset(new PhaseFieldSolver<dim>(parameters, triangulation));
    else
      AssertThrow(false, ExcMessage("Unknown solver selected in parameter file"));
  }


  /*Partition boundary
    We write here a function that enable us to partition
    the boundary by setting different boundary indicator
    This will allow us to define different boundary constraint
    on each face. By default the boundary indicator is 0. We will
    flagged some part with indicators 1, 2, 3 or 4.
  */



  template <int dim>
  void set_boundary_indicators (Triangulation<dim> &triangulation)
  {
    typename Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();

    for ( ; cell!=endc; ++cell)
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
          const Point<dim> face_center = cell->face(face)->center();

          if (cell->face(face)->at_boundary() &&
              (std::fabs(cell->face(face)->center()[0]+2)<1e-14))
            cell->face(face)->set_all_boundary_ids(1);

          if (cell->face(face)->at_boundary() &&
              (std::fabs(cell->face(face)->center()[0]-3)<1e-14))        //x=2.5
            cell->face(face)->set_all_boundary_ids(3);


          if (cell->face(face)->at_boundary() &&
              (std::fabs(cell->face(face)->center()[1]+0.5)<1e-14))
            cell->face(face)->set_all_boundary_ids(2);

          if (cell->face(face)->at_boundary() &&
              (std::fabs(cell->face(face)->center()[1]-8)<1e-14))
            cell->face(face)->set_all_boundary_ids(4);

          if (cell->face(face)->at_boundary())
            {
              if (
                (face_center[1] <= 1.5 && face_center[1] >= 0.5 )
                &&
                (face_center[0] <= 0    && face_center[0] >= -1 )
              )
                cell->face(face)->set_all_manifold_ids(12);

              if (
                (face_center[1] <= 1.5  && face_center[1] >= 0.5 )
                &&
                (face_center[0] <= 2   && face_center[0] >= 1)
              )
                cell->face(face)->set_all_manifold_ids(23);

              if (
                (face_center[1] <= 4.5  && face_center[1] >= 3.5 )
                &&
                (face_center[0] <= 2   && face_center[0] >= 1)
              )
                cell->face(face)->set_all_manifold_ids(34);

              if (
                (face_center[1] <= 4.5  && face_center[1] >= 3.5 )
                &&
                (face_center[0] <= 0   && face_center[0] >= -1 )
              )
                cell->face(face)->set_all_manifold_ids(41);
            }

          if (dim == 3)
            {
              if (cell->face(face)->at_boundary() &&
                  (std::fabs(cell->face(face)->center()[2])<1e-14))
                cell->face(face)->set_all_boundary_ids(0);
              if (cell->face(face)->at_boundary() &&
                  (std::fabs(cell->face(face)->center()[2]-1.)<1e-14))
                cell->face(face)->set_all_boundary_ids(0);
            }
        }
  }



  void create_triangulation(Triangulation<2> &triangulation)
  {
    const int dim = 2;
    GridIn<dim> grid_in;
    grid_in.attach_triangulation (triangulation);
    std::string filename = "microfluidic_2D.msh";
    std::ifstream file (filename.c_str());
    AssertThrow (file, ExcFileNotOpen (filename.c_str()));
    grid_in.read_msh (file);

    const PolarManifold<2> boundary_description_12(Point<2>(-1,0.5));  //Point 20
    triangulation.set_manifold (12, boundary_description_12);

    const PolarManifold<2> boundary_description_23(Point<2>(2,0.5));  //Point 17
    triangulation.set_manifold (23, boundary_description_23);


    const PolarManifold<2> boundary_description_34(Point<2>(2,4.5));  //Point 18
    triangulation.set_manifold (34, boundary_description_34);

    const PolarManifold<2> boundary_description_41(Point<2>(-1,4.5)); //Point 19
    triangulation.set_manifold (41, boundary_description_41);
    set_boundary_indicators(triangulation);
  }



  void create_triangulation(Triangulation<3> &triangulation)
  {
    Triangulation<2> tria_2d;
    create_triangulation(tria_2d);
    GridGenerator::extrude_triangulation(tria_2d, 9, 1., triangulation);

    set_boundary_indicators(triangulation);
    const CylindricalManifold<3> boundary_description_12(Point<3>(0,0,1),Point<3>(-1,0.5,0));  //Point 20
    triangulation.set_manifold (12, boundary_description_12);

    const CylindricalManifold<3> boundary_description_23(Point<3>(0,0,1),Point<3>(2,0.5,0));  //Point 17
    triangulation.set_manifold (23, boundary_description_23);


    const CylindricalManifold<3> boundary_description_34(Point<3>(0,0,1),Point<3>(2,4.5,0));  //Point 18
    triangulation.set_manifold (34, boundary_description_34);

    const CylindricalManifold<3> boundary_description_41(Point<3>(0,0,1),Point<3>(-1,4.5,0)); //Point 19
    triangulation.set_manifold (41, boundary_description_41);

  }



  // @sect3{MicroFluidicProblem::set_and_apply_boundary_conditions}
  template <int dim>
  void MicroFluidicProblem<dim>::set_all_boundary_conditions (const TwoPhaseParameters &parameters)
  {
    //Check if it much with what we have in the file set_type_boundaries.h


    //Boundary flagged by the indicator 1, 2 and 3
    //Not that in each boundary we applied different condition
    //using the velocity

    solver->set_velocity_dirichlet_boundary(1,
                                            std::shared_ptr<Function<dim> > (new BoundaryValuesVelocity<dim>(1)), -1);
    solver->set_velocity_dirichlet_boundary(2,
                                            std::shared_ptr<Function<dim> > (new BoundaryValuesVelocity<dim>(2)), -1);
    solver->set_velocity_dirichlet_boundary(3,
                                            std::shared_ptr<Function<dim> > (new BoundaryValuesVelocity<dim>(3)), -1);


    //Boundary flagged by the indicator 0, 12, 23, 34 and  41
    //We remind that they define the round part of the boundary

    solver->set_no_slip_boundary(0);
    solver->set_no_slip_boundary(12);
    solver->set_no_slip_boundary(23);
    solver->set_no_slip_boundary(34);
    solver->set_no_slip_boundary(41);

    if (parameters.pressure_constraint)
      {
        solver->set_no_slip_boundary(0);
        solver->set_no_slip_boundary(12);
        solver->set_no_slip_boundary(23);
        solver->set_no_slip_boundary(34);
        solver->set_no_slip_boundary(41);
      }


    /*Boundary flagged by the indicator 4. We just force the the
      tangential component of the flow will be constrained to zero.
    */
    //set_open_boundary_with_normal_flux is defined in boundary_description.h (cf Navier-Stokes file)
    solver->set_open_boundary_with_normal_flux (4,
                                                std::shared_ptr<Function<dim> > (new Functions::ConstantFunction<dim>(50,1)));
  }



  // @sect4{MicroFluidicProblem::run}
  template <int dim>
  void MicroFluidicProblem<dim>::run (const TwoPhaseParameters &parameters)
  {

    //Print the finite element pair
    pcout << std::endl;
    pcout<<"  ****************** Finite Element pair used in this computation *************"<<std::endl;
    pcout<<"  *****                                                                   *****"<<std::endl;

    if (parameters.augmented_taylor_hood==1)
      pcout<<"  *****              The Augmented Taylor-Hood element is used.           *****"<<std::endl;
    else
      pcout<<"  *****              The Taylor-Hood element is used.                     *****"<<std::endl;
    pcout <<"  *****              Number of MPI processes: "
          << std::setw(5) << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << "                       *****" << std::endl;
    pcout <<"  *****              Number of threads:       "
          << std::setw(5) << MultithreadInfo::n_threads()
          << "                       *****" << std::endl;
    pcout<<"  *****                                                                   *****"<<std::endl;
    pcout<<"  ***************************************************************************** "<<std::endl;

    //Here  we create the geometry  and define all the boundary indicators

    pcout<<"Create geometry for "<< dim << "D problem"<<std::endl;
    create_triangulation(triangulation);

    // set_all_boundary_conditions ();

    pcout<<"Set boundary conditions "<<std::endl;

    set_all_boundary_conditions (parameters);

    //Setup the problem

    pcout<<"Setup problem "<<std::endl;


    solver->setup_problem(Functions::ConstantFunction<dim>(0., dim), InitialValuesLS<dim>());

    /* We  compute divergence-free velocity field in case we start from zero
     * velocity but the boundary condition says something else. It reads as:
     *    "if (navier_stokes.solution.block(0).l2_norm() == 0)
     *         navier_stokes.compute_initial_stokes_field();"
     * (cf two_phase_base.cc, line 97-100)!
    */


    pcout<<"Analyse solution "<<std::endl;

// solver->analyze_solution ();

    pcout<<" Output Solution "<<std::endl;

    solver->output_solution(parameters.output_filename);

    pcout<<" Time Stepping "<<std::endl;

    std::vector<std::vector<double> > solution_data;
    solution_data.push_back(solver->compute_bubble_statistics(0));


    // @sect{Time loop}
    bool first_output = true;
    while (solver->get_time_stepping().at_end() == false)
      {

        solver->advance_time_step();

        solver->output_solution(parameters.output_filename);

        solution_data.push_back(solver->compute_bubble_statistics());

        if (solution_data.size() > 0 &&
            Utilities::MPI::this_mpi_process(triangulation.get_communicator()) == 0 &&
            solver->get_time_stepping().at_tick(parameters.output_frequency))

          {
            const int time_step = 1.000001e4*solver->get_time_stepping().max_step_size();

            std::ostringstream filename3;
            filename3 << parameters.output_filename << "-"
                      << Utilities::int_to_string((int)parameters.adaptive_refinements) << "-"
                      << Utilities::int_to_string(parameters.global_refinements) << "-"
                      << Utilities::int_to_string(time_step, 4)
                      << ".txt";

            std::fstream output_positions3 (filename3.str().c_str(),
                                            first_output ? std::ios::out : std::ios::out
                                            | std::ios::app);

            output_positions3.precision (14);
            if (first_output)
              output_positions3 << "#    time        area      perimeter   circularity   bubble_xvel   bubble_yvel   bubble_xpos    bubble_ypos" << std::endl;
            for (unsigned int i=0; i<solution_data.size(); ++i)
              {
                output_positions3 << " ";
                for (unsigned int j=0; j<solution_data[i].size(); ++j)
                  output_positions3 << solution_data[i][j] << "   ";
                output_positions3 << std::endl;
              }

            solution_data.clear();
            first_output = false;
          }

        solver->refine_grid();
      }

  }




} //Fin namespace




// @sect3{The <code>main</code> function}

// The main function only creates the object
// for the TwoPhaseFlowProblem in 2d or 3d
// (set the respective number to
// <code><2></code> or <code><3></code>!) and
// then runs it. Exceptions are to be catched
// and, as far as known, indicated.
int main (int argc, char **argv)
{
  using namespace dealii;
  using namespace MicroFluidic;

  // change mode for rounding: denormals are flushed to zero to avoid computing
  // on denormals which can slow down things.
#define MXCSR_DAZ (1 << 6)      /* Enable denormals are zero mode */
#define MXCSR_FTZ (1 << 15)     /* Enable flush to zero mode */

  unsigned int mxcsr = __builtin_ia32_stmxcsr ();
  mxcsr |= MXCSR_DAZ | MXCSR_FTZ;
  __builtin_ia32_ldmxcsr (mxcsr);


  try
    {
      deallog.depth_console (0);
      deallog.depth_console (0);
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      std::string paramfile;
      if (argc>1)
        paramfile = argv[1];
      else
        paramfile = "micro_particle.prm";

      TwoPhaseParameters parameters (paramfile);
      if (parameters.dimension == 2)
        {
          MicroFluidicProblem<2> flow_problem (parameters);
          flow_problem.run (parameters);
        }
      else if (parameters.dimension == 3)
        {
          MicroFluidicProblem<3> flow_problem (parameters);
          flow_problem.run (parameters);
        }
      else
        AssertThrow (false, ExcNotImplemented());
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
