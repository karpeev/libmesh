/* The libMesh Finite Element Library. */
/* Copyright (C) 2003  Benjamin S. Kirk */

/* This library is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU Lesser General Public */
/* License as published by the Free Software Foundation; either */
/* version 2.1 of the License, or (at your option) any later version. */

/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU */
/* Lesser General Public License for more details. */

/* You should have received a copy of the GNU Lesser General Public */
/* License along with this library; if not, write to the Free Software */
/* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA */

// <h1>Systems Example 2 - Unsteady Nonlinear Navier-Stokes</h1>
//
// This example shows how a simple, unsteady, nonlinear system of equations
// can be solved in parallel.  The system of equations are the familiar
// Navier-Stokes equations for low-speed incompressible fluid flow.  This
// example introduces the concept of the inner nonlinear loop for each
// timestep, and requires a good deal of linear algebra number-crunching
// at each step.  If you have a ExodusII viewer such as ParaView installed,
// the script movie.sh in this directory will also take appropriate screen
// shots of each of the solution files in the time sequence.  These rgb files
// can then be animated with the "animate" utility of ImageMagick if it is
// installed on your system.  On a PIII 1GHz machine in debug mode, this
// example takes a little over a minute to run.  If you would like to see
// a more detailed time history, or compute more timesteps, that is certainly
// possible by changing the n_timesteps and dt variables below.

// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <sstream>
#include <math.h>

// Basic include file needed for the mesh functionality.
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/serial_mesh.h"
#include "libmesh/dof_map.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/transient_system.h"
#include "libmesh/perf_log.h"
//
#include "libmesh/utility.h"
#include "libmesh/getpot.h"
#include "libmesh/periodic_boundaries.h"
#include "libmesh/periodic_boundary.h"

//#include "libmesh/fe.h"
//#include "libmesh/quadrature_gauss.h"
//#include "libmesh/dof_map.h"
//#include "libmesh/sparse_matrix.h"
//#include "libmesh/dense_matrix.h"
//#include "libmesh/dense_vector.h"

#include "navier_stokes_assemble.h"
#include "libmesh/petsc_ts_system.h"
#include "libmesh/petsc_ts_solver.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;


// ============================================================================================
// ============================================================================================
// -------------------------------- check requires for libmesh --------------------------------
int check_libmesh()
{
  // This example NaNs with the Eigen sparse linear solvers and Trilinos solvers,
  // but should work OK with either PETSc or Laspack.
  libmesh_example_requires(libMesh::default_solver_package() != EIGEN_SOLVERS,
                           "--enable-petsc or --enable-laspack");
  libmesh_example_requires(libMesh::default_solver_package() != TRILINOS_SOLVERS,
                           "--enable-petsc or --enable-laspack");
  
  // Skip this 2D example if libMesh was compiled as 1D-only.
  libmesh_example_requires(2 <= LIBMESH_DIM, "2D/3D support");
  
  // read control parameters from input file
#ifndef LIBMESH_ENABLE_AMR
  libmesh_example_requires(false, "--enable-amr");
#endif
  
  
  // my output: test the LibMesh compile info
#if !defined(LIBMESH_HAVE_TRIANGLE)
  std::cout << "LIBMESH does not have Triangle interface! " << std::endl;
#endif
#if !defined(LIBMESH_HAVE_TETGEN)
  std::cout << "LIBMESH does not have TetGen interface! " << std::endl;
#endif
#if !defined(LIBMESH_ENABLE_AMR)
  std::cout << "LIBMESH does not enable AMR! " << std::endl;
#endif
  
  return 0;
}





// ============================================================================================
// ============================================================================================
// The main program.
int main (int argc, char** argv)
{
  // Initialize libMesh.
  LibMeshInit init (argc, argv);

  // check libmesh system
  check_libmesh();
  
  // Parse the input file and Read in parameters from the input file
  GetPot input_file("navier_stokes_control.in");
  const int max_linear_iterations   = input_file("max_linear_iterations", 5000);
  const unsigned int dim            = input_file("dimension", 2);
  const Real viscosity              = input_file("viscosity", 1.0);   // (kinematic viscosity)
  const Real dt                     = input_file("dt", 0.005);        // time increment
  const unsigned int n_timesteps    = input_file("n_time_steps", 10);
  const Real initial_time           = input_file("initial_time", 0.0);
  const Real final_time             = input_file("final_time", 1.0);
  const bool periodicity            = input_file("periodicity", true);   // periodic bc
  
//  const bool write_es               = input_file("write_es", true);      // write out eqn-system
//  const unsigned int write_interval = input_file("write_interval", 1);
  const Real XA = input_file("XA", -1.);  const Real XB = input_file("XB", +1.);
  const Real YA = input_file("YA", -1.);  const Real YB = input_file("YB", +1.);
  const Real ZA = input_file("ZA", -1.);  const Real ZB = input_file("ZB", +1.);
  const unsigned int nx_mesh = input_file("nx_mesh", 10);
  const unsigned int ny_mesh = input_file("ny_mesh", 10);
  const unsigned int nz_mesh = input_file("nz_mesh", 10);
  
  // output the mesh and domain information
  std::cout << "nx_mesh = " << nx_mesh <<", Lx = " << XB-XA <<std::endl
            << "ny_mesh = " << ny_mesh <<", Ly = " << YB-YA <<std::endl
            << "nz_mesh = " << nz_mesh <<", Lz = " << ZB-ZA <<std::endl;
  std::cout << "initial time = " << initial_time << ", final time = " << final_time
            << ", dt = " << dt << ", timesteps = " << n_timesteps << std::endl; 

  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
//  Mesh mesh(init.comm());
  SerialMesh mesh(init.comm());

  // Use the MeshTools::Generation mesh generator to create a uniform
  // 2D grid on the square [-1,1]^2.  We instruct the mesh generator
  // to build a mesh of 8x8 \p Quad9 elements in 2D.  Building these
  // higher-order elements allows us to use higher-order
  // approximation, as in example 3.
  if(dim==2)
    MeshTools::Generation::build_square (mesh, nx_mesh, ny_mesh,
                                         XA, XB, YA, YB, QUAD4);
  else if(dim==3)
    MeshTools::Generation::build_cube (mesh, nx_mesh, ny_mesh, nz_mesh,
                                       XA, XB, YA, YB, ZA, ZB, HEX8);
  else
    libmesh_example_requires(dim <= LIBMESH_DIM, "2D/3D support");
  // end if

  mesh.all_second_order();
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);

  // Creates a transient system named "Navier-Stokes"
  NSPetscTSSystem &system = equation_systems.add_system<NSPetscTSSystem> ("Navier-Stokes");

  // Add the variables "u" & "v" to "Navier-Stokes".  They
  // will be approximated using second-order approximation.
  unsigned int u_var = 0, v_var = 0, w_var = 0, p_var = 0;
  u_var = system.add_variable ("u", SECOND);
  v_var = system.add_variable ("v", SECOND);
  if(dim==3)w_var = system.add_variable ("w", SECOND);
  p_var = system.add_variable ("p", FIRST);

  // Give the system a pointer to the matrix assembly function.
//  system.attach_assemble_function (assemble_navier_stokes);
//  system.attach_init_function (init_navier_stokes);

  // set parameters of equations systems
  equation_systems.parameters.set<unsigned int>("linear solver maximum iterations") = max_linear_iterations;
  equation_systems.parameters.set<Real>        ("linear solver tolerance") = TOLERANCE*1E-3;
  equation_systems.parameters.set<Real>        ("viscosity")   = viscosity;
  equation_systems.parameters.set<Real>        ("XA_boundary") = XA;
  equation_systems.parameters.set<Real>        ("XB_boundary") = XB;
  equation_systems.parameters.set<Real>        ("YA_boundary") = YA;
  equation_systems.parameters.set<Real>        ("YB_boundary") = YB;
  equation_systems.parameters.set<Real>        ("ZA_boundary") = ZA;
  equation_systems.parameters.set<Real>        ("ZB_boundary") = ZB;
  equation_systems.parameters.set<bool>        ("periodicity") = periodicity;
  
  
  equation_systems.parameters.set<Real>        ("dt")          = dt;
  equation_systems.parameters.set<Real>        ("time steps")  = n_timesteps;
  equation_systems.parameters.set<Real>        ("initial time")= initial_time;
  equation_systems.parameters.set<Real>        ("final time")  = final_time;

  // add periodic boundary conditions for u v w, but not for p
  if (periodicity) //periodicity
  {
    PeriodicBoundary horz(RealVectorValue(XB-XA, 0., 0.));
    horz.set_variable(u_var);
    horz.set_variable(v_var);
    
    // is this boundary number still true for 3D?
    if(dim==2)
    {
      horz.myboundary = 3;
      horz.pairedboundary = 1;
    }
    else if(dim==3)
    {
      horz.set_variable(w_var);
      horz.myboundary = 4;
      horz.pairedboundary = 2;
    } // end if
    
    DofMap& dof_map = system.get_dof_map();
    dof_map.add_periodic_boundary(horz);
    std::cout<<"========================= periodic bc is applied ========================"<<std::endl;
  }
  
  // Initialize the data structures for the equation system.
  equation_systems.init ();

  // Prints information about the system to the screen.
  equation_systems.print_info();
  
  // Create a performance-logging object for this example
  //PerfLog perf_log("Navier Stokes Equation");

  // Get a reference to the Stokes system to use later.
  // is this necessary, because it is the same as the \p system above
  NSPetscTSSystem&  navier_stokes_system = equation_systems.get_system<NSPetscTSSystem>("Navier-Stokes");


  // Petsc TS solver
  printf("************* Entering Petsc TS solver ...... \n");
  PetscTSSolver<Number> ts_solver(navier_stokes_system);
  ts_solver.set_duration(initial_time, final_time);
  ts_solver.set_timestep(dt, n_timesteps);
  
//  navier_stokes_system.solve();
  ts_solver.init();
  ts_solver.solve();
  
  // All done.
  return 0;
  
}






