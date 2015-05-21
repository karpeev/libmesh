//
//  petsc_ts_system.c
//  
//
//  Created by Xujun Zhao on 5/5/15.
//
//
#include "libmesh/exodusII_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dof_map.h"

#include "libmesh/petsc_ts_system.h"
//#include "navier_stokes_assemble.h"

namespace libMesh
{

// ---------------------------------------------------------------------
PetscTSSystem::PetscTSSystem(EquationSystems& es,
                             const std::string& name_in,
                             const unsigned int number_in):
  Parent   (es, name_in, number_in)
{
  // do nothing right now
}


// ---------------------------------------------------------------------
PetscTSSystem::~PetscTSSystem ()
{
  // Clear data
  this->clear();
}


// ---------------------------------------------------------------------
void PetscTSSystem::clear ()
{
  // clear the parent data
  Parent::clear();
}

  
// ---------------------------------------------------------------------
void PetscTSSystem::init ()
{
  // initialize parent data
  Parent::init();
}


// ---------------------------------------------------------------------
void PetscTSSystem::reinit ()
{
  // re-initialize the ts solver interface
  ts_solver->clear();
  
  // initialize parent data
  Parent::reinit();
}


// ---------------------------------------------------------------------
void PetscTSSystem::set_solver_parameters ()
{
  // Get a reference to the EquationSystems
  const EquationSystems& es = this->get_equation_systems();
  
  // Get the user-specifiied ts solver tolerances
  const Real   t0 = es.parameters.get<Real>("initial time");
  const Real maxt = es.parameters.get<Real>("final time");
  const Real   dt = es.parameters.get<Real>("dt");
  const unsigned int nsteps = es.parameters.get<unsigned int>("time steps");

  printf("inital tme = %f, final time = %f, dt = %f, time steps = %u\n",
         t0, maxt, dt, nsteps);
  
  // set the parameters for ts solver
  if (ts_solver.get())
  {
    ts_solver->set_duration(t0, maxt);
    ts_solver->set_timestep (dt,nsteps);
  }
}
  
// ---------------------------------------------------------------------
void PetscTSSystem::solve ()
{
  // Log how long the nonlinear solve takes.
  START_LOG("solve()", "PetscTSSystem");
  
  // what parameters can we set for ts_solver?
   this->set_solver_parameters();
  
  // there is solver constructed through build(), but not init()
  if (!ts_solver.get())
    ts_solver->init();
  
  // call ts solver to solve the system
  ts_solver->solve();

  
  // Stop logging the nonlinear solve
  STOP_LOG("solve()", "PetscTSSystem");
  
  // Update the system after the solve
  this->update();
}


// ---------------------------------------------------------------------
// F(t,U,U_t)
void PetscTSSystem::IFunction (Real time,
                               const NumericVector<Number>& X,
                               const NumericVector<Number>& Xdot,
                               NumericVector<Number>& F)
{
}


// ---------------------------------------------------------------------
// compute the matrix dF/dU + a*dF/dU_t where F(t,U,U_t)
void PetscTSSystem::IJacobian (Real time,
                               const NumericVector<Number>& X,
                               const NumericVector<Number>& Xdot,
                               Real shift,
                               SparseMatrix<Number>& IJ,
                               SparseMatrix<Number>& IJpre)
{
}


// ---------------------------------------------------------------------
void PetscTSSystem::monitor (int  step, Real time,
                             NumericVector<Number>& X)
{
}
} // end of namespace
