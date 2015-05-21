// The libMesh Finite Element Library.
// Copyright (C) 2002-2014 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



#include "libmesh/libmesh_common.h"
#include "libmesh/petsc_matrix.h"  // for PetscMatrix
#include "libmesh/petsc_vector.h"
#include "libmesh/dof_map.h"
#include "libmesh/petscdmlibmesh.h"

#ifdef LIBMESH_HAVE_PETSC

// C++ includes

// Local Includes
#include "libmesh/petsc_ts_system.h"
#include "libmesh/petsc_ts_solver.h"

namespace libMesh
{

//--------------------------------------------------------------------
// Functions with C linkage to pass to PETSc.  PETSc will call these
// methods as needed.
//
// Since they must have C linkage they have no knowledge of a namespace.
// Give them an obscure name to avoid namespace pollution.
  
extern "C"
{

  //-------------------------------------------------------------------
  // this function is called by PETSc at the end of each step
  PetscErrorCode
  __libmesh_petsc_ts_monitor (TS ts, PetscInt step, PetscReal time, Vec x, void *ctx)
  {
    START_LOG("Monitor()", "PetscTSSolver");
    
    libmesh_assert(x); // make sure x is non-NULL
    
    // No way to safety-check this cast, since we got a void*...
    PetscTSSystem *tssys = (PetscTSSystem *)ctx;
    
    // Wrap PETSc Vec as a libMesh::PetscVector<Number> .
    PetscVector<Number> X (x,tssys->comm());
    
    // Call the monitor on the PetscVector X, along with the other arguments.
    tssys->monitor(step,time,X);
    
    STOP_LOG("Monitor()", "PetscTSSolver");
    return 0;
  }

  /*
  //---------------------------------------------------------------
  // this function is called by TS to evaluate the rhs function at X
  PetscErrorCode
  __libmesh_petsc_ts_rhsfunction (TS ts, PetscReal time, Vec x, Vec r, void *ctx)
  {
    START_LOG("RHSFunction()", "PetscTSSolver");

    libmesh_assert(x);
    libmesh_assert(r);
    libmesh_assert(ctx);

    // No way to safety-check this cast, since we got a void*...
    PetscTSSystem *tssys = (PetscTSSystem *)ctx;
    PetscVector<Number> X (x,tssys->comm());
    PetscVector<Number> R (r,tssys->comm());
    tssys->RHSFunction(time,X,R);
    R.close();

    STOP_LOG("RHSFunction()", "PetscTSSolver");
    return 0;
  }


  //---------------------------------------------------------------
  // this function is called by PETSc to evaluate the RHS Jacobian at X ant time t
  PetscErrorCode
  __libmesh_petsc_ts_rhsjacobian(TS ts, PetscReal time, Vec x, Mat jac, Mat jacpre, void *ctx)
  {
    START_LOG("RHSJacobian()", "PetscTSSolver");
    PetscErrorCode ierr=0;

    libmesh_assert(x);
    libmesh_assert(jac);
    libmesh_assert(ctx);

    // No way to safety-check this cast, since we got a void*...
    PetscTSSystem *tssys = (PetscTSSystem *)ctx;
    PetscVector<Number> X     (x,     tssys->comm());
    PetscMatrix<Number> Jac   (jac,   tssys->comm());
    PetscMatrix<Number> Jacpre(jacpre,tssys->comm());
    
    // What do we do if jacpre is NULL?  Should we switch to passing
    // PetscVector* and PetscMatrix* to TSSystem::RHSJacobian() etc.?
    tssys->RHSJacobian(time,X,Jac,Jacpre);
    
    STOP_LOG("RHSJacobian()", "PetscTSSolver");
    return ierr;
  }
  */
  
  //---------------------------------------------------------------
  PetscErrorCode
  __libmesh_petsc_ts_ifunction (TS ts, PetscReal time, Vec x, Vec xdot, Vec f, void *ctx)
  {
    START_LOG("IFunction()", "PetscTSSolver");
    PetscErrorCode ierr = 0;
    
    libmesh_assert(x);
    libmesh_assert(xdot);
    libmesh_assert(f);
    libmesh_assert(ctx);
    
    // No way to safety-check this cast, since we got a void*...
    PetscTSSystem *tssys = (PetscTSSystem *)ctx;
    
    PetscVector<Number> X     (x,     tssys->comm());
    PetscVector<Number> Xdot  (xdot,  tssys->comm());
    PetscVector<Number> F     (f,     tssys->comm());
    
    // evaluate the ifunction
    tssys->IFunction(time,X,Xdot,F);
    
    STOP_LOG("IFunction()", "PetscTSSolver");
    
    // ---------------------- view the vector f ---------------------------
    //PetscViewer vec_viewer;
    //ierr = PetscPrintf(tssys->comm().get() ,"View Vec info: \n");
    //ierr = PetscViewerCreate(tssys->comm().get(), &vec_viewer);
    //ierr = PetscViewerSetType(vec_viewer, PETSCVIEWERASCII);
    //ierr = VecView(f, vec_viewer);
    //ierr = PetscViewerDestroy(&vec_viewer);
    //CHKERRABORT(tssys->comm().get(), ierr);
    // --------------------------------------------------------------------
    
    return ierr;
  }
  

  //---------------------------------------------------------------
  PetscErrorCode
  __libmesh_petsc_ts_ijacobian (TS ts, PetscReal time, Vec x, Vec xdot, PetscReal shift,Mat ijac, Mat ijacpre, void *ctx)
  {
    START_LOG("IJacobian()", "PetscTSSolver");
    PetscErrorCode ierr = 0;
    
    libmesh_assert(x);
    libmesh_assert(xdot);
    libmesh_assert(ijac);
    libmesh_assert(ctx);
    
    // No way to safety-check this cast, since we got a void*...
    PetscTSSystem *tssys = (PetscTSSystem *)ctx;
    
    PetscVector<Number> X     (x,       tssys->comm());
    PetscVector<Number> Xdot  (xdot,    tssys->comm());
    PetscMatrix<Number> IJ    (ijac,    tssys->comm());
    PetscMatrix<Number> IJpre (ijacpre, tssys->comm());
    
    // evaluate the matrices
    tssys->IJacobian(time,X,Xdot,shift,IJ,IJpre);
    
    STOP_LOG("IJacobian()", "PetscTSSolver");
    
    // ---------------------- view the matrix ijac ---------------------------
//    PetscViewer mat_viewer;
//    ierr = PetscPrintf(tssys->comm().get(),"View Mat info: \n"); CHKERRABORT(tssys->comm().get(), ierr);
//    ierr = PetscViewerCreate(tssys->comm().get(), &mat_viewer);  CHKERRABORT(tssys->comm().get(), ierr);
//    ierr = PetscViewerSetType(mat_viewer, PETSCVIEWERASCII);   CHKERRABORT(tssys->comm().get(), ierr);
//    ierr = MatView(ijac, mat_viewer);                          CHKERRABORT(tssys->comm().get(), ierr);
//    ierr = PetscViewerDestroy(&mat_viewer);                    CHKERRABORT(tssys->comm().get(), ierr);
    // -----------------------------------------------------------------------
    
    return ierr;
  }

} // end extern "C"
//---------------------------------------------------------------------



//---------------------------------------------------------------------
// PetscTSSolver methods
template <typename T>
PetscTSSolver<T>::PetscTSSolver (sys_type& tssys, const char* name) :
  ParallelObject(tssys),
  _ts(NULL),
  _system(tssys),
  _name(name),
  _initialized(false),
  _reason(TS_CONVERGED_ITERATING  /*==0*/ ), // Arbitrary initial value...
  _initial_time(0.),
  _max_time(0.),
  _dt(0.),
  _max_steps(0),
  _n_linear_iterations(0),
  _current_nonlinear_iteration_number(0)
{
  // do nothing
}


//---------------------------------------------------------------------
template <typename T>
PetscTSSolver<T>::~PetscTSSolver ()
{
  this->clear ();
}
  
  
//------------------------------------------------------------------
// PetscTSSolver members
#if defined(LIBMESH_HAVE_PETSC)
template <typename T>
UniquePtr<PetscTSSolver<T> > PetscTSSolver<T>::build(sys_type& s)
{
  // Build the appropriate solver
  return UniquePtr<PetscTSSolver<T> >(new PetscTSSolver<T>(s));
}
  
#else // LIBMESH_HAVE_PETSC
template <typename T>
UniquePtr<PetscTSSolver<T> > PetscTSSolver<T>::build(sys_type& s)
{
  libmesh_not_implemented_msg("ERROR: libMesh was compiled without PETSc TS solver support");
}
#endif



//---------------------------------------------------------------------
template <typename T>
void PetscTSSolver<T>::clear ()
{
  if (this->_initialized)
  {
    this->_initialized = false;
    
    PetscErrorCode ierr = 0;
    ierr = TSDestroy(&_ts);   LIBMESH_CHKERRABORT(ierr);

    delete _name;
//    delete _R;
//    delete _J;
//    delete _Jpre;

    // Reset the nonlinear iteration counter.  This information is only relevant
    // *during* the solve().  After the solve is completed it should return to
    // the default value of 0.
    _initial_time = 0.;
    _max_time     = 0.;
    _dt           = 0.;
    _max_steps    = 0;
    _n_linear_iterations                = 0;
    _current_nonlinear_iteration_number = 0;
  }
}


//---------------------------------------------------------------------
template <typename T>
void PetscTSSolver<T>::init ()
{
  START_LOG("init()", "PetscTSSolver");
  
  // Initialize the data structures if not done so already.
  if (!this->_initialized)
  {
    //PetscPrintf(this->comm().get(),"************* Initializing the Petsc TS solver ...... \n");
    this->_initialized  = true;
    PetscErrorCode ierr = 0;

    // Create TS
    ierr = TSCreate(this->comm().get(),&_ts);     LIBMESH_CHKERRABORT(ierr);
    ierr = TSSetProblemType(_ts,TS_NONLINEAR);    LIBMESH_CHKERRABORT(ierr);
    //PetscPrintf(this->comm().get(),"************* --- Petsc TS is created ...... \n");
    
    // Attaching a DM to TS.
//    DM dm;
//    ierr = DMCreate(this->comm().get(), &dm);     LIBMESH_CHKERRABORT(ierr);
//    ierr = DMSetType(dm,DMLIBMESH);               LIBMESH_CHKERRABORT(ierr);
//    ierr = DMlibMeshSetSystem(dm,this->system()); LIBMESH_CHKERRABORT(ierr);

    if (_name)
    {
      std::string prefix = std::string(_name)+std::string("_");
      ierr = TSSetOptionsPrefix(_ts, prefix.c_str()); LIBMESH_CHKERRABORT(ierr);
//      ierr = DMSetOptionsPrefix(dm,_name);            LIBMESH_CHKERRABORT(ierr);
    }
    ierr = TSMonitorSet(_ts, __libmesh_petsc_ts_monitor,&this->system(),NULL); LIBMESH_CHKERRABORT(ierr);
//    ierr = DMSetFromOptions(dm);               LIBMESH_CHKERRABORT(ierr);
//    ierr = DMSetUp(dm);                        LIBMESH_CHKERRABORT(ierr);
//    ierr = TSSetDM(this->_ts, dm);             LIBMESH_CHKERRABORT(ierr);
//    ierr = DMDestroy(&dm);                     LIBMESH_CHKERRABORT(ierr);
    //PetscPrintf(this->comm().get(),"************* --- Petsc TS monitor is set ...... \n");
    
    // Build the vector and matrices
    // TODO: how can we ensure it's a PetscVector*? Is cast_ptr enough?
//    NumericVector<Number> *R    = NumericVector<Number>::build(this->comm()).release();
//    _R = cast_ptr<PetscVector<Number>*>(R);
//    SparseMatrix<Number>  *J    = SparseMatrix<Number>::build(this->comm()).release();
//    _J = cast_ptr<PetscMatrix<Number>*>(J);
//    SparseMatrix<Number>  *Jpre = SparseMatrix<Number>::build(this->comm()).release();
//    _Jpre = cast_ptr<PetscMatrix<Number>*>(Jpre);
//    PetscPrintf(this->comm().get(),"************* --- Petsc TS SparseMatix/Vector are created ...... \n");

    // Set the IFunction
    ierr = TSSetIFunction(_ts,NULL,__libmesh_petsc_ts_ifunction,&this->system());
    LIBMESH_CHKERRABORT(ierr);
    //PetscPrintf(this->comm().get(),"************* --- Petsc TS TSSetIFunction are completed ...... \n");
    
    // Set the IJacobian
    PetscMatrix<Number>& Jac_sys = *cast_ptr<PetscMatrix<Number>*>(this->system().matrix);
    if (this->system().request_matrix("Preconditioner"))
    {
      PetscPrintf(this->comm().get(),"************* --- Preconditioner matrix has been found! ...... \n");
      this->system().request_matrix("Preconditioner")->close();
      PetscMatrix<Number>& PC_sys = *cast_ptr<PetscMatrix<Number>*>( this->system().request_matrix("Preconditioner") );
      ierr = TSSetIJacobian(_ts,Jac_sys.mat(),PC_sys.mat(),__libmesh_petsc_ts_ijacobian,&this->system());
    }
    else
      ierr = TSSetIJacobian(_ts,Jac_sys.mat(),Jac_sys.mat(),__libmesh_petsc_ts_ijacobian,&this->system());
    // end if-else
    LIBMESH_CHKERRABORT(ierr);
    //PetscPrintf(this->comm().get(),"************* --- Petsc TS TSSetIJacobian are completed ...... \n");

    // solution of the system
    PetscVector<Number>& X_sys = *cast_ptr<PetscVector<Number>*>(_system.solution.get());
    ierr = TSSetDuration(_ts, _max_steps, _max_time);   LIBMESH_CHKERRABORT(ierr);
    ierr = TSSetSolution(_ts,X_sys.vec());              LIBMESH_CHKERRABORT(ierr);
    ierr = TSSetInitialTimeStep(_ts,_initial_time,_dt); LIBMESH_CHKERRABORT(ierr);
    ierr = TSSetFromOptions(_ts);                       LIBMESH_CHKERRABORT(ierr);
    //PetscPrintf(this->comm().get(),"************* --- Petsc TSSetFromOptions are completed ...... \n");
    
  } // end if
  
  STOP_LOG("init()", "PetscTSSolver");
}

  
//---------------------------------------------------------------------
template <typename T>
void PetscTSSolver<T>::solve ()
{
  PetscPrintf(this->comm().get(),"************* the Petsc TS solver starts to solve ...... \n");
  START_LOG("solve()", "PetscTSSolver");
  PetscErrorCode ierr=0;
  
  // Attaching a DM to TS.
//  DM dm;
//  ierr = DMCreate(this->comm().get(), &dm);     LIBMESH_CHKERRABORT(ierr);
//  ierr = DMSetType(dm,DMLIBMESH);               LIBMESH_CHKERRABORT(ierr);
//  ierr = DMlibMeshSetSystem(dm,this->system()); LIBMESH_CHKERRABORT(ierr);
//  if (_name)
//  {
//    ierr = DMSetOptionsPrefix(dm,_name);        LIBMESH_CHKERRABORT(ierr);
//  }
//  ierr = DMSetFromOptions(dm);               LIBMESH_CHKERRABORT(ierr);
//  ierr = DMSetUp(dm);                        LIBMESH_CHKERRABORT(ierr);
//  ierr = TSSetDM(this->_ts, dm);             LIBMESH_CHKERRABORT(ierr);
  
  
  // Set the solution
  PetscVector<Number>* PETScX  = cast_ptr<PetscVector<Number>*>(_system.solution.get());
  ierr = TSSolve (_ts,PETScX->vec());     LIBMESH_CHKERRABORT(ierr);

  // Get and store the reason for convergence
  PetscReal         ftime;
  PetscInt          nsteps;
  ierr = TSGetSolveTime(_ts,&ftime);      LIBMESH_CHKERRABORT(ierr);
  ierr = TSGetTimeStepNumber(_ts,&nsteps);LIBMESH_CHKERRABORT(ierr);
  TSGetConvergedReason(_ts, &_reason);    LIBMESH_CHKERRABORT(ierr);
  
  PetscPrintf(this->comm().get(),"************* %s at time %g after %D steps\n",
              TSConvergedReasons[_reason], (double)ftime, nsteps);

  ierr = TSDestroy(&_ts);        LIBMESH_CHKERRABORT(ierr);
  //ierr = DMDestroy(&dm);                     LIBMESH_CHKERRABORT(ierr);
  STOP_LOG("solve()", "PetscTSSolver");
  
  this->system().update();
}

  
//// ... PetscTSSolver::step(), etc.
//
//TSConvergedReason PetscTSSolver::get_converged_reason()
//{
//  PetscErrorCode ierr=0;
//
//  if (this->_initialized)
//    {
//      ierr = TSGetConvergedReason(_ts, &_reason);
//      LIBMESH_CHKERRABORT(ierr);
//    }
//
//  return _reason;
//}


//------------------------------------------------------------------
// Explicit instantiations
template class PetscTSSolver<Number>;

} // namespace libMesh

#endif // #ifdef LIBMESH_HAVE_PETSC
