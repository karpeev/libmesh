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


#include "libmesh/libmesh_config.h"
#include "libmesh/libmesh_common.h"
#include "libmesh/petsc_matrix.h"  // for PetscMatrix
#include "libmesh/petsc_vector.h"

#ifdef LIBMESH_HAVE_PETSC
//#if !PETSC_VERSION_LESS_THAN(3,5,0)

// C++ includes

// Local Includes
#include "libmesh/ts_system.h"
#include "libmesh/petsc_ts_solver.h"
// PETSc includes
#include "petscts.h"


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
    libmesh_assert(x); // make sure x is non-NULL
    TSSystem *tssys = (TSSystem *)ctx;
    // Wrap PETSc Vec as a libMesh::PetscVector<Number> (Number is the scalar type, typically PetscScalar).
    PetscVector<Number> X = PetscVector<Number>(x,tssys->comm());
    // Call the monitor on the PetscVector X, along with the other arguments.
    tssys->monitor(step,time,X);
    return 0;
};



  //---------------------------------------------------------------
  // this function is called by TS to evaluate the rhs function at X
  PetscErrorCode
  __libmesh_petsc_ts_rhsfunction (TS ts, PetscReal t, Vec x, Vec r, void *ctx)
  {
    START_LOG("RHSFunction()", "PetscTSSolver");
    PetscErrorCode ierr=0;

    libmesh_assert(x);
    libmesh_assert(r);
    libmesh_assert(ctx);

    TSSystem *tssys = (TSSystem *)ctx;
    PetscVector<Number> X = PetscVector<Number>(x,tssys->comm());
    PetscVector<Number> R = PetscVector<Number>(x,tssys->comm());
    tssys->RHSFunction(t,X,R);
    R.close();
    STOP_LOG("RHSFunction()", "PetscTSSolver");
    return ierr;
  }


  //---------------------------------------------------------------
  // this function is called by PETSc to evaluate the RHS Jacobian at X ant time t
  PetscErrorCode
   __libmesh_petsc_ts_rhsjacobian(TS ts, PetscReal t, Vec x, Mat jac, Mat jacpre, void *ctx)
  {
    START_LOG("rhsjacobian()", "PetscTSSolver");
    PetscErrorCode ierr=0;

    libmesh_assert(x);
    libmesh_assert(jac);
    libmesh_assert(ctx);

    TSSystem *tssys = (TSSystem *)ctx;
    PetscVector<Number> X      = PetscVector<Number>(x,tssys->comm());
    PetscMatrix<Number> Jac    = PetscMatrix<Number>(jac,tssys->comm());
    PetscMatrix<Number> Jacpre = PetscMatrix<Number>(jacpre,tssys->comm());
    // What do we do if jacpre is NULL?  Should we switch to passing PetscVector* and PetscMatrix* to TSSystem::RHSJacobian() etc.?
    tssys->RHSJacobian(t,X,Jac,Jacpre);
    Jac.close();
    Jacpre.close();
    STOP_LOG("rhsjacobian()", "PetscTSSolver");
    return ierr;
  }

  // this function is called by TS to evaluate the ifunction at X
  PetscErrorCode
  __libmesh_petsc_ts_ifunction (TS ts, PetscReal t, Vec x, Vec xdot, Vec r, void *ctx)
  {
    START_LOG("IFunction()", "PetscTSSolver");
    PetscErrorCode ierr=0;

    libmesh_assert(x);
    libmesh_assert(xdot);
    libmesh_assert(r);
    libmesh_assert(ctx);

    TSSystem *tssys = (TSSystem *)ctx;
    PetscVector<Number> X    = PetscVector<Number>(x,tssys->comm());
    PetscVector<Number> Xdot = PetscVector<Number>(x,tssys->comm());
    PetscVector<Number> R    = PetscVector<Number>(x,tssys->comm());
    tssys->IFunction(t,X,Xdot,R);
    R.close();
    STOP_LOG("IFunction()", "PetscTSSolver");
    return ierr;
  }

  // this function is called by PETSc to evaluate the IJacobian at X ant time t
  PetscErrorCode
  __libmesh_petsc_ts_ijacobian(TS ts, PetscReal t, Vec x, Vec xdot, PetscReal shift, Mat jac, Mat jacpre, void *ctx)
  {
    START_LOG("IJacobian()", "PetscTSSolver");
    PetscErrorCode ierr=0;

    libmesh_assert(x);
    libmesh_assert(xdot);
    libmesh_assert(jac);
    libmesh_assert(ctx);

    TSSystem *tssys = (TSSystem *)ctx;
    PetscVector<Number> X      = PetscVector<Number>(x,tssys->comm());
    PetscVector<Number> Xdot   = PetscVector<Number>(xdot,tssys->comm());
    PetscMatrix<Number> Jac    = PetscMatrix<Number>(jac,tssys->comm());
    PetscMatrix<Number> Jacpre = PetscMatrix<Number>(jacpre,tssys->comm());
    // What do we do if jacpre is NULL?  Should we switch to passing PetscVector* and PetscMatrix* to TSSystem::RHSJacobian() etc.?
    tssys->IJacobian(t,X,Xdot,shift,Jac,Jacpre);
    Jac.close();
    Jacpre.close();
    STOP_LOG("IJacobian()", "PetscTSSolver");
    return ierr;
  }
} // end extern "C"
//---------------------------------------------------------------------



//---------------------------------------------------------------------
// PetscTSSolver methods
PetscTSSolver::PetscTSSolver (TSSystem& tssys,const char* name) :
    ParallelObject(tssys),
    _ts(NULL),
    _tssys(tssys),
    _name(name),
    _initialized(false),
    _reason(TS_CONVERGED_ITERATING/*==0*/) // Arbitrary initial value...
{
}


PetscTSSolver::~PetscTSSolver ()
{
  this->clear ();
}


void PetscTSSolver::clear ()
{
  if (this->_initialized)
    {
      this->_initialized = false;

      PetscErrorCode ierr=0;
      // ... destroy TS etc.

      delete _R;
      delete _J;
      delete _Jpre;

      // Reset the nonlinear iteration counter.  This information is only relevant
      // *during* the solve().  After the solve is completed it should return to
      // the default value of 0.
    }
}



void PetscTSSolver::init ()
{
  // Initialize the data structures if not done so already.
  if (!this->_initialized)
    {
      this->_initialized = true;
      PetscErrorCode ierr=0;

      ierr = TSCreate(this->comm().get(),&_ts); LIBMESH_CHKERRABORT(ierr);

      if (_name) {
	std::string prefix = std::string(_name)+std::string("_");
	ierr = TSSetOptionsPrefix(_ts, prefix.c_str()); LIBMESH_CHKERRABORT(ierr);
      }
      ierr = TSMonitorSet (_ts, __libmesh_petsc_ts_monitor,this,NULL); LIBMESH_CHKERRABORT(ierr);
      ierr = TSMonitorSet (_ts, __libmesh_petsc_ts_monitor,this,NULL); LIBMESH_CHKERRABORT(ierr);

      // We need TSSystem to construct a suitably-sized vector.
      // TODO: how can we ensure it's a PetscVector*? Is cast_ptr enough?
      NumericVector<Number> *R   = _tssys.create_vector();
      _R = cast_ptr<PetscVector<Number>*>(R);

      SparseMatrix<Number>  *J      = _tssys.create_matrix();
      _J = cast_ptr<PetscMatrix<Number>*>(J);

      SparseMatrix<Number>  *Jpre   = _tssys.create_matrix();
      _Jpre = cast_ptr<PetscMatrix<Number>*>(Jpre);

      ierr = TSSetRHSFunction(_ts,_R->vec(),__libmesh_petsc_ts_rhsfunction,this); LIBMESH_CHKERRABORT(ierr);

      ierr = TSSetRHSJacobian(_ts,_J->mat(),_Jpre->mat(),__libmesh_petsc_ts_rhsjacobian,this); LIBMESH_CHKERRABORT(ierr);

      ierr = TSSetIFunction(_ts,_R->vec(),__libmesh_petsc_ts_ifunction,this); LIBMESH_CHKERRABORT(ierr);

      ierr = TSSetIJacobian(_ts,_J->mat(),_Jpre->mat(),__libmesh_petsc_ts_ijacobian,this); LIBMESH_CHKERRABORT(ierr);

      ierr = TSSetFromOptions(_ts); LIBMESH_CHKERRABORT(ierr);

      this->_initialized = true;
    }
}


void PetscTSSolver::solve (NumericVector<Number>& X)
{
  libmesh_assert(X);
  START_LOG("solve()", "PetscTSSolver");

  // Make sure the data passed in are really of Petsc types
  PetscVector<Number>* PETScX  = cast_ptr<PetscVector<Number>*>(&X);

  PetscErrorCode ierr=0;
  ierr = TSSolve (_ts,PETScX->vec()); LIBMESH_CHKERRABORT(ierr);

  // Get and store the reason for convergence
  ierr = TSGetConvergedReason(_ts, &_reason); LIBMESH_CHKERRABORT(ierr);

  STOP_LOG("solve()", "PetscTSSolver");
}

// ... PetscTSSolver::step(), etc.

TSConvergedReason PetscTSSolver::get_converged_reason()
{
  PetscErrorCode ierr=0;

  if (this->_initialized)
    {
      ierr = TSGetConvergedReason(_ts, &_reason); LIBMESH_CHKERRABORT(ierr);
    }

  return _reason;
}

//------------------------------------------------------------------

} // namespace libMesh


//#endif // if !PETSC_VERSION_LESS_THAN(3,5,0)
#endif // #ifdef LIBMESH_HAVE_PETSC
