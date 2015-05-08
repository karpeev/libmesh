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



#ifndef LIBMESH_PETSC_TS_SOLVER_H
#define LIBMESH_PETSC_TS_SOLVER_H

#include "libmesh/libmesh_config.h"

// Petsc include files.
#ifdef LIBMESH_HAVE_PETSC

// Local includes
//#include "libmesh/petsc_ts_system.h"
#include "libmesh/petsc_macro.h"
#include "libmesh/parallel_object.h"
#if !PETSC_RELEASE_LESS_THAN(3,5,0)

#include "libmesh/ts_system.h"

// PETSc includes
EXTERN_C_FOR_PETSC_BEGIN
# include <petscts.h>
EXTERN_C_FOR_PETSC_END

// C++ includes

namespace libMesh
{
/**
 * Allow users access to these functions in case they want to reuse them.
 * Note that users shouldn't need access to these most of the time
 * as they are used internally by this object.
 */
  extern "C" 
  {
  PetscErrorCode __libmesh_petsc_ts_monitor(TS ts, PetscInt step, PetscReal time, Vec x, void *ctx);
  PetscErrorCode __libmesh_petsc_ts_rhsfunction (TS ts, PetscReal t, Vec x, Vec r, void *ctx);
  PetscErrorCode __libmesh_petsc_ts_rhsjacobian (TS ts, PetscReal t, Vec x, Mat jac, Mat jacpre, void *ctx);
  PetscErrorCode __libmesh_petsc_ts_ifunction (TS ts, PetscReal t, Vec x, Vec xdot, Vec f, void *ctx);
  PetscErrorCode __libmesh_petsc_ts_ijacobian (TS ts, PetscReal t, Vec x, Vec xdot, PetscReal shift,Mat ijac, Mat ijacpre, void *ctx);
  }
  // Forward Declarations
  //class TSSystem;
  //template <typename T> class SparseMatrix;
  //template <typename T> class NumericVector;
  //template <typename T> class PetscMatrix;
  //template <typename T> class PetscVector;

class PetscTSSolver : public ParallelObject 
{
public:
   /**
    * Constructor.
    * Name will be used as the options prefix.
    */
   PetscTSSolver(TSSystem& tssys, const char* name = NULL);
  /**
   * Destructor.
   */
  ~PetscTSSolver ();


  /**
   * Initialize data structures if not done so already.
   */
  virtual void init ();

  /**
   * Release all memory and clear data structures.
   */
  virtual void clear ();

  /**
   * Set initial time.
   */
  void set_initial_time(Real t0);

  /**
   * Set duration of integration.
   */
  void set_duration(Real T);

  // ... and other functions that set the parameters of the underlying TS.


  /**
   * Call the Petsc TS solver to integrate from initial to final time.
   */
  void solve(NumericVector<Number>& X);

  /**
   * Call the Petsc TS solver to take one step.
   */
  void step();

  /**
   * Returns the currently-available (or most recently obtained, if the TS object has
   * been destroyed) convergence reason.  Refer to PETSc docs for the meaning of different
   * TSConvergedReasons.
   */
  TSConvergedReason get_converged_reason();

  // ... and other functions querying the underlying TS state

  /**
   * Returns the raw PETSc TS context pointer.
   * This is used as a last resort, if the above methods do not provide the necessary
   * functionality to configure the underlying TS.
   */
  TS ts() { this->init(); return _ts; }

protected:
  /**
   * TS solver context
   */
  TS _ts;

  /**
   * TSSystem
   */
  TSSystem& _tssys;

  /**
   * Name of the object; it is also used to build the PETSc options prefix: <name>_
   */
  const char* _name;

  /**
   *  init() has been called and the solver is ready to use.
   */
  bool _initialized;

  /**
   * Store the reason for TS convergence/divergence for use even after the _ts
   * has been cleared. Note that
   * this value is therefore necessarily *not* cleared by the clear() function.
   */
  TSConvergedReason _reason;

  // Current timestep value should go into _tsys, if anywhere.

  PetscVector<Number> *_R;
  PetscMatrix<Number> *_J, *_Jpre;


};



} // namespace libMesh

#endif // if !PETSC_RELEASE_LESS_THAN(3,5,0)
#endif // #ifdef LIBMESH_HAVE_PETSC
#endif // LIBMESH_PETSC_TS_SOLVER_H
