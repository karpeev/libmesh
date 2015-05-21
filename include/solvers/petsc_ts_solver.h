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

#ifdef LIBMESH_HAVE_PETSC

// Local includes
#include "libmesh/libmesh.h"
#include "libmesh/petsc_macro.h"
#include "libmesh/libmesh_common.h"
#include "libmesh/reference_counted_object.h"
#include "libmesh/parallel_object.h"
//#include "libmesh/auto_ptr.h"

#include "libmesh/petsc_ts_system.h"

// PETSc includes
EXTERN_C_FOR_PETSC_BEGIN
# include <petscts.h>
# include <petscdm.h>
EXTERN_C_FOR_PETSC_END

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
class PetscTSSystem;
template <typename T> class SparseMatrix;
template <typename T> class NumericVector;
template <typename T> class PetscMatrix;
template <typename T> class PetscVector;
  
/**
 * This class provides a uniform interface for PETSc TS solvers that are
 * compatible with the \p libMesh.
 *
 * @author Xujun Zhao, Hong Zhang, Dmitry Karpeyev, 2015
 */

template <typename T>
class PetscTSSolver :   public ReferenceCountedObject<PetscTSSolver<T> >,
                        public ParallelObject
{
public:
  /**
   * The type of system
   */
  typedef PetscTSSystem sys_type;
  
  /**
  * Constructor.
  * Name will be used as the options prefix.
  */
  explicit
  PetscTSSolver(sys_type& sys, const char* name = NULL);

  /**
  * Destructor.
  */
  ~PetscTSSolver ();
  
  /**
   * Builds a \p PetscTSSolver
   */
  static UniquePtr<PetscTSSolver<T> > build(sys_type& s);
  
  /**
  * Initialize data structures if not done so already.
  */
  virtual void init ();

  /**
   * @returns true if the data structures are
   * initialized, false otherwise.
   */
  bool initialized () const { return _initialized; }
  
  /**
  * Release all memory and clear data structures.
  */
  virtual void clear ();

  /**
  * Returns the raw PETSc TS context pointer.
  * This is used as a last resort, if the above methods do not provide the necessary
  * functionality to configure the underlying TS.
  */
  TS ts() { this->init(); return _ts; }
  
  /**
   * @returns a constant reference to the system we are solving.
   */
  const sys_type & system () const { return _system; }
  
  /**
   * @returns a writeable reference to the system we are solving.
   */
  sys_type & system () { return _system; }
  
  /**
  * Set the time range [t0, max_t](initial time and maximum time)
  */
  void set_duration(const Real t0, const Real max_t)
  {  _initial_time = t0; _max_time = max_t; }

  /**
  * Set time step options: size of timestep dt and number of timesteps.
  */
  void set_timestep(const Real dt, const unsigned int steps)
  { _dt = dt; _max_steps = steps; }
  
  // ... and other functions that set the parameters of the underlying TS.

  /**
  * Call the Petsc TS solver to integrate from initial to final time.
  */
  void solve ();

//  /**
//  * Call the Petsc TS solver to take one step.
//  */
//  void step();
  
  /**
  * Returns the currently-available (or most recently obtained, if the TS object has
  * been destroyed) convergence reason.  Refer to PETSc docs for the meaning of different
  * TSConvergedReasons.
  */
  TSConvergedReason get_converged_reason(){ return _reason; }

  // ... and other functions querying the underlying TS state

protected:
  /**
  * TS solver context
  */
  TS _ts;

  /**
  * TSSystem
  */
  sys_type& _system;

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

  /**
   * the inital time for the ts solver
   */
  PetscReal _initial_time;
  
  /**
   * the duration time for the ts solver
   */
  PetscReal _max_time;
  
  /**
   * the size of the timestep for the ts solver
   */
  PetscReal _dt;
  
  /**
   * the max time step for the ts solver
   */
  PetscInt _max_steps;
  
  /**
  * Stores the total number of linear iterations from the last solve.
  */
  PetscInt _n_linear_iterations;

  /**
  * Stores the current nonlinear iteration number
  */
  PetscInt _current_nonlinear_iteration_number;

  // Current timestep value should go into _tsys, if anywhere.
//  PetscVector<Number> *_R;
//  PetscMatrix<Number> *_J, *_Jpre;

  // friend class
  friend PetscErrorCode __libmesh_petsc_ts_rhsfunction (TS ts, PetscReal t, Vec x, Vec r, void *ctx);
  friend PetscErrorCode __libmesh_petsc_ts_rhsjacobian (TS ts, PetscReal t, Vec x, Mat jac, Mat jacpre, void *ctx);
  friend PetscErrorCode __libmesh_petsc_ts_ifunction (TS ts, PetscReal t, Vec x, Vec xdot, Vec f, void *ctx);
  friend PetscErrorCode __libmesh_petsc_ts_ijacobian (TS ts, PetscReal t, Vec x, Vec xdot, PetscReal shift,Mat ijac, Mat ijacpre, void *ctx);
  friend PetscErrorCode __libmesh_petsc_ts_monitor(TS ts, PetscInt step, PetscReal time, Vec x, void *ctx);

};  // end of class PetscTSSolver

} // namespace libMesh

#endif // #ifdef LIBMESH_HAVE_PETSC
#endif // LIBMESH_PETSC_TS_SOLVER_H
