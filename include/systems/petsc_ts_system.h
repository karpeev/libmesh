//
//  petsc_ts_system.h
//  
//
//  Created by Xujun Zhao on 5/5/15.
//
//

#ifndef ____petsc_ts_system__
#define ____petsc_ts_system__

#include <stdio.h>



// Local Includes
#include "libmesh/nonlinear_implicit_system.h"

// C++ includes


#include "petsc_ts_solver.h"

namespace libMesh
{
  
  
/**
 * This class provides a time-dependent system class.
 * This is an derived class from NonlinearImplicitSystem 
 * so that it can directly take advantage of infrastructures
 * such as mesh, element, dofmap, etc. through multiple
 * inheritance of its parent class to construct the matrices
 * and vectors needed for PetscTSSolver.
 *
 * @author Xujun Zhao, Hong Zhang, Dmitry Karpeyev, 2015
 */

// ------------------------------------------------------------


class PetscTSSystem : public NonlinearImplicitSystem
{
public:
  
  /**
   * Constructor.  Optionally initializes required
   * data structures.
   */
  
  PetscTSSystem (EquationSystems& es,
                 const std::string& name,
                 const unsigned int number);
  
  
  /**
   * Destructor.
   */
  virtual ~PetscTSSystem ();
  
  
  /**
   * The type of system.
   */
  typedef PetscTSSystem sys_type;
  
  
  /**
   * The type of the parent.
   */
  typedef NonlinearImplicitSystem Parent;
  
  
  /**
   * @returns a clever pointer to the system.
   */
  sys_type & system () { return *this; }
  
  
  /**
   * Clear all the data structures associated with
   * the system.
   */
  virtual void clear ();
  
  
  /**
   * Initializes the member data fields associated with the system.
   */
  virtual void init ();
  
  
  /**
   * Reinitializes the member data fields associated with the system.
   */
  virtual void reinit ();
 
 
  /**
   * Set parameters associated with the ts solver.
   */
  void set_solver_parameters ();
  
  
  /**
   * Assembles & solves the ts system F( t, u, u_t ) = 0.
   */
  virtual void solve ();
  
  
  /**
   * @returns \p "PetscTSSystem".  Helps in identifying
   * the system type in an equation system file.
   */
  virtual std::string system_type () const { return "PetscTSSystem"; }
  
  
  /**
   * The \p PetscTSSolver defines an interface used to
   * solve the petsc_ts_system.
   */
  UniquePtr<PetscTSSolver<Number> > ts_solver;
  
  // RSHFunction computes the R in M\dot X = R(X,t).
  // Implementations might need to localize X to a ghosted version.
  // virtual void RHSFunction (Real time,
  //                          const NumericVector<Number>& X,
  //                          NumericVector<Number>& R)=0;
  
  
  // virtual void RHSJacobian (Real time,
  //                          const NumericVector<Number>& X,
  //                          SparseMatrix<Number>& J,
  //                          SparseMatrix<Number> &Jpre)=0;
  
  
  // F(t,U,U_t)
  virtual void IFunction (Real time,
                          const NumericVector<Number>& X,
                          const NumericVector<Number>& Xdot,
                          NumericVector<Number>& F);
  
  // compute the matrix dF/dU + a*dF/dU_t where F(t,U,U_t)
  virtual void IJacobian (Real time,
                          const NumericVector<Number>& X,
                          const NumericVector<Number>& Xdot,
                          Real shift,
                          SparseMatrix<Number>& IJ,
                          SparseMatrix<Number>& IJpre);
  
  
  virtual void monitor (int  step, Real time,
                        NumericVector<Number>& X);
  
  // ... and other callbacks required by (Petsc)TSSolver: pre/postsolve(),adjoint-related stuff, etc.
  // The optional callback methods should be noops, rather than purely abstract
  // so that the user doesn't have to implement a bunch of dummy methods (see IFunction and IJacobian above).
  
  
protected:
  
};
  
  
  
} // namespace libMesh

// ------------------------------------------------------------
// TSSystem inline methods



#endif /* defined(____petsc_ts_system__) */
