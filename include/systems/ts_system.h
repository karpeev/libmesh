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



#ifndef LIBMESH_TS_SYSTEM_H
#define LIBMESH_TS_SYSTEM_H

// Local Includes
#include "libmesh/implicit_system.h"

// C++ includes

namespace libMesh
{


/**
 * This class provides a time-dependent system class.
 * This is an abstract class that codifies the interface
 * used by TSSolver.  It should be inherited from
 * by a user, who might combine this class with another
 * such as the NonlinearImplicitSystem through multiple
 * inheritance.  NonlinearImplicitSystem would then provide
 * the infrastructure (mesh traversal, dof_map, etc.) to
 * implement the TSSystem interface.
 */

// ------------------------------------------------------------
// TSSystem class definition

class TSSystem : public ImplicitSystem
{
public:

  /**
   * Constructor.  Optionally initializes required
   * data structures.
   */
  TSSystem (EquationSystems& es,
            const std::string& name,
            const unsigned int number);

  /**
   * Destructor.
   */
  virtual ~TSSystem ();

  typedef ImplicitSystem Parent;

  // RSHFunction computes the R in M\dot X = R(X,t).
  // Implementations might need to localize X to a ghosted version.
  virtual void RHSFunction  (Real time, const NumericVector<Number>& X, NumericVector<Number>& R){};
  virtual void RHSJacobian  (Real time, const NumericVector<Number>& X, SparseMatrix<Number>& J,SparseMatrix<Number> &Jpre){};
  virtual void IFunction    (Real time, const NumericVector<Number>& X, const NumericVector<Number>& Xdot,NumericVector<Number>& F){};
  virtual void IJacobian    (Real time, const NumericVector<Number>& X, const NumericVector<Number>& Xdot,Real shift,SparseMatrix<Number>& IJ,SparseMatrix<Number>& IJpre){};
  virtual void monitor      (int  step, Real time, NumericVector<Number>& X){} ;
  // ... and other callbacks required by (Petsc)TSSolver: pre/postsolve(),adjoint-related stuff, etc.
  // The optional callback methods should be noops, rather than purely abstract so that the user doesn't have to implement a bunch of dummy methods (see IFunction and IJacobian above).

  // These methods must be provided by the system in order to be able to set up the solver.
  // A specialization of this class might inherit them (or delegate to other inherited methods)
  // from another System class via multiple inheritance.
  virtual NumericVector<Number>* create_vector()=0;
  virtual SparseMatrix<Number>*  create_matrix()=0;

protected:
};



} // namespace libMesh

// ------------------------------------------------------------
// TSSystem inline methods


#endif // LIBMESH_TS_SYSTEM_H
