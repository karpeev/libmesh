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



#ifndef LIBMESH_DM_H
#define LIBMESH_DM_H

// Local Includes
#include "libmesh/libmesh_common.h"
#include "libmesh/system.h"
#include "libmesh/equation_systems.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"

// C++ includes
#include <cstddef>
#include <set>
#include <vector>

namespace libMesh
{

/**
 * This is a wrapper/delegator class for systems that enables
 * decomposition by subdomain, field or by scale (multigrid).
 * While this system mimics PETSc's DM interface, its definition
 * is entirely generic and can, in principle, be used by other
 * solver solver classes.
 *
 * @author Dmitry Karpeyev, 2015
 */

// ------------------------------------------------------------
// DM class definition
class DM : public ReferenceCountedObject<DM>,
               public ParallelObject
{
public:

  /**
   * Constructor.
   */
  DM (EquationSystems& es,
          const std::string& sys_name);


  /**
   * Destructor.
   */
  virtual ~DM ();

  /**
   * Returns the underlying EquationSystems object.
   */
  EquationSystems& equation_systems();

  /**
   * Returns the name of the underlying System object.
   */
  std::string system_name();



  /**
   * Creates an interpolation to another, generally finer DM
   */
  UniquePtr<DM>
    coarsen();

  /**
   * Initializes the matrix sparsity structure based on the mesh connectivity,
   * locally-owned part, variables and their type.
   */
  void initMatrix(SparseMatrix<Number>& mat); // The idea is to keep this independent of the solver-specific matrix type.

  /**
   * Initializes the vector layout based on the local part of the mesh,
   * system variables, their type, and in accordance with the \p ParallelType.
   */
  void initVector(NumericVector<Number>& vec, const ParallelType ptype);

  /**
   * Creates an interpolation to another, generally finer DMSystem
   */
  void assembleInterpolation(const DM& fine, SparseMatrix<Number> &interp); // or should it be 'coarse'?

  /**
   * Function that assembles the system residual.  Should be implemented by the user in a derived class.
   */
  // virtual void assembleResidual(const NumericVector<Number>& X, NumericVector<Number>& R) = 0;

  // virtual void assembleJacobians(const NumericVector<Number>& X, SparseMatrix<Number>& J, SparseMatrix<Number>& J_pre) = 0;

 protected:

  EquationSystems& _es;

  std::string       _sys_name;

  System&           _sys;
};



// ------------------------------------------------------------
// DM inline methods
inline
  EquationSystems& DM::equation_systems()
{
  return _es;
}

inline
  std::string DM::system_name()
{
  return _sys_name;
}



} // namespace libMesh

#endif // LIBMESH_DM_H
