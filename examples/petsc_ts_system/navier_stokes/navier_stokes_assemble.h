//
//  navier_stokes_assemble.h
//  
//
//  Created by Xujun Zhao on 2/23/15.
//
//

#ifndef ____navier_stokes_assemble__
#define ____navier_stokes_assemble__

#include <stdio.h>



// C++ Includes
#include <iostream>
#include <cstring>
#include <utility>
#include <vector>
#include <map>

#include "libmesh/libmesh.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/equation_systems.h"
#include "libmesh/point.h"
#include "libmesh/petsc_ts_system.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// ============================================================================================
// ============================================================================================
// Initialization functions are optional for systems.
// If an init function is not provided then the default (0) solution is provided.
void init_navier_stokes (EquationSystems& es,
                         const std::string& system_name);



// ============================================================================================
// ============================================================================================
// assemble the IFunction for time-dependent navier-stokes: F(t,U,U_t)
void assemble_ifunction (libMesh::EquationSystems& es,
                         const std::string& system_name,
                         const Real& time,
                         const NumericVector<Number>& X,
                         const NumericVector<Number>& Xdot);



// ============================================================================================
// ============================================================================================
// assemble the IJacobian for time-dependent navier-stokes: dF/dU + a*dF/dU_t
void assemble_ijacobian (libMesh::EquationSystems& es,
                         const std::string& system_name,
                         const Real& time,
                         const Real& shift,
                         const NumericVector<Number>& X,
                         const NumericVector<Number>& Xdot);



// ---------------------------------------------------------------------------------------
// compute the rhs vector of an element resulting from the body force and traction.
// In this case, there is no body force, but only pressure jump along x-direction(inlet/outlet)
// a1*F(n+1) + a0*F(n).   see derived equation (49)
// can not use AutoPtr<FEBase> as arguments, AutoPtr will destroy the object.
void compute_element_rhs(const MeshBase& mesh,
                         const Elem* elem,
                         const unsigned int n_u_dofs,
                         const unsigned int n_p_dofs,
                         FEBase& fe_vel,
                         FEBase& fe_pres,
                         const bool& periodicity,
                         const Real& time, 
                         DenseSubVector<Number>& Fu,
                         DenseSubVector<Number>& Fv,
                         DenseSubVector<Number>& Fw,
                         DenseSubVector<Number>& Fp);



// ---------------------------------------------------------------------------------------
// apply Dirichlet boundary conditions by penalty method
// note, this is an element-wise operation, so should be put inside the elem-loop
void apply_bc_by_penalty(const MeshBase& mesh,
                         const Elem* elem,
                         const Real& time,
                         const std::string& matrix_or_vector,
                         DenseSubMatrix<Number>& Kuu,
                         DenseSubMatrix<Number>& Kvv,
                         DenseSubMatrix<Number>& Kww,
                         DenseSubMatrix<Number>& Kpp,
                         DenseSubVector<Number>& Fu,
                         DenseSubVector<Number>& Fv,
                         DenseSubVector<Number>& Fw,
                         DenseSubVector<Number>& Fp);



// ---------------------------------------------------------------------------------------
Real boundary_pressure(const Point& pt,
                       const Real& t,
                       const std::string& which_side);



// ---------------------------------------------------------------------------------------
// output the velocity u profile at the cross section x = x_pos
int write_out_section_profile(const libMesh::EquationSystems& es,
                               const Real x_pos,
                               const unsigned int y_div,
                               const unsigned int n_step);


// ---------------------------------------------------------------------------------------
// output the history of velocity u profile at the cross section x = x_pos
int write_out_section_profile_history(const libMesh::EquationSystems& es,
                                      const Real x_pos,
                                      const unsigned int y_div,
                                      const unsigned int n_step,
                                      const Real time);

class NSPetscTSSystem : public PetscTSSystem
{
public:
  using PetscTSSystem::PetscTSSystem;
  void IFunction (Real time,
                  const NumericVector<Number>& X,
                  const NumericVector<Number>& Xdot,
                  NumericVector<Number>& F);

// ---------------------------------------------------------------------
// compute the matrix dF/dU + a*dF/dU_t where F(t,U,U_t)
  void IJacobian (Real time,
                  const NumericVector<Number>& X,
                  const NumericVector<Number>& Xdot,
                  Real shift,
                  SparseMatrix<Number>& IJ,
                  SparseMatrix<Number>& IJpre);

// ---------------------------------------------------------------------
  void monitor (int  step, Real time,
                NumericVector<Number>& X);
};

#endif /* defined(____navier_stokes_assemble__) */
