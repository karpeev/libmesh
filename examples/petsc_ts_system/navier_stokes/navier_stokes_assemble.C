//
//  navier_stokes_assemble.C
//  
//
//  Created by Xujun Zhao on 2/23/15.
//
//


// C++ include files that we need
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string.h>
#include <math.h>
#include "libmesh/libmesh_common.h"
// Basic include file needed for the mesh functionality.
#include "libmesh/mesh.h"
#include "libmesh/fe.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/transient_system.h"
#include "libmesh/boundary_info.h"
//#include "libmesh/quadrature_gauss.h"
#include "libmesh/exodusII_io.h"

// For systems of equations the DenseSubMatrix and DenseSubVector provide convenient ways
// for assembling the element matrix and vector on a component-by-component basis.
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dense_submatrix.h"
#include "libmesh/dense_subvector.h"

// include the header file
#include "libmesh/petsc_ts_system.h"
#include "navier_stokes_assemble.h"
#include "GeomTools.h"





// ============================================================================================
// ============================================================================================
void init_navier_stokes (libMesh::EquationSystems& es,
                         const std::string& system_name)
{
  // It is a good idea to make sure we are assembling the proper system.
  libmesh_assert_equal_to (system_name, "Navier-Stokes");
  
  // Get a reference to the Stokes system object.
  TransientLinearImplicitSystem & navier_stokes_system =
      es.get_system<TransientLinearImplicitSystem> ("Navier-Stokes");
  
  // Project initial conditions at time 0
  es.parameters.set<Real> ("time") = navier_stokes_system.time = 0;
//  system.project_solution( exact_value, NULL, es.parameters );
}




// ============================================================================================
// ============================================================================================
void assemble_ifunction (libMesh::EquationSystems& es,
                         const std::string& system_name,
                         const Real& time,
                         const NumericVector<Number>& X,
                         const NumericVector<Number>& Xdot)
{
  // It is a good idea to make sure we are assembling the proper system.
  libmesh_assert_equal_to (system_name, "Navier-Stokes");
  
  // Get a constant reference to the mesh object.
  const MeshBase& mesh    = es.get_mesh();
  const unsigned int dim  = mesh.mesh_dimension();
  
  // Get a reference to the Stokes system object.
  PetscTSSystem & navier_stokes_system = es.get_system<PetscTSSystem> ("Navier-Stokes");
  
  // Numeric ids corresponding to each variable in the system
  const unsigned int u_var = navier_stokes_system.variable_number ("u");  // u_var = 0
  const unsigned int v_var = navier_stokes_system.variable_number ("v");  // u_var = 1
  unsigned int w_var = 0;
  if(dim==3)   w_var = navier_stokes_system.variable_number ("w");        // w_var = 2
  const unsigned int p_var = navier_stokes_system.variable_number ("p");  // p_var = 2(dim=2); = 3(dim=3)
  
  // Get the Finite Element type for "u" and "p"
  FEType fe_vel_type  = navier_stokes_system.variable_type(u_var);
  FEType fe_pres_type = navier_stokes_system.variable_type(p_var);
  
  // Build a Finite Element object for "u" and "p"
  AutoPtr<FEBase> fe_vel  (FEBase::build(dim, fe_vel_type));
  AutoPtr<FEBase> fe_pres (FEBase::build(dim, fe_pres_type));
  
  // A Gauss quadrature rule for numerical integration.
  QGauss qrule (dim, fe_vel_type.default_quadrature_order());
  
  // Tell the finite element objects to use our quadrature rule.
  fe_vel->attach_quadrature_rule (&qrule);
  fe_pres->attach_quadrature_rule (&qrule);
  
  // Here we define some references to cell-specific data that
  // will be used to assemble the linear system.
  //
  // The element Jacobian * quadrature weight at each integration point.
  const std::vector<Real>& JxW                        = fe_vel->get_JxW();
  const std::vector<std::vector<Real> >& phi          = fe_vel->get_phi();
  const std::vector<std::vector<RealGradient> >& dphi = fe_vel->get_dphi();
  
  // The element shape functions for the "p" evaluated at the q-points.
  const std::vector<std::vector<Real> >& psi          = fe_pres->get_phi();
  
  // Define data structures to contain the element matrix and rhs vector.
  DenseMatrix<Number> Me;   // element mass matrix
  DenseSubMatrix<Number>
  Muu(Me), Muv(Me), Muw(Me), Mup(Me),
  Mvu(Me), Mvv(Me), Mvw(Me), Mvp(Me),
  Mwu(Me), Mwv(Me), Mww(Me), Mwp(Me),
  Mpu(Me), Mpv(Me), Mpw(Me), Mpp(Me);
  
  DenseMatrix<Number> Ke;   // element convection & diffusion matrix
  DenseSubMatrix<Number>
  Kuu(Ke), Kuv(Ke), Kuw(Ke), Kup(Ke),
  Kvu(Ke), Kvv(Ke), Kvw(Ke), Kvp(Ke),
  Kwu(Ke), Kwv(Ke), Kww(Ke), Kwp(Ke),
  Kpu(Ke), Kpv(Ke), Kpw(Ke), Kpp(Ke);
  
  DenseVector<Number> Fe;   // element force vector
  DenseSubVector<Number>  Fu(Fe),    Fv(Fe),    Fw(Fe),    Fp(Fe);
  
  DenseVector<Number> Ve, Vedot;   // element velocity and v_dot vector
  DenseSubVector<Number>  Vu(Ve),    Vv(Ve),    Vw(Ve),    Vp(Ve);
  DenseSubVector<Number>  Vudot(Vedot),  Vvdot(Vedot),  Vwdot(Vedot),  Vpdot(Vedot);
  
  // A reference to the \p DofMap object for this system.
  const DofMap & dof_map = navier_stokes_system.get_dof_map();
  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_u, dof_indices_v, dof_indices_w, dof_indices_p;

  // system parameters
  const Real mu           =  es.parameters.get<Real> ("viscosity");
  const bool periodicity  =  es.parameters.get<bool> ("periodicity");
  
  // loop over all the elements in the mesh that live on the local processor.
  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();
  for ( ; el != end_el; ++el)
  {
    // Store a pointer to the element we are currently working on.
    const Elem* elem = *el;
//    printf("************* assemble_ifunction in element %u/%u \n",
//           elem->id(), mesh.n_elem() );
    
    // Get the degree of freedom indices for the current element.
    dof_map.dof_indices (elem, dof_indices);
    dof_map.dof_indices (elem, dof_indices_u, u_var);
    dof_map.dof_indices (elem, dof_indices_v, v_var);
    dof_map.dof_indices (elem, dof_indices_p, p_var);
    
    const unsigned int n_dofs   = dof_indices.size();
    const unsigned int n_u_dofs = dof_indices_u.size();
    const unsigned int n_v_dofs = dof_indices_v.size();
    const unsigned int n_p_dofs = dof_indices_p.size();
    unsigned int n_w_dofs = 0;
    if(dim==3)
    {
      dof_map.dof_indices (elem, dof_indices_w, w_var);
      n_w_dofs = dof_indices_w.size();
    }
    
    // Compute the element-specific data for the current element.
    // This involves computing the location of the quadrature points
    // and the shape functions (phi, dphi) for the current element.
    fe_vel->reinit  (elem);
    fe_pres->reinit (elem);
    
    // Zero the element matrix and right-hand side before summing them.
    Me.resize (n_dofs, n_dofs);
    Ke.resize (n_dofs, n_dofs);
    Fe.resize (n_dofs);
    Ve.resize (n_dofs);   Vedot.resize (n_dofs);
    
    // Reposition the submatrices...  The idea is this:
    //         -           -          -  -
    //        | Kuu Kuv Kup |        | Fu |
    //   Ke = | Kvu Kvv Kvp |;  Fe = | Fv |
    //        | Kpu Kpv Kpp |        | Fp |
    //         -           -          -  -
    // DenseSubMatrix.repostition (row_offset, column_offset, row_size, column_size).
    Muu.reposition (u_var*n_u_dofs, u_var*n_u_dofs, n_u_dofs, n_u_dofs);
    Muv.reposition (u_var*n_u_dofs, v_var*n_u_dofs, n_u_dofs, n_v_dofs);
    Mup.reposition (u_var*n_u_dofs, p_var*n_u_dofs, n_u_dofs, n_p_dofs);
    
    Mvu.reposition (v_var*n_v_dofs, u_var*n_v_dofs, n_v_dofs, n_u_dofs);
    Mvv.reposition (v_var*n_v_dofs, v_var*n_v_dofs, n_v_dofs, n_v_dofs);
    Mvp.reposition (v_var*n_v_dofs, p_var*n_v_dofs, n_v_dofs, n_p_dofs);
    
    Mpu.reposition (p_var*n_u_dofs, u_var*n_u_dofs, n_p_dofs, n_u_dofs);
    Mpv.reposition (p_var*n_u_dofs, v_var*n_u_dofs, n_p_dofs, n_v_dofs);
    Mpp.reposition (p_var*n_u_dofs, p_var*n_u_dofs, n_p_dofs, n_p_dofs);
    
    Kuu.reposition (u_var*n_u_dofs, u_var*n_u_dofs, n_u_dofs, n_u_dofs);
    Kuv.reposition (u_var*n_u_dofs, v_var*n_u_dofs, n_u_dofs, n_v_dofs);
    Kup.reposition (u_var*n_u_dofs, p_var*n_u_dofs, n_u_dofs, n_p_dofs);
    
    Kvu.reposition (v_var*n_v_dofs, u_var*n_v_dofs, n_v_dofs, n_u_dofs);
    Kvv.reposition (v_var*n_v_dofs, v_var*n_v_dofs, n_v_dofs, n_v_dofs);
    Kvp.reposition (v_var*n_v_dofs, p_var*n_v_dofs, n_v_dofs, n_p_dofs);
    
    Kpu.reposition (p_var*n_u_dofs, u_var*n_u_dofs, n_p_dofs, n_u_dofs);
    Kpv.reposition (p_var*n_u_dofs, v_var*n_u_dofs, n_p_dofs, n_v_dofs);
    Kpp.reposition (p_var*n_u_dofs, p_var*n_u_dofs, n_p_dofs, n_p_dofs);
    
    // DenseSubVector.reposition () member takes the (row_offset, row_size)
    Fu.reposition (u_var*n_u_dofs, n_u_dofs);
    Fv.reposition (v_var*n_u_dofs, n_v_dofs);
    Fp.reposition (p_var*n_u_dofs, n_p_dofs);
    
    Vu.reposition (u_var*n_u_dofs, n_u_dofs);
    Vv.reposition (v_var*n_u_dofs, n_v_dofs);
    Vp.reposition (p_var*n_u_dofs, n_p_dofs);
    
    Vudot.reposition (u_var*n_u_dofs, n_u_dofs);
    Vvdot.reposition (v_var*n_u_dofs, n_v_dofs);
    Vpdot.reposition (p_var*n_u_dofs, n_p_dofs);
    
    if(dim==3)
    {
      Muw.reposition (u_var*n_u_dofs, w_var*n_u_dofs, n_u_dofs, n_w_dofs);  // 0 matrix
      Mvw.reposition (v_var*n_v_dofs, w_var*n_v_dofs, n_v_dofs, n_w_dofs);  // 0 matrix
      Mpw.reposition (p_var*n_w_dofs, w_var*n_w_dofs, n_p_dofs, n_w_dofs);
      
      Mwu.reposition (w_var*n_w_dofs, u_var*n_w_dofs, n_w_dofs, n_u_dofs);  // 0 matrix
      Mwv.reposition (w_var*n_w_dofs, v_var*n_w_dofs, n_w_dofs, n_v_dofs);  // 0 matrix
      Mww.reposition (w_var*n_w_dofs, w_var*n_w_dofs, n_w_dofs, n_w_dofs);
      Mwp.reposition (w_var*n_w_dofs, p_var*n_w_dofs, n_w_dofs, n_p_dofs);
      
      Kuw.reposition (u_var*n_u_dofs, w_var*n_u_dofs, n_u_dofs, n_w_dofs);  // 0 matrix
      Kvw.reposition (v_var*n_v_dofs, w_var*n_v_dofs, n_v_dofs, n_w_dofs);  // 0 matrix
      Kpw.reposition (p_var*n_w_dofs, w_var*n_w_dofs, n_p_dofs, n_w_dofs);
      
      Kwu.reposition (w_var*n_w_dofs, u_var*n_w_dofs, n_w_dofs, n_u_dofs);  // 0 matrix
      Kwv.reposition (w_var*n_w_dofs, v_var*n_w_dofs, n_w_dofs, n_v_dofs);  // 0 matrix
      Kww.reposition (w_var*n_w_dofs, w_var*n_w_dofs, n_w_dofs, n_w_dofs);
      Kwp.reposition (w_var*n_w_dofs, p_var*n_w_dofs, n_w_dofs, n_p_dofs);
      
      Fw.reposition (w_var*n_u_dofs, n_w_dofs);
      Vw.reposition (w_var*n_u_dofs, n_w_dofs);
      Vwdot.reposition (w_var*n_u_dofs, n_w_dofs);
    } // end if (dim==3)
    
    
    // First we need nodal values of u, v, w, p in this element
    // navier_stokes_system.solution->get (dof_indices_u, elem_u);
    std::vector<Real> elem_u, elem_v, elem_w, elem_p;
    X.get (dof_indices_u, elem_u);
    X.get (dof_indices_v, elem_v);
    X.get (dof_indices_p, elem_p);
    if (dim==3) X.get (dof_indices_w, elem_w);
    
    
    // retrieve the nodal values of u_dot, v_dot, w_dot, p_dot, in this element
    std::vector<Real> elem_udot, elem_vdot, elem_wdot, elem_pdot;
    Xdot.get (dof_indices_u, elem_udot);
    Xdot.get (dof_indices_v, elem_vdot);
    Xdot.get (dof_indices_p, elem_pdot);
    if (dim==3) Xdot.get (dof_indices_w, elem_wdot);
    
    
    // set the DenseVector Ve and Vedot using std::vector
    for (unsigned int i=0; i<n_u_dofs; i++)
    {
      Vu(i) = elem_u[i];  Vudot(i) = elem_udot[i];
      Vv(i) = elem_v[i];  Vvdot(i) = elem_vdot[i];
      if (dim==3) { Vw(i) = elem_w[i];  Vwdot(i) = elem_wdot[i]; }
    }
    for (unsigned int i=0; i<n_p_dofs; i++)
    {  Vp(i) = elem_p[i];  Vpdot(i) = elem_pdot[i]; }
    
    // Loop over each Gauss point and compute element matices and vectors
    for (unsigned int qp=0; qp<qrule.n_points(); qp++)
    {
      // Note that Fe = Fe_v + Fe_s
      // the part of volume integral Fe_v is zero, and only surface
      // integral exists due to the boundary traction.
      
      // First we need evaluate the value of velocity at the gauss pt
      Number   u = 0.,   v = 0.,  w = 0. ;
      for (unsigned int l=0; l<n_u_dofs; l++)
      {
        // From the previous Newton iterate:
        u += phi[l][qp]*elem_u[l];
        v += phi[l][qp]*elem_v[l];
        if (dim==3)  w += phi[l][qp]*elem_w[l];
      }
      const NumberVectorValue U     (u, v, w);  // velocity vector
      //printf("************* assemble_ifunction: (u,v,w) = (%f, %f, %f ),\n",u,v,w);
      
      // Matrix contributions for the uu and vv couplings.
      for (unsigned int i=0; i<n_u_dofs; i++)
      {
        for (unsigned int j=0; j<n_u_dofs; j++)
        {
          // Mass matrix
          Muu(i,j) += JxW[qp]*phi[i][qp]*phi[j][qp];
          Mvv(i,j) += JxW[qp]*phi[i][qp]*phi[j][qp];
          if (dim==3) Mww(i,j) += JxW[qp]*phi[i][qp]*phi[j][qp];
          
          // convection and diffusion matrix
          Kuu(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp]*mu +     // diffusion term
                               phi[i][qp]*(U*dphi[j][qp]) );    // convection term
          
          Kvv(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp]*mu +     // diffusion term
                               phi[i][qp]*(U*dphi[j][qp]) );    // convection term
          
          if (dim ==3)
            Kww(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp]*mu +   // diffusion term
                                 phi[i][qp]*(U*dphi[j][qp]) );  // convection term
        } // end for j-loop
        
        // Matrix contributions for the up and vp couplings.
        for (unsigned int j=0; j<n_p_dofs; j++)
        {
          Kup(i,j) += -JxW[qp]*(psi[j][qp]*dphi[i][qp](0));
          Kvp(i,j) += -JxW[qp]*(psi[j][qp]*dphi[i][qp](1));
          if (dim==3) Kwp(i,j) += JxW[qp]*(psi[j][qp]*dphi[i][qp](2));
          //printf("************* assemble_ifunction: I am here 3,\n");
          
          Kpu(j,i) += -JxW[qp]*psi[j][qp]*dphi[i][qp](0);
          Kpv(j,i) += -JxW[qp]*psi[j][qp]*dphi[i][qp](1);
          if (dim==3) Kpw(j,i) += -JxW[qp]*(psi[j][qp]*dphi[i][qp](2));
        } // end for j-loop
        //printf("************* assemble_ifunction: I am here 4,\n");
      } // end for i-loop
    
    } // end of the quadrature point qp-loop
    
    // compuate the rhs vector caused by the pressure jump
    compute_element_rhs(mesh, elem, n_u_dofs, n_p_dofs, *fe_vel, *fe_pres,
                        periodicity, time, Fu,Fv,Fw,Fp);
    
    
    // At this point the interior element integration has been completed.
    // Now, we impose boundary conditions via the penalty method.
    // *** only Fe is changed, Ke is an auxiliary matrix.
    apply_bc_by_penalty(mesh, elem, time,"both", Kuu, Kvv, Kww, Kpp, Fu, Fv, Fw, Fp);
    //apply_bc_by_penalty(mesh, elem, time,"matrix", Muu, Mvv, Mww, Mpp, Fu, Fv, Fw, Fp);
    
    // Now we have element-wise quantities: Me, Ke, Fe, then we can
    // compute the element ifunction
    for (unsigned int i=0; i<n_dofs; i++)
    {
      Fe(i) = -Fe(i);
      for (unsigned int j=0; j<n_dofs; j++)
        Fe(i) += Me(i,j)*Vedot(j) + Ke(i,j)*Ve(j);
    }
    
    
    // At this point the interior element integration has been completed.
    // Now, we impose boundary conditions via the penalty method.
    // *** only Fe is changed, Ke is an auxiliary matrix.
    //apply_bc_by_penalty(mesh, elem, time,"vector", Kuu, Kvv, Kww, Kpp, Fu, Fv, Fw, Fp);
    
  
    
    // add to the Residual vector
    // should we apply the zero filter to the solution/rhs before assemble?
    GeomTools::zero_filter_dense_vector(Fe, 1e-10);
    navier_stokes_system.rhs->add_vector (Fe, dof_indices);
    
    // --------------- output test ----------------
    //GeomTools::output_dense_vector(Fe, Fe.size());
    // --------------------------------------------
  } // end of element loop
  
//  navier_stokes_system.rhs->close();
  
  // That's it.
  return;
  
} // end of assemble_ifunction()




// ============================================================================================
// ============================================================================================
void assemble_ijacobian (libMesh::EquationSystems& es,
                         const std::string& system_name,
                         const Real& time,
                         const Real& shift,
                         const NumericVector<Number>& X,
                         const NumericVector<Number>& Xdot)
{
  // It is a good idea to make sure we are assembling the proper system.
  libmesh_assert_equal_to (system_name, "Navier-Stokes");
  
  // Get a constant reference to the mesh object.
  const MeshBase& mesh    = es.get_mesh();
  const unsigned int dim  = mesh.mesh_dimension();
  
  // Get a reference to the Stokes system object.
  PetscTSSystem & navier_stokes_system = es.get_system<PetscTSSystem> ("Navier-Stokes");
  
  // Numeric ids corresponding to each variable in the system
  const unsigned int u_var = navier_stokes_system.variable_number ("u");  // u_var = 0
  const unsigned int v_var = navier_stokes_system.variable_number ("v");  // u_var = 1
  unsigned int w_var = 0;
  if(dim==3)   w_var = navier_stokes_system.variable_number ("w");        // w_var = 2
  const unsigned int p_var = navier_stokes_system.variable_number ("p");  // p_var = 2(dim=2); = 3(dim=3)
  
  // Get the Finite Element type for "u" and "p"
  FEType fe_vel_type  = navier_stokes_system.variable_type(u_var);
  FEType fe_pres_type = navier_stokes_system.variable_type(p_var);
  
  // Build a Finite Element object for "u" and "p"
  AutoPtr<FEBase> fe_vel  (FEBase::build(dim, fe_vel_type));
  AutoPtr<FEBase> fe_pres (FEBase::build(dim, fe_pres_type));
  
  // A Gauss quadrature rule for numerical integration.
  QGauss qrule (dim, fe_vel_type.default_quadrature_order());
  
  // Tell the finite element objects to use our quadrature rule.
  fe_vel->attach_quadrature_rule (&qrule);
  fe_pres->attach_quadrature_rule (&qrule);
  
  // Here we define some references to cell-specific data that
  // will be used to assemble the linear system.
  //
  // The element Jacobian * quadrature weight at each integration point.
  const std::vector<Real>& JxW                        = fe_vel->get_JxW();
  const std::vector<std::vector<Real> >& phi          = fe_vel->get_phi();
  const std::vector<std::vector<RealGradient> >& dphi = fe_vel->get_dphi();
  const std::vector<std::vector<Real> >& psi          = fe_pres->get_phi();
  
  
  // Define data structures to contain the element matrix and rhs vector.
  DenseMatrix<Number> Me;   // element mass matrix
  DenseSubMatrix<Number>
  Muu(Me), Muv(Me), Muw(Me), Mup(Me),
  Mvu(Me), Mvv(Me), Mvw(Me), Mvp(Me),
  Mwu(Me), Mwv(Me), Mww(Me), Mwp(Me),
  Mpu(Me), Mpv(Me), Mpw(Me), Mpp(Me);
  
  DenseMatrix<Number> Ke;   // element convection & diffusion matrix
  DenseSubMatrix<Number>
  Kuu(Ke), Kuv(Ke), Kuw(Ke), Kup(Ke),
  Kvu(Ke), Kvv(Ke), Kvw(Ke), Kvp(Ke),
  Kwu(Ke), Kwv(Ke), Kww(Ke), Kwp(Ke),
  Kpu(Ke), Kpv(Ke), Kpw(Ke), Kpp(Ke);
  
  DenseVector<Number> Fe;   // element force vector
  DenseSubVector<Number>  Fu(Fe),    Fv(Fe),    Fw(Fe),    Fp(Fe);
  
  // A reference to the \p DofMap object for this system.
  const DofMap & dof_map = navier_stokes_system.get_dof_map();
  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_u, dof_indices_v, dof_indices_w, dof_indices_p;
  
  // system parameters
  const Real mu           =  es.parameters.get<Real> ("viscosity");
  
  // loop over all the elements in the mesh that live on the local processor.
  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();
  for ( ; el != end_el; ++el)
  {
    // Store a pointer to the element we are currently working on.
    const Elem* elem = *el;
    
    // Get the degree of freedom indices for the current element.
    dof_map.dof_indices (elem, dof_indices);
    dof_map.dof_indices (elem, dof_indices_u, u_var);
    dof_map.dof_indices (elem, dof_indices_v, v_var);
    dof_map.dof_indices (elem, dof_indices_p, p_var);
    
    const unsigned int n_dofs   = dof_indices.size();
    const unsigned int n_u_dofs = dof_indices_u.size();
    const unsigned int n_v_dofs = dof_indices_v.size();
    const unsigned int n_p_dofs = dof_indices_p.size();
    unsigned int n_w_dofs = 0;
    if(dim==3)
    {
      dof_map.dof_indices (elem, dof_indices_w, w_var);
      n_w_dofs = dof_indices_w.size();
    }
    
    // Compute the element-specific data for the current element.
    // This involves computing the location of the quadrature points
    // and the shape functions (phi, dphi) for the current element.
    fe_vel->reinit  (elem);
    fe_pres->reinit (elem);
    
    // Zero the element matrix and right-hand side before summing them.
    Me.resize (n_dofs, n_dofs);
    Ke.resize (n_dofs, n_dofs);
    Fe.resize (n_dofs);
    
    // Reposition the submatrices...  The idea is this:
    //         -           -          -  -
    //        | Kuu Kuv Kup |        | Fu |
    //   Ke = | Kvu Kvv Kvp |;  Fe = | Fv |
    //        | Kpu Kpv Kpp |        | Fp |
    //         -           -          -  -
    // DenseSubMatrix.repostition (row_offset, column_offset, row_size, column_size).
    Muu.reposition (u_var*n_u_dofs, u_var*n_u_dofs, n_u_dofs, n_u_dofs);
    Muv.reposition (u_var*n_u_dofs, v_var*n_u_dofs, n_u_dofs, n_v_dofs);
    Mup.reposition (u_var*n_u_dofs, p_var*n_u_dofs, n_u_dofs, n_p_dofs);
    
    Mvu.reposition (v_var*n_v_dofs, u_var*n_v_dofs, n_v_dofs, n_u_dofs);
    Mvv.reposition (v_var*n_v_dofs, v_var*n_v_dofs, n_v_dofs, n_v_dofs);
    Mvp.reposition (v_var*n_v_dofs, p_var*n_v_dofs, n_v_dofs, n_p_dofs);
    
    Mpu.reposition (p_var*n_u_dofs, u_var*n_u_dofs, n_p_dofs, n_u_dofs);
    Mpv.reposition (p_var*n_u_dofs, v_var*n_u_dofs, n_p_dofs, n_v_dofs);
    Mpp.reposition (p_var*n_u_dofs, p_var*n_u_dofs, n_p_dofs, n_p_dofs);
    
    Kuu.reposition (u_var*n_u_dofs, u_var*n_u_dofs, n_u_dofs, n_u_dofs);
    Kuv.reposition (u_var*n_u_dofs, v_var*n_u_dofs, n_u_dofs, n_v_dofs);
    Kup.reposition (u_var*n_u_dofs, p_var*n_u_dofs, n_u_dofs, n_p_dofs);
    
    Kvu.reposition (v_var*n_v_dofs, u_var*n_v_dofs, n_v_dofs, n_u_dofs);
    Kvv.reposition (v_var*n_v_dofs, v_var*n_v_dofs, n_v_dofs, n_v_dofs);
    Kvp.reposition (v_var*n_v_dofs, p_var*n_v_dofs, n_v_dofs, n_p_dofs);
    
    Kpu.reposition (p_var*n_u_dofs, u_var*n_u_dofs, n_p_dofs, n_u_dofs);
    Kpv.reposition (p_var*n_u_dofs, v_var*n_u_dofs, n_p_dofs, n_v_dofs);
    Kpp.reposition (p_var*n_u_dofs, p_var*n_u_dofs, n_p_dofs, n_p_dofs);
    
    // DenseSubVector.reposition () member takes the (row_offset, row_size)
    Fu.reposition (u_var*n_u_dofs, n_u_dofs);
    Fv.reposition (v_var*n_u_dofs, n_v_dofs);
    Fp.reposition (p_var*n_u_dofs, n_p_dofs);
    
    if(dim==3)
    {
      Muw.reposition (u_var*n_u_dofs, w_var*n_u_dofs, n_u_dofs, n_w_dofs);  // 0 matrix
      Mvw.reposition (v_var*n_v_dofs, w_var*n_v_dofs, n_v_dofs, n_w_dofs);  // 0 matrix
      Mpw.reposition (p_var*n_w_dofs, w_var*n_w_dofs, n_p_dofs, n_w_dofs);
      
      Mwu.reposition (w_var*n_w_dofs, u_var*n_w_dofs, n_w_dofs, n_u_dofs);  // 0 matrix
      Mwv.reposition (w_var*n_w_dofs, v_var*n_w_dofs, n_w_dofs, n_v_dofs);  // 0 matrix
      Mww.reposition (w_var*n_w_dofs, w_var*n_w_dofs, n_w_dofs, n_w_dofs);
      Mwp.reposition (w_var*n_w_dofs, p_var*n_w_dofs, n_w_dofs, n_p_dofs);
      
      Kuw.reposition (u_var*n_u_dofs, w_var*n_u_dofs, n_u_dofs, n_w_dofs);  // 0 matrix
      Kvw.reposition (v_var*n_v_dofs, w_var*n_v_dofs, n_v_dofs, n_w_dofs);  // 0 matrix
      Kpw.reposition (p_var*n_w_dofs, w_var*n_w_dofs, n_p_dofs, n_w_dofs);
      
      Kwu.reposition (w_var*n_w_dofs, u_var*n_w_dofs, n_w_dofs, n_u_dofs);  // 0 matrix
      Kwv.reposition (w_var*n_w_dofs, v_var*n_w_dofs, n_w_dofs, n_v_dofs);  // 0 matrix
      Kww.reposition (w_var*n_w_dofs, w_var*n_w_dofs, n_w_dofs, n_w_dofs);
      Kwp.reposition (w_var*n_w_dofs, p_var*n_w_dofs, n_w_dofs, n_p_dofs);
      
      Fw.reposition (w_var*n_u_dofs, n_w_dofs);
    } // end if (dim==3)
    
    
    // First we need nodal values of u, v, w, p in this element
    std::vector<Real> elem_u, elem_v, elem_w, elem_p;
    X.get (dof_indices_u, elem_u);
    X.get (dof_indices_v, elem_v);
    X.get (dof_indices_p, elem_p);
    if (dim==3) X.get (dof_indices_w, elem_w);
    
//    std::vector<Real> elem_u, elem_v, elem_w, elem_p;
//    navier_stokes_system.solution->get (dof_indices_u, elem_u);
//    navier_stokes_system.solution->get (dof_indices_v, elem_v);
//    navier_stokes_system.solution->get (dof_indices_p, elem_p);
//    if (dim==3) navier_stokes_system.solution->get (dof_indices_w, elem_w);
    
    // Loop over each Gauss point and compute element matices and vectors
    for (unsigned int qp=0; qp<qrule.n_points(); qp++)
    {
      // Note that Fe = Fe_v + Fe_s
      // the part of volume integral Fe_v is zero, and only surface
      // integral exists due to the boundary traction.
      
      // Evaluate the value of velocity and its gradient at the gauss pt
      Number   u = 0.,   v = 0.,  w = 0. ;
      Gradient grad_u, grad_v, grad_w;
      for (unsigned int l=0; l<n_u_dofs; l++)
      {
        u += phi[l][qp]*elem_u[l];
        v += phi[l][qp]*elem_v[l];
        if (dim==3)  w += phi[l][qp]*elem_w[l];
        
        grad_u.add_scaled (dphi[l][qp],elem_u[l]);
        grad_v.add_scaled (dphi[l][qp],elem_v[l]);
        if (dim==3)  grad_w.add_scaled (dphi[l][qp],elem_w[l]);
      }
      const NumberVectorValue U     (u, v, w);  // velocity vector
      
      // velocity gradient
      Number  u_x = 0., u_y = 0., u_z = 0.,
              v_x = 0., v_y = 0., v_z = 0.,
              w_x = 0., w_y = 0., w_z = 0.;
      u_x = grad_u(0);  u_y = grad_u(1);  u_z = grad_u(2);
      v_x = grad_v(0);  v_y = grad_v(1);  v_z = grad_v(2);
      w_x = grad_w(0);  w_y = grad_w(1);  w_z = grad_w(2);
      
      // Matrix contributions for the uu and vv couplings.
      for (unsigned int i=0; i<n_u_dofs; i++)
      {
        for (unsigned int j=0; j<n_u_dofs; j++)
        {
          // mass matrix
          Muu(i,j) += shift*JxW[qp]*phi[i][qp]*phi[j][qp];      // mass-matrix term
          Mvv(i,j) += shift*JxW[qp]*phi[i][qp]*phi[j][qp];      // mass-matrix term
          
          // convection and diffusion matrix
          Kuu(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp]*mu +       // diffusion term
                               phi[i][qp]*(U*dphi[j][qp]) +       // convection term
                               u_x*phi[i][qp]*phi[j][qp] ); // Newton term
          
          Kuv(i,j) += JxW[qp]*u_y*phi[i][qp]*phi[j][qp];    // Newton term
          
          Kvu(i,j) += JxW[qp]*v_x*phi[i][qp]*phi[j][qp];    // Newton term
          
          Kvv(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp]*mu +       // diffusion term
                               phi[i][qp]*(U*dphi[j][qp]) +       // convection term
                               v_y*phi[i][qp]*phi[j][qp] ); // Newton term
          
          if (dim ==3)
          {
            Kuw(i,j) += JxW[qp]*u_z*phi[i][qp]*phi[j][qp];    // Newton term
            Kvw(i,j) += JxW[qp]*v_z*phi[i][qp]*phi[j][qp];    // Newton term
            Kwu(i,j) += JxW[qp]*w_x*phi[i][qp]*phi[j][qp];    // Newton term
            Kwv(i,j) += JxW[qp]*w_y*phi[i][qp]*phi[j][qp];    // Newton term
            
            Mww(i,j) += shift*JxW[qp]*phi[i][qp]*phi[j][qp];      // mass-matrix term
            Kww(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp]*mu +       // diffusion term
                                 phi[i][qp]*(U*dphi[j][qp]) +       // convection term
                                 w_z*phi[i][qp]*phi[j][qp] ); // Newton term
          } // end if (dim==3)
        } // end for j-loop
        
        // Matrix contributions for the up and vp couplings. (No Newton term)
        for (unsigned int j=0; j<n_p_dofs; j++)
        {
          Kup(i,j) += -JxW[qp]*(psi[j][qp]*dphi[i][qp](0));
          Kvp(i,j) += -JxW[qp]*(psi[j][qp]*dphi[i][qp](1));
          if (dim==3) Kwp(i,j) += -JxW[qp]*(psi[j][qp]*dphi[i][qp](2));
          
          Kpu(j,i) += -JxW[qp]*psi[j][qp]*dphi[i][qp](0);
          Kpv(j,i) += -JxW[qp]*psi[j][qp]*dphi[i][qp](1);
          if (dim==3) Kpw(j,i) += -JxW[qp]*(psi[j][qp]*dphi[i][qp](2));
        } // end for j-loop
      } // end for i-loop
      
    } // end of the quadrature point qp-loop
    
    
    // At this point the interior element integration has been completed.
    // Now, we impose boundary conditions via the penalty method.
    // *** only Ke is changed, Fe is an auxiliary vector, which is not changed!
    apply_bc_by_penalty(mesh, elem, time, "matrix", Kuu, Kvv, Kww, Kpp, Fu, Fv, Fw, Fp);
    Ke += Me;
    // Note Me is small compared with the penalty, so it doesn't matter to "add" before
    // or after ???
    
    // add to the Jacobian matrix
    GeomTools::zero_filter_dense_matrix(Ke, 1e-10);
    navier_stokes_system.matrix->add_matrix (Ke, dof_indices);
    
    // --------------- output test ----------------
    //GeomTools::output_dense_matrix(Ke);
    // --------------------------------------------
  } // end of element loop
  std::cout<<"*********************** shift = "<<shift <<std::endl;
  //navier_stokes_system.matrix->close();
  
  // That's it.
  return;
} // end of assemble_ijacobian()



// ============================================================================================
// ============================================================================================
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
                         DenseSubVector<Number>& Fp)
{
  // ---------------------------- Add the pressure at the boundary ----------------------
  if(periodicity)
  {
    // problem dimension
    const unsigned int dim = mesh.mesh_dimension();
    
    // 2.1 loop over each side of the element
    for (unsigned int s=0; s<elem->n_sides(); s++)
    {
      // 2.2 if this element has NO neighors in side s direction,
      // then side s must be on the boundary.
      if (elem->neighbor(s) == NULL)
      {
        // First, check if this side is on the inlet or outlet boundary
        // Boundary ids
        //      build_square():  1=right; 3=left (2D)
        //      build_cube()  :  2=right; 4=left (3D)
        boundary_id_type left_id = 3, right_id = 1;
        if(dim==3)     { left_id = 4, right_id = 2; }
        const bool left_boundary  = mesh.get_boundary_info().has_boundary_id(elem,s,left_id);
        const bool right_boundary = mesh.get_boundary_info().has_boundary_id(elem,s,right_id);
        
        // check if both are equal to true, it is obviously wrong
        if( (left_boundary) && (right_boundary) )
        {
          std::cout <<"**************** error in compute_element_rhs(): *******************"<<std::endl
                    <<"*** the current node can not be on both left and right boundary! ***"<<std::endl
                    <<"********************************************************************"<<std::endl;
        }
        // if this side is neither on the left nor on the right, continue to check next side
        if( (!left_boundary) && (!right_boundary) ) { continue; }
        
        
        // Otherwise, if this side is either left or right side, we will
        // compute the pressure traction along this side.
        
        // construct the side element, which has a lower dimension than the body elem
        AutoPtr<Elem> s_elem ( elem->build_side(s) );
        
        // for reinit(side_elem), note: don't use release(), which leads to leak memory
        const Elem* side_elem = s_elem.get();
        
        // build FEBase of the side element. (* use vel instead of pres!)
        FEType side_fe_type = fe_vel.get_fe_type();
        AutoPtr<FEBase> side_fe_vel ( FEBase::build(dim-1, side_fe_type) );
        
        // A Gauss quadrature rule for numerical integration.
        QGauss side_qrule (dim-1, side_fe_type.default_quadrature_order());
        side_fe_vel->attach_quadrature_rule( &side_qrule );
        
        // The element base function and Jacobian * Qweight at each integration point.
        // the xyz coordinates of the gauss pts (real coord not ref coord!)
        const std::vector<Point>& side_qp_xyz = side_fe_vel->get_xyz();
        const std::vector<Real>& JxW = side_fe_vel->get_JxW();
        const std::vector<std::vector<Real> >& phi = side_fe_vel->get_phi();
        side_fe_vel->reinit (side_elem);
        //side_qrule.print_info(); // generate 3 Gauss pts for 2D, 9 of 3D by default
        
        
        // Now we will build the rhs vector due to the pressure jump.
        std::vector<Real> Fu_s ( phi.size() );  // Fu on side s
        for (unsigned int qp=0; qp<side_qrule.n_points(); qp++)
        {
          // obtain the pressure values and normal directions of the boundary.
          // Generally, the pressure values may vary along the boundaries.
          Real pres = 0.0, boundary_normal = 0.0;
          const Point pt = side_qp_xyz[qp];
          if(left_boundary)
          {
            // pressure of the last and current time step
            pres =  boundary_pressure( pt,time,"left");
            boundary_normal = -1.0;
          }
          
          if(right_boundary)
          {
            pres =  boundary_pressure( pt,time,"right");
            boundary_normal = +1.0;
          }
          
          // the force vector of the side element: a1*F(n+1) + a0*F(n).
          for (unsigned int i=0; i<phi.size(); i++)
            Fu_s[i] += -JxW[qp]*phi[i][qp]*boundary_normal*pres;
        } // end of the quadrature point qp-loop
        
        
        // add the entries of side to the element vector Fu
        // first loop over all the nodes in this element
        for (unsigned int n=0; n<elem->n_nodes(); n++)
        {
          // Loop over the nodes on the side.
          for (unsigned int ns=0; ns<side_elem->n_nodes(); ns++)
          {
            if (elem->node(n) == side_elem->node(ns)) // compare global id (dof_id_type)
              Fu(n) += Fu_s[ns];
          } // end for ns-loop
        } // end for n-loop
        
      } // end if (elem->neighbor)
      
    } // end for s-loop
    
  } // end if(periodicity)
  
  // All done and return!
  return;
}



// ============================================================================================
// ============================================================================================
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
                         DenseSubVector<Number>& Fp)
{
  const unsigned int dim = mesh.mesh_dimension();
  
  // The penalty value.
  const Real penalty = 1.e8;
  
  // The following loops over the sides of the element.
  // If the element has no neighbor on a side then that
  // side MUST live on a boundary of the domain.
  for (unsigned int s=0; s<elem->n_sides(); s++)
    if (elem->neighbor(s) == NULL)
    {
      // Boundary ids are set internally by build_square().
      // build_square():  0=bottom; 1=right; 2=top; 3=left
      // Set no-slip boundary conditions on the channel walls
      bool wall_boundary = false;
      const bool inlet_boundary  = mesh.get_boundary_info().has_boundary_id(elem,s,3);
      const bool outlet_boundary = mesh.get_boundary_info().has_boundary_id(elem,s,1);
      if( (!inlet_boundary) && (!outlet_boundary) ) wall_boundary = true;
      
      // -1.- build the reduced-order side element for  for "p" Dirichlet BC.
      // this operation is only on the inlet or outlet boundary
      if( inlet_boundary || outlet_boundary )
      {
        AutoPtr<Elem> p_side ( elem->side(s) );
        for (unsigned int ns=0; ns<p_side->n_nodes(); ns++)
        {
          // if this is inlet/outlet, We impose the pressure Dirichlet BC by penalty method.
          Real p_value = 0.;
          if( inlet_boundary )
            p_value = boundary_pressure( p_side->point(ns), time, "left");
          else if ( outlet_boundary )
            p_value = boundary_pressure( p_side->point(ns), time, "right");
          // end if-else
          
          for (unsigned int n=0; n<elem->n_nodes(); n++)
            if (elem->node(n) == p_side->node(ns))
            {
              if (matrix_or_vector=="matrix")
                Kpp(n,n) += penalty;        // Matrix contribution.
              else if (matrix_or_vector=="vector")
                Fp(n) += penalty*p_value;   // Right-hand-side contribution.
              else if (matrix_or_vector=="both")
              { Kpp(n,n) += penalty;   Fp(n) += penalty*p_value;}
              // end if-else
            } // end if
          // end for n-loop
          
        } // end for ns
        
        // if this is the inlet or outlet, it MUST not be the no-slip wall
        // then we skip the -2.- part and continue to the next s-loop.
        continue;
      } // end if( inlet_boundary || outlet_boundary )
      
      
      // if this is neither the inlet nor the outlet, it MUST be the no-slip walls
      // the following part -2- will impose the no-slip BC by penalty method.
      
      // -2.- build the full-order side element for "v" Dirichlet BC.
      AutoPtr<Elem> side (elem->build_side(s));
      for (unsigned int ns=0; ns<side->n_nodes(); ns++)
      {
        // Otherwise if this is the wall of the channel, set u/v/w = 0
        const Real u_value = 0., v_value = 0., w_value = 0.;
        
        // Find the node on the element matching this node on the side.
        // That defined where in the element matrix
        // the boundary condition will be applied.
        for (unsigned int n=0; n<elem->n_nodes(); n++)
          if (elem->node(n) == side->node(ns))
          {
            if (matrix_or_vector=="matrix")       // Matrix contribution.
            { Kuu(n,n) += penalty;      Kvv(n,n) += penalty;}
            else if (matrix_or_vector=="vector")  // Right-hand-side contribution.
            { Fu(n) += penalty*u_value; Fv(n) += penalty*v_value;  }
            else if (matrix_or_vector=="both")
            {
              Kuu(n,n) += penalty;  Fu(n) += penalty*u_value;
              Kvv(n,n) += penalty;  Fv(n) += penalty*v_value;
            }
            
            if(dim==3)
            {
              if (matrix_or_vector=="matrix")
                Kww(n,n) += penalty;        // Matrix contribution.
              else if (matrix_or_vector=="vector")
                Fw(n) += penalty*w_value;   // Right-hand-side contribution.
              else if (matrix_or_vector=="both")
              { Kww(n,n) += penalty;   Fw(n) += penalty*w_value;}
              // end if-else
            }
            //penalty_u[n] = true;   // label the penalized local dof indices
          } // end if (elem->node(n) == side->node(ns))
        // end for n-loop
      } // end for ns-loop
      
    } // end if (elem->neighbor(side) == NULL)
  // -- end for s-loop
  
  // Pin the pressure to zero at global node number "pressure_node". This effectively
  // removes the non-trivial null space of constant pressure solutions.
  // * this is not necessary for periodic bc with pressure jump! we here set "false"!
  const bool pin_pressure = false;
  if (pin_pressure)
  {
    const unsigned int pressure_node = 0;
    const Real p_value               = 0.0;
    for (unsigned int c=0; c<elem->n_nodes(); c++)
      if (elem->node(c) == pressure_node)
      {
        Kpp(c,c) += penalty;
        Fp(c)    += penalty*p_value;
      }
  } // end if (pin_pressure)
  
} // end of applying boundary condition by penalty method




// ============================================================================================
// ============================================================================================
Real boundary_pressure(const Point& pt,
                       const Real& t,
                       const std::string& which_side)
{
  // set the pressure on the left boundary as a time dependent function
  if(which_side=="left")
  {
    //return 10.0;
    
    const Real t0 = 5, p0 = 0.01;
    const Real T = 10.0*t0;  // periodicity of sine function
    return p0*std::sin(2.0*pi*t/T);
    
    // -----------------------------------
//    if(t<t0)
//      return p0*t/t0;
//    else
//      return p0;
    // -----------------------------------
  }
  // set the pressure on the right boundary as a constant
  else if(which_side=="right")
  {
    return 0.0;
  }
  else
  {
    std::cout <<"**************** error in boundary_pressure(): *******************"<<std::endl
              <<"********** the side can only be left or right boundary !**********"<<std::endl
              <<"******************************************************************"<<std::endl;
    return 1E100;
  } // end if-else
  
}



// ============================================================================================
// ============================================================================================
int write_out_section_profile(const libMesh::EquationSystems& es,
                              const Real x_pos,
                              const unsigned int y_div,
                              const unsigned int n_step)
{
  // Get a reference to the Stokes system object.
  const TransientLinearImplicitSystem & navier_stokes_system =
         es.get_system<TransientLinearImplicitSystem> ("Navier-Stokes");
  const Real XA =  es.parameters.get<Real> ("XA_boundary");
  const Real XB =  es.parameters.get<Real> ("XB_boundary");
  const Real YA =  es.parameters.get<Real> ("YA_boundary");
  const Real YB =  es.parameters.get<Real> ("YB_boundary");
  
  libmesh_example_requires(x_pos <= XB, " x_pos must be between XA and XB");
  libmesh_example_requires(x_pos >= XA, " x_pos must be between XA and XB");
  const Real h = (YB - YA)/Real(y_div);
  
  // Numeric ids corresponding to each variable in the system
  const unsigned int u_var = navier_stokes_system.variable_number ("u");  // u_var = 0

  // write out data
  {
    std::string filename = "u_profile_" + std::to_string(x_pos) + "_" + std::to_string(n_step) + ".txt";
    std::ofstream outfile(filename,std::ios_base::out);
    
    //
    for ( unsigned int i=0; i<y_div+1; ++i )
    {
      const Real y_coord = YA + Real(i)*h;  // y coordinate of the point
      Point pt(x_pos,y_coord);
      const Real u_value = navier_stokes_system.point_value(u_var,pt);
      outfile << y_coord << "   " << u_value << "\n";
    } // end for i-loop
    
    outfile.close();
    std::cout << "====================== u profile at the cross section is write out! ================="
    <<std::endl;
  } // end if
  
  return 0;
}


  
// ============================================================================================
// ============================================================================================
int write_out_section_profile_history(const libMesh::EquationSystems& es,
                                      const Real x_pos,
                                      const unsigned int y_div,
                                      const unsigned int n_step,
                                      const Real time)
{
  // Get a reference to the Stokes system object and its parameters.
  const TransientLinearImplicitSystem & navier_stokes_system =
  es.get_system<TransientLinearImplicitSystem> ("Navier-Stokes");
  const Real XA =  es.parameters.get<Real> ("XA_boundary");
  const Real XB =  es.parameters.get<Real> ("XB_boundary");
  const Real YA =  es.parameters.get<Real> ("YA_boundary");
  const Real YB =  es.parameters.get<Real> ("YB_boundary");
  
  libmesh_example_requires(x_pos <= XB, " x_pos must be between XA and XB");
  libmesh_example_requires(x_pos >= XA, " x_pos must be between XA and XB");
  const Real h = (YB - YA)/Real(y_div);
  
  // Numeric ids corresponding to each variable in the system
  const unsigned int u_var = navier_stokes_system.variable_number ("u");  // u_var = 0
  
  // write out history data
  {
    std::string filename = "u_profile_history_" + std::to_string(x_pos) + ".txt";
    std::ofstream outfile;
    int o_width = 15, o_precision = 9;
    
    if(n_step==1 )
    {
      // first, we write the y coordinates
      outfile.open(filename,std::ios_base::out);
      outfile.setf(std::ios::right);    outfile.setf(std::ios::fixed);
      outfile.precision(o_precision);   outfile.width(o_width);
      if( es.comm().rank()==0 )outfile << 0 ; // fill zeros for time step and time
      
      outfile.setf(std::ios::right);    outfile.setf(std::ios::fixed);
      outfile.precision(o_precision);   outfile.width(o_width);
      if( es.comm().rank()==0 )outfile << 0.0; // fill zeros for time step and time
      
      for ( unsigned int i=0; i<y_div+1; ++i )
      {
        const Real y_coord = YA + Real(i)*h;  // y coordinate of the point
        outfile.setf(std::ios::right);  outfile.setf(std::ios::fixed);
        outfile.precision(o_precision);           outfile.width(o_width);
        if( es.comm().rank()==0 ) outfile  << y_coord; // write the coords at the first time step
      }
      if( es.comm().rank()==0 ) outfile << "\n";
      
    }
    
    if(n_step>1)
    {
      outfile.open(filename,std::ios_base::app);
      outfile.setf(std::ios::right);  outfile.setf(std::ios::fixed);
      outfile.precision(o_precision); outfile.width(o_width);
      if( es.comm().rank()==0 )outfile << n_step;    // time step and time
      
      outfile.setf(std::ios::right);  outfile.setf(std::ios::fixed);
      outfile.precision(o_precision); outfile.width(o_width);
      if( es.comm().rank()==0 )outfile << time;    // time step and time
      
      for ( unsigned int i=0; i<y_div+1; ++i )
      {
        const Real y_coord = YA + Real(i)*h;  // y coordinate of the point
        Point pt(x_pos,y_coord);
        const Real u_value = navier_stokes_system.point_value(u_var,pt);
        outfile.setf(std::ios::right);  outfile.setf(std::ios::fixed);
        outfile.precision(o_precision); outfile.width(o_width);
        if( es.comm().rank()==0 ) outfile  << u_value;
      } // end for i-loop
      if( es.comm().rank()==0 ) outfile << "\n";
    }
    
    outfile.close();
    std::cout << "=================== u profile at the cross section is write out! ================="
              <<std::endl;
  }

  return 0;
}

void NSPetscTSSystem:: IFunction (Real time,
                             const NumericVector<Number>& X,
                             const NumericVector<Number>& Xdot,
                             NumericVector<Number>& F)
{
  this->rhs->zero();
  assemble_ifunction (this->get_equation_systems(), "Navier-Stokes", time, X, Xdot);
  this->rhs->close();
  F = *this->rhs->clone();   // this is assignable!
}

void NSPetscTSSystem::IJacobian (Real time,
                               const NumericVector<Number>& X,
                               const NumericVector<Number>& Xdot,
                               Real shift,
                               SparseMatrix<Number>& IJ,
                               SparseMatrix<Number>& IJpre)
{
  // call the user provided function
  this->matrix->zero();
  assemble_ijacobian (this->get_equation_systems(), "Navier-Stokes", time, shift, X, Xdot);
  IJ =  (*(this->matrix) ); // this is NOT assignable!

  this->matrix->close();
}

void NSPetscTSSystem::monitor (int  step, Real time,
                             NumericVector<Number>& X)
{
  // do nothing
  const Real x_pos = -50.0;
  const unsigned int ny_mesh = 30;
  const unsigned int y_div = ny_mesh*2;   // for 2nd order Q2 element
  write_out_section_profile_history(this->get_equation_systems(),x_pos,y_div,step,time);

#ifdef LIBMESH_HAVE_EXODUS_API
//  // We write the file in the ExodusII format.
//    std::ostringstream file_name;
//    file_name << "ns_out_"
//              << std::setw(3)
//              << std::setfill('0')
//              << std::right
//              << step + 1
//              << ".e";
//    ExodusII_IO(this->get_mesh()).write_equation_systems (file_name.str(),
//                                                          this->get_equation_systems());

  // write out the equation systems
  std::string exodus_filename = "ns_system.e";
  if ( step==0 )
    ExodusII_IO(this->get_mesh()).write_equation_systems (exodus_filename,
                                                          this->get_equation_systems());

  // output options from transient ex1
    ExodusII_IO exodus_IO(this->get_mesh());
    exodus_IO.append(true);
    exodus_IO.write_timestep (exodus_filename, this->get_equation_systems(),step+1,time);
#else
      std::ostringstream file_name;

      file_name << "out_"
                << std::setw(3)
                << std::setfill('0')
                << std::right
                << t_step+1
                << ".gmv";

      GMVIO(mesh).write_equation_systems (file_name.str(), equation_systems);
#endif // #ifdef LIBMESH_HAVE_EXODUS_API
}
