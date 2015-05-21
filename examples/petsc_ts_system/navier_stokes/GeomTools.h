//
//  GeomTools.h
//  
//
//  Created by Xujun ZHao on 10/14/14.
//
//

#ifndef _GeomTools_h
#define _GeomTools_h

// C++ Includes
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cstring>
#include <math.h>


// LibMesh library includes
//#include "libmesh/petsc_macro.h"
//#include "libmesh/libmesh_common.h"
#include "libmesh/point.h"

// dense matrix help to debug
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dense_submatrix.h"
#include "libmesh/dense_subvector.h"


// Bring in everything from the libMesh namespace
using namespace libMesh;



/*
 this class defines basic tools used in our codes
 */
class GeomTools
{
public:
  
  // return the norm of a point |x| = sqrt(x^2 + y^2 + z^2)
  static Real point_norm(const Point& pt);
  
  // return the distance between two points |pt0 - pt1|
  static Real point_distance(const Point& pt0,
                             const Point& pt1);
  
  // quadratic function used for applying BC to avoid singularities at corners
  static Real quadratic_function_2d(const Real& y,
                                    const Real& YA, const Real& YB);
  
  static Real quadratic_function_3d(const Real& y,  const Real& z,
                                    const Real& YA, const Real& YB,
                                    const Real& ZA, const Real& ZB);


  static void output_dense_matrix(const DenseMatrix<Number>& Ke);
                                  
  static void output_dense_matrix(const DenseMatrix<Number>& Ke,
                                  const unsigned int m,
                                  const unsigned int n);
  
  static void output_dense_vector(const DenseVector<Number>& Fe,
                                  const unsigned int n);

  static void output_subdense_matrix(const DenseSubMatrix<Number>& Ke,
                                     const unsigned int m,
                                     const unsigned int n);
  
  static void output_subdense_vector(const DenseSubVector<Number>& Fe,
                                     const unsigned int n);
  
  template <typename T>
  static void output_std_vector(const std::vector<T>& std_v);
  

  static void zero_filter_dense_matrix(DenseMatrix<Number>& Ae, const Real tol);
  static void zero_filter_dense_vector(DenseVector<Number>& Ve, const Real tol);
};



// =============================================================================================
Real GeomTools::point_norm(const Point& pt)
{
  Real val = 0.;
  for(unsigned int i=0; i<LIBMESH_DIM; ++i)
    val += pt(i)*pt(i);
  
  return std::sqrt(val);
}


// =============================================================================================
Real GeomTools::point_distance(const Point& pt0,
                               const Point& pt1)
{
  Point pt = pt0 - pt1;
  return point_norm(pt);
}


// =============================================================================================
Real GeomTools::quadratic_function_2d(const Real& y,
                                      const Real& YA,
                                      const Real& YB)
{
  Real Y0 = (YA + YB)/2.0;
  Real DY = YB - Y0;    // note, this also equals to -(YA - Y0)
  Real V0 = DY/2.;      // maximum velocity magnitude
  Real A0 = V0/(DY*DY);
  return V0 - A0*(y - Y0)*(y - Y0);
  
  // a controls the magnitude of velocity, b is the mag of geometry(height of channal)
  // to make sure u=0 at boundary, it requires b^2 = a*y^2
//  Real a = 1.0, b = 0.5, y0 = 0.0;
//  return b*b - a*(yb - y0)*(yb - y0);
}


// =============================================================================================
Real GeomTools::quadratic_function_3d(const Real& y,
                                      const Real& z,
                                      const Real& YA,
                                      const Real& YB,
                                      const Real& ZA,
                                      const Real& ZB)
{
  Real Y0 = (YA + YB)/2.0,  Z0 = (ZA + ZB)/2.0;
  Real DY = YB - Y0,        DZ = ZB - Z0;  // note, this also equals to -(YA - Y0)
  Real VY = DY/2.,          VZ = VY;       // maximum velocity magnitudes
  Real AY = VY/(DY*DY),     AZ = VZ/(DZ*DZ);
  
  Real value = ( VY - AY*(y - Y0)*(y - Y0) )*( VZ - AZ*(z - Z0)*(z - Z0) );
  //std::cout<<"quadratic_function_3d test: value = "<<value<<std::endl;
  return value;
  
  
  // a controls the magnitude of velocity, b is the mag of geometry(height of channal)
  // make sure that ( b^2-a*y^2 )*( b^2 - z^2 )
//  Real a = 1.0, b = 0.5;
//  Real y0 = 0.0,z0 = 0.0;
//  return ( b*b - a*(yb - y0)*(yb - y0) )*( b*b - a*(zb - z0)*(zb - z0) );
}




// =============================================================================================
void GeomTools::output_dense_matrix(const DenseMatrix<Number>& Ke)
{
  output_dense_matrix( Ke, Ke.m(), Ke.n() );
}



// =============================================================================================
void GeomTools::output_dense_matrix(const DenseMatrix<Number>& Ke,
                                    const unsigned int m,
                                    const unsigned int n)
{
  std::cout << "--------------------------- output matrix " << m << " x " << n
            << " ---------------------------" <<std::endl;
  for(unsigned int i=0; i<m; ++i)
  {
    for(unsigned int j=0; j<n; ++j)
    {
      std::cout << Ke(i,j) << "  " << std::setw(5);
    }
    std::cout << std::endl;
  }
  std::cout << "--------------------------- end of matrix ---------------------------" <<std::endl;
  
  // K(i,j) - K(j,i)
//  std::cout << "--------------------------- output matrix K - KT" << m << " x " << n
//  << " ---------------------------" <<std::endl;
//  for(unsigned int i=0; i<m; ++i)
//  {
//    for(unsigned int j=0; j<n; ++j)
//    {
//      if( Ke(i,j)-Ke(j,i) != 0.)
//        std::cout << std::setw(5) << Ke(i,j)-Ke(j,i) << "  " ;
//    }
//    //std::cout << std::endl;
//  }
//  std::cout << "--------------------------- end of matrix ---------------------------" <<std::endl;
  
}


// =============================================================================================
void GeomTools::output_dense_vector(const DenseVector<Number>& Fe,
                                    const unsigned int n)
{
  std::cout << "--------------------------- output vector ---------------------------" <<std::endl;
  for(unsigned int j=0; j<n; ++j)
    std::cout << Fe(j) << "  " << std::setw(5);
  std::cout << std::endl;
  std::cout << "--------------------------- end of vector ---------------------------" <<std::endl;
}


// =============================================================================================
void GeomTools::output_subdense_matrix(const DenseSubMatrix<Number>& Ke,
                                       const unsigned int m,
                                       const unsigned int n)
{
  std::cout << "--------------------------- output matrix " << m << " x " << n
            << " ---------------------------" <<std::endl;
  for(unsigned int i=0; i<m; ++i)
  {
    for(unsigned int j=0; j<n; ++j)
      std::cout << Ke(i,j) << "  " << std::setw(5);
    
    std::cout << std::endl;
  }
  std::cout << "--------------------------- end of matrix ---------------------------" <<std::endl;
  
  
  // K(i,j) - K(j,i)
//  std::cout << "--------------------------- output matrix K - KT " << m << " x " << n
//            << " ---------------------------" <<std::endl;
//  for(unsigned int i=0; i<m; ++i)
//  {
//    for(unsigned int j=0; j<n; ++j)
//      std::cout << std::setw(5) << Ke(i,j)-Ke(j,i) << "  " ;
//    
//    std::cout << std::endl;
//  }
//  std::cout << "--------------------------- end of matrix ---------------------------" <<std::endl;
  
}


// =============================================================================================
void GeomTools::output_subdense_vector(const DenseSubVector<Number>& Fe,
                                       const unsigned int n)
{
  std::cout << "--------------------------- output vector ---------------------------" <<std::endl;
  for(unsigned int j=0; j<n; ++j)
    std::cout << Fe(j) << "  " << std::setw(5);
  std::cout << "--------------------------- end of vector ---------------------------" <<std::endl;
}


// =============================================================================================
template <typename T>
void GeomTools::output_std_vector(const std::vector<T>& std_v)
{
  std::cout << "--------------------------- output vector ---------------------------" <<std::endl;
  for(unsigned int j=0; j<std_v.size(); ++j)
    std::cout << std::setw(5) << std_v[j] << "  " ;
  
  std::cout << std::endl;
  std::cout << "--------------------------- end of vector ---------------------------" <<std::endl;
  std::cout << std::endl;
}


// =============================================================================================
void GeomTools::zero_filter_dense_matrix(DenseMatrix<Number>& Ke, const Real tol)
{
  for(unsigned int i=0; i<Ke.m(); ++i)
  {
    for(unsigned int j=0; j<Ke.n(); ++j)
      if ( std::abs( Ke(i,j) ) <= tol ) Ke(i,j) = 0.0;
  } // end for i-loop
  
}

// =============================================================================================
void GeomTools::zero_filter_dense_vector(DenseVector<Number>& Ve, const Real tol)
{
  for(unsigned int j=0; j<Ve.size(); ++j)
    if ( std::abs( Ve(j) ) <= tol ) Ve(j) = 0.0;
}


#endif
