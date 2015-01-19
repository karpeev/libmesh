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
#include "libmesh/libmesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/point.h"
#include "libmesh/serial_mesh.h"
#include "libmesh/elem.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/centroid_partitioner.h"
#include "libmesh/particle_mesh.h"
#include <iostream>
#include <istream>
#include <ostream>
#include <vector>
#include <sstream>

using namespace libMesh;

#ifdef DEBUG
std::vector<std::string> vdebug;
bool debug(const char* s) {return std::find(vdebug.begin(),vdebug.end(),std::string(s)) != vdebug.end();}
#endif
// Global variables that need to be constructed collectively.
// Some would prefer to put it into the state of an 'Example' class object, but that tends to make this straightforward
// code more convoluted with lots of inversion of control patterns.
SerialMesh              *mesh;


std::ostream& rankprint(const Parallel::Communicator& comm,std::ostream& os) {os << "["<<comm.rank()<<"|"<<comm.size()<<"]: "; return os;}

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  for(int i = 0; i < (int)vec.size(); i++) {
    os << vec[i];
    if(i < (int)vec.size() - 1) os << ", ";
  }
  os << "]";
  return os;
}

class ChargedParticle : public Point {
public:
  ChargedParticle(const Point& point, Real charge) : Point(point), _charge(charge) {}
  const Real& charge() const {return _charge;}
  void operator=(const Point& p) {(*this)(0)=p(0);(*this)(1)=p(1);(*this)(2)=p(2);}

private:
  Real _charge;
};

std::ostream& operator<<(std::ostream& os, const ChargedParticle& q) {
  const Point& p = (Point)q;
  os << "("<<p(0)<<","<<p(1)<<","<<p(2)<<"):" << q.charge();
  return os;
}

class ChargedParticles : public ParticleMesh::Particles, public std::vector<ChargedParticle> {
public:
  const Point& operator()(unsigned int i) const {return (*this)[i];}; // implicit cast to const Point&
  void translate(const std::vector<Point>& shift) {
    for(unsigned int i = 0; i < this->size(); ++i) (*this)[i] = (*this)[i] + shift[i];
  }
  void write(Scatter::OutBuffer& obuf, unsigned int i) const {ChargedParticle p=(*this)[i];obuf.write(p(0));obuf.write(p(1));obuf.write(p(2));obuf.write(p.charge());}
  void read(Scatter::InBuffer& ibuf, unsigned int& i) {
    Real x,y,z,q;
    ibuf.read(x);ibuf.read(y);ibuf.read(z);ibuf.read(q);
    ChargedParticle p(Point(x,y,z),q);
    this->push_back(p);
    i = this->size()-1;
  }
  unsigned int size()  const {return this->std::vector<ChargedParticle>::size();}
  Particles*   clone() const {return new ChargedParticles();} // implicit cast to Particles*
  void         clear() {this->std::vector<ChargedParticle>::clear();}
  void         view(std::ostream& os) const {os << (const std::vector<ChargedParticle>&)(*this);}
};


Real time_diff(timeval start, timeval end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1e-6;
}

bool add_particle(MeshBase& mesh, ChargedParticles& qparticles, Real x, Real y, Real z, Real q) {
#ifdef DEBUG
  std::vector<std::string> vdebug;
  command_line_vector("debug",vdebug);
  bool debug = std::find(vdebug.begin(),vdebug.end(),std::string("add_particle")) != vdebug.end();
#endif
  Real testX = x;
  Real testY = y;
  Real testZ = z;
  if(testX == (int)testX) testX += .5;
  if(testY == (int)testY) testY += .5;
  if(testZ == (int)testZ) testZ += .5;
  Point testPoint(testX, testY, testZ);
  const Elem* elem = mesh.point_locator()(testPoint);
  if(elem == NULL) {
#ifdef DEBUG
    if (debug) {
      std::cout << "[" << mesh.comm().rank()<<"|"<<mesh.comm().size()<<"]: " << "rejected particle outside of mesh: location  " << testPoint << std::endl;
    }
#endif
    return false;
  }
  if(elem->processor_id() != mesh.processor_id()) {
#ifdef DEBUG
    if (debug) {
      std::cout << "[" << mesh.comm().rank()<<"|"<<mesh.comm().size()<<"]: " << "rejected particle outside of processor partition: location  " << testPoint << std::endl;
    }
#endif
    return false;
  }
  qparticles.push_back(ChargedParticle(testPoint, q));
#ifdef DEBUG
    if (debug) {
      std::cout << "[" << mesh.comm().rank()<<"|"<<mesh.comm().size()<<"]: " << "added particle: location  " << testPoint << std::endl;
    }
#endif
  return true;
}

void make_mesh_and_particles(UnstructuredMesh& mesh, int dim, int width, int particles_per_axis,  ChargedParticles& qparticles)
{
  int height = dim >= 2 ? width : 1;
  int depth = dim >= 3 ? width : 1;

  int x_count = particles_per_axis;
  int y_count = dim >= 2 ? particles_per_axis : 1;
  int z_count = dim >= 3 ? particles_per_axis : 1;

  if(dim == 1) {
    mesh.partitioner().reset(new CentroidPartitioner(CentroidPartitioner::X));
  }
  MeshTools::Generation::build_cube(mesh, width, height, depth, 0, width, 0, height, 0, depth);

  MeshTools::BoundingBox processor_box
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  if(dim == 1) {
    double minX = processor_box.min()(0);
    double maxX = processor_box.max()(0);
    double interval = width/(double)particles_per_axis;
    int minI = (int)ceil(minX/interval - .5);
    int maxI = (int)ceil(maxX/interval - .5);
    if(maxX == width) maxI = particles_per_axis;
    for(int i = minI; i < maxI; i++) {
      Point point((i + .5)*interval, .5, .5);
      add_particle(mesh,qparticles,point(0),point(1),point(2),point(0)+point(1)+point(2));
    }
  }
  else {
    for(int i = 0; i < x_count; i++) {
      for(int j = 0; j < y_count; j++) {
        for(int k = 0; k < z_count; k++) {
	  Real x = width*(i + .5)/x_count;
	  Real y = height*(j + .5)/y_count;
	  Real z = depth*(k + .5)/z_count;
          add_particle(mesh,qparticles,x,y,z,x+y+z);
        }
      }
    }
  }
}

// Print usage information if requested on command line
void print_help(int, char** argv)
{
  libMesh::out << "This example builds the basic data structures for the ParticleMesh system\n"
               << "BUILDING:\n"
               << "METHOD=<method> make \n"
               << "where <method> is dbg or opt.\n"
               << "\n-----------\n"
               << "HELP:        "
               << "\n-----------\n"
               << "Print this help message:\n"
               << argv[0] << " --help\n"
               << "\n-----------\n"
               << "RUNNING:     "
               << "\n-----------\n"
               << "Run in serial with build METHOD <method> as follows:\n"
               << "\n"
               << argv[0] << "\n"
               << "               [--verbose [--keep-cout]] [layout=2|3] [width=<w>] [density=<k>] [debug='[function[ function]...]']\n"
	       << "MEANING:\n"
	       << "Mesh:\n"
	       << "  The mesh is made up of unit cube elements laid out as follows:\n"
	       << "  layout=1: sequentially as a long string of cubes or\n"
	       << "  layout=2: as a square prism with the base paved by the cross-sections of the unit cubes\n"
	       << "               making up the mesh; the number of cubes in the cross-section is given by 'width'\n"
	       << "  layout=3: a cube made up of unit cubes with 'width' unit cubes in each direction.\n"
	       << "Particles:\n"
	       << "  Particles are laid out within the mesh with 'density' particles in each direction\n"
	       << "DEFAULTS:\n"
	       << "   layout  ............. 1     (dimensionality of the mesh layout)\n"
	       << "   width ............... 1     (mesh width in unit cubes)\n"
	       << "   density ............. 3\n"
	       << "NOTES:\n"
               << "   Use standard libMesh --keep-cout option to enable stdout on all processor\n"
	       << "   (enabled only on rank 0 by default); useful for debugging output with debug='...'\n"
               << "\n"
               << std::endl;
}


// NOTE: options are as follows:
//  layout_dim: The dimensionality for laying out the particles (1, 2, or 3)
//  width: The width of the mesh.  Mesh is made up of unit cube elements.
//      The height and depth will either be 1 or be equal to the width,
//      depending on dim.
//  density: Number of particles location along each of the
//      dim axes.
int main(int argc, char** argv) {
  LibMeshInit init(argc, argv);

  if (on_command_line("--help")) {
    print_help(argc, argv);
    return(0);
  }

  int layout  = command_line_value("layout",1);
  int width   = command_line_value("width",1);
  int density = command_line_value("density",1);

#ifdef DEBUG
  command_line_vector("debug",vdebug);
#endif


  if (on_command_line("--verbose")) {
    if (!init.comm().rank()) {
      std::cout << "layout:  " << layout  << std::endl;
      std::cout << "width:   " << width   << std::endl;
      std::cout << "density: " << density << std::endl;
      std::cout << "debug:   " << vdebug  << std::endl;
    }
  }

  mesh    = new SerialMesh(init.comm());
  // Force the construction of the point locator, since it is collective.
  // Point locator may then be used serially by the individual ranks.
  // This is a behavior that is hard to encapsulate, short of making ANY
  // use of point locator collective.
  mesh->point_locator();
  AutoPtr<ChargedParticles> qparticles(new ChargedParticles());
  make_mesh_and_particles(*mesh,layout, width, density,*qparticles);
  mesh->print_info();

  ParticleMesh pm(*mesh);
  // To avoid ambiguity of conversion (there are at least two ways to make AutoPtr<T> from AutoPtr<T1> when T1 is derived from T),
  // explicitly force a cast (i.e., pick one of those conversions).
  pm.set_local_particles((AutoPtr<ParticleMesh::Particles>)qparticles);
  pm.print_info();
  // Translate the qparticles by 0.75 to the right.
  Point shift(0.75,0.0,0.0);
  std::vector<Point> shifts(qparticles->size(),shift);
  if (!init.comm().processor_id()) {
    std::cout << "Translating local particles ..." << std::endl;
  }
  pm.translate_local_particles(shifts);
  pm.print_info();
  return 0;
}

