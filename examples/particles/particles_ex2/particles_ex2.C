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

// Global array of function names to turn debugging prints on.
// Initialized in main.
#ifdef DEBUG
std::vector<std::string> vdebug;
bool debug(const char* s) {return std::find(vdebug.begin(),vdebug.end(),std::string(s)) != vdebug.end();}
#endif


class ChargedParticle : public Point {
public:
  ChargedParticle(const Point& point, Real charge) : Point(point), _charge(charge) {}
  const Real& charge() const {return _charge;}
  void operator=(const Point& p) {(*this)(0)=p(0);(*this)(1)=p(1);(*this)(2)=p(2);}

private:
  Real _charge;
};

class ChargedParticles : public ParticleMesh::ParticlePacker, public ParticleMesh::ParticleUnPacker {
public:
  unsigned int size() const {return _particles.size();}

  const Point& operator()(dof_id_type id) const {
#ifdef DEBUG
    std::map<dof_id_type,ChargedParticle>::const_iterator it = _particles.find(id);
    if (it == _particles.end()) {
      std::ostringstream err;
      err << "No particle with id << " << id << std::endl;
      libmesh_error_msg(err.str());
    }
#endif
    const Point& p= it->second; // implicit cast from ChargedParticle to Point
    return p;
  };

  void pack(Scatter::OutBuffer& obuf, const Point& point,const dof_id_type& id) {
    // WARNING! This may be confusing: we want to ignore the Point component of the ChargedParticle
    // stored in this container and use the point passed in here: it might have been modified during
    // translation.  Should we ask to pack only the additional payload?
#ifdef DEBUG
    std::map<dof_id_type,ChargedParticle>::const_iterator it = _particles.find(id);
    if (it == _particles.end()) {
      std::ostringstream err;
      err << "No particle with id << " << id << std::endl;
      libmesh_error_msg(err.str());
    }
#endif
    const ChargedParticle& p=it->second;
    // WARNING! This is where we used coordinates of the passed in point, rather than
    // the coordinates stored in p.
    Real x = point(0), y = point(1), z = point(2), q = p.charge();
    obuf.write(x);
    obuf.write(y);
    obuf.write(z);
    obuf.write(q);
    obuf.write(id);
  }

  void unpack(Scatter::InBuffer& ibuf, Point& point,dof_id_type& id) {
    Real x,y,z,q;
    ibuf.read(x);ibuf.read(y);ibuf.read(z);ibuf.read(q);ibuf.read(id);
    point = Point(x,y,z);
    ChargedParticle p(point,q);
    _particles.insert(std::pair<dof_id_type,ChargedParticle>(id,p));
  }

  // This is a pure convenience method built for this example only.
  bool add_mesh_particle(MeshBase& mesh,Real x, Real y, Real z, Real q) {
#ifdef DEBUG
    std::vector<std::string> vdebug;
    command_line_vector("debug",vdebug);
    bool debug = std::find(vdebug.begin(),vdebug.end(),std::string("ChargedParticles::add_mesh_particle")) != vdebug.end();
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
    dof_id_type id = (dof_id_type)_particles.size();
    ChargedParticle p(testPoint,q);
    _particles.insert(std::pair<dof_id_type,ChargedParticle>(id,p));
#ifdef DEBUG
    if (debug) {
      std::cout << "[" << mesh.comm().rank() << "|" << mesh.comm().size()<<"]: " << "added particle at location  " << testPoint << std::endl;
    }
#endif
    return true;
  }
protected:
  std::map<dof_id_type,ChargedParticle> _particles;
};

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
               << "               [--verbose [--keep-cout]] [--elem-info] [layout=2|3] [width=<w>] [density=<k>] [debug='[function[ function]...]']\n"
	       << "MEANING:\n"
	       << "Output:\n"
	       << "  --verbose: print lots of output\n"
	       << "  --keep-cout: each processor has a standard out (only rank 0 has it by default)\n"
	       << "  --elem-info: report the element data and the particles attached to each element in the ParticleMesh object\n"
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
      std::cout << "density: " << density << std::endl; // particles_per_axis
      std::cout << "debug:   [ ";
      for (unsigned int i = 0; i < vdebug.size(); ++i) std::cout << vdebug[i] << " ";
      std::cout << "]" << std::endl;
    }
  }

  bool elem_info = on_command_line("--elem-info");

  // Initialize mesh and particles
  SerialMesh mesh(init.comm());
  ChargedParticles qparticles;

  int height = layout >= 2 ? width : 1;
  int depth  = layout >= 3 ? width : 1;
  int x_count = density;
  int y_count = layout >= 2 ? density : 1;
  int z_count = layout >= 3 ? density : 1;

  if(layout == 1) {
    mesh.partitioner().reset(new CentroidPartitioner(CentroidPartitioner::X));
  }
  MeshTools::Generation::build_cube(mesh, width, height, depth, 0, width, 0, height, 0, depth);

  MeshTools::BoundingBox processor_box
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  // effectively a barrier -- need to make sure the point locator is constructed collectively
  // Otherwise -- bad MPI errors :-)
  // This is a wrinkle in libMesh's design.
  mesh.point_locator();

  if(layout == 1) {
    double minX = processor_box.min()(0);
    double maxX = processor_box.max()(0);
    double interval = width/(double)density;
    int minI = (int)ceil(minX/interval - .5);
    int maxI = (int)ceil(maxX/interval - .5);
    if(maxX == width) maxI = density;
    for(int i = minI; i < maxI; i++) {
      Point point((i + .5)*interval, .5, .5);
      qparticles.add_mesh_particle(mesh,point(0),point(1),point(2),point(0)+point(1)+point(2));
    }
  }
  else {
    for(int i = 0; i < x_count; i++) {
      for(int j = 0; j < y_count; j++) {
        for(int k = 0; k < z_count; k++) {
	  Real x = width*(i + .5)/x_count;
	  Real y = height*(j + .5)/y_count;
	  Real z = depth*(k + .5)/z_count;
          qparticles.add_mesh_particle(mesh,x,y,z,x+y+z);
        }
      }
    }
  }
  mesh.print_info();

  ParticleMesh pm(mesh);
  // Add the particles to ParticleMesh. Use its index in qparticles as the id.
  for (unsigned int i = 0; i < qparticles.size(); ++i) {
    pm.add_particle(qparticles(i),i);
  }
  pm.setup();
  pm.print_info(elem_info);
  // Translate the qparticles by 1.25 to the right.
  Point shift(1.25,0.0,0.0);
  std::vector<Point> shifts(pm.num_particles(),shift);
  if (!init.comm().rank()) {
    std::cout << "Translating particles ..." << std::endl;
  }
  // Create a new ChargedParticles container
  // FIXME: we will fix this with a 'postpack/preunpack' function that will allow us to clear out and reuse a container.
  ChargedParticles tparticles;
  pm.translate_particles(shifts,qparticles,tparticles);
  pm.print_info(elem_info);
  return 0;
}

