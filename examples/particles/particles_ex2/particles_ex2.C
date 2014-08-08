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
#include "libmesh/halo_manager.h"
#include "libmesh/serializer.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/point.h"
#include "libmesh/parallel_mesh.h"
#include "libmesh/elem.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/centroid_partitioner.h"
#include <iostream>
#include <istream>
#include <ostream>
#include <vector>
#include <sstream>

using namespace libMesh;
using MeshTools::BoundingBox;

class Particle : public Point {
public:
  Particle(Point& point, Real value) : Point(point), value(value)
  {}

  Particle() {}

  class PSerializer : public Serializer<Point*> {
  public:
    void read(std::istream& stream, Point*& buffer) {
      buffer = new Particle();
      stream.read((char*)buffer, sizeof(Particle));
      received_particles.push_back((Particle*)buffer);
    }
    void write(std::ostream& stream, Point* const & buffer) {
      stream.write((char*)buffer, sizeof(Particle));
    }
    void free(Point*& value) {
      delete value;
    }
    void delete_received_particles() {
      for(unsigned int i = 0; i < received_particles.size(); i++) {
        delete received_particles[i];
      }
      received_particles.clear();
    }
    std::vector<Particle*>& get_received_particles() {
      return received_particles;
    }
  private:
    std::vector<Particle*> received_particles;
  };
  
  Real get_value() const {return value;}
  
private:
  Real value;
};

std::ostream& operator<<(std::ostream& os, const BoundingBox& box) {
  os << "(" << box.min() << ", " << box.max() << ")";
  return os;
}

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

void print_coords(std::ostream& os, Point* point, int dim) {
  if(dim > 1) os << "(";
  for(int d = 0; d < dim; d++) {
    os << (*point)(d);
    if(d + 1 < dim) os << ", ";
  }
  if(dim > 1) os << ")";
}

void print_coords(std::ostream& os, const std::vector<Particle*> points,
    int dim)
{
  unsigned int num_printed = 15;
  if(num_printed > points.size()) num_printed = points.size();
  
  os << "[";
  for(unsigned int i = 0; i < num_printed; i++) {
    print_coords(os, points[i], dim);
    if(i + 1 < num_printed) os << ", ";
  }
  os << "]";
  
  if(points.size() > num_printed) {
    os << " ( + " << (points.size() - num_printed) << " more...)";
  }
}

Real time_diff(timeval start, timeval end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1e-6;
}

Particle* make_particle(ParallelMesh& mesh, Real x, Real y, Real z) {
  Real testX = x;
  Real testY = y;
  Real testZ = z;
  if(testX == (int)testX) testX += .5;
  if(testY == (int)testY) testY += .5;
  if(testZ == (int)testZ) testZ += .5;
  Point testPoint(testX, testY, testZ);
  const Elem* elem = mesh.point_locator()(testPoint);
  if(elem == NULL) return NULL;
  if(elem->processor_id() != mesh.processor_id()) return NULL;
  Point point(x, y, z);
  return new Particle(point, point(0)*10);
}

void make_mesh_and_particles(int dim, int width, int particles_per_axis,
    ParallelMesh& mesh, std::vector<Particle*>& particles)
{
  int height = dim >= 2 ? width : 1;
  int depth = dim >= 3 ? width : 1;
  
  int x_count = particles_per_axis;
  int y_count = dim >= 2 ? particles_per_axis : 1;
  int z_count = dim >= 3 ? particles_per_axis : 1;

  if(dim == 1) {
    mesh.partitioner().reset(new CentroidPartitioner(CentroidPartitioner::X));
  }
  MeshTools::Generation::build_cube(mesh, width, height, depth,
      0, width, 0, height, 0, depth);

  BoundingBox processor_box
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
      particles.push_back(new Particle(point, point(0)*10));
    }
  }
  else {
    for(int i = 0; i < x_count; i++) {
      for(int j = 0; j < y_count; j++) {
        for(int k = 0; k < z_count; k++) {
          Particle* particle = make_particle(mesh,
              width*(i + .5)/x_count,
              height*(j + .5)/y_count,
              depth*(k + .5)/z_count);
          if(particle != NULL) particles.push_back(particle);
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  LibMeshInit init(argc, argv);
  
  if (argc != 8) {
    libmesh_error_msg("Usage: " << argv[0]
        << " dim width halo_pad particles_per_axis num_reps use_point_tree use_all_gather");
  }
  HaloManager::Opts opts;
  int dim = std::atoi(argv[1]);
  int width = std::atoi(argv[2]);
  Real halo_pad = std::atof(argv[3]);
  int particles_per_axis = std::atoi(argv[4]);
  int num_reps = std::atoi(argv[5]);
  opts.use_point_tree = std::atoi(argv[6]);
  opts.use_all_gather = std::atoi(argv[7]);
  
  ParallelMesh mesh(init.comm());
  std::vector<Particle*> particles;
  make_mesh_and_particles(dim, width, particles_per_axis, mesh, particles);
  mesh.print_info();

  Particle::PSerializer serializer;
  std::ostringstream sout;
  std::vector<std::vector<Particle*> > result;
  HaloManager* hm = NULL;
  for(int c = 0; c < num_reps; c++) {
    if(hm != NULL) delete hm;
    serializer.delete_received_particles();
    result.clear();
    hm = new HaloManager(mesh, halo_pad, opts);
    hm->set_serializer(serializer);
    hm->find_particles_in_halos(
        reinterpret_cast<std::vector<Point*>& >(particles),
        reinterpret_cast<std::vector<std::vector<Point*> >& >(result));
  }
  
  BoundingBox processor_box
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  sout << "======== Processor " << mesh.processor_id() << " ========\n";
  sout << "Processor Box: " << processor_box << "\n";
  sout << "Halo Pad: " << halo_pad << "\n";
  sout << "Neighbors: " << hm->neighbor_processors() << "\n";
  sout << "Halo Neighbors: " << hm->box_halo_neighbor_processors() << "\n";
  sout << "Particle Inbox: ";
  print_coords(sout, serializer.get_received_particles(), dim);
  sout << "\n";
  sout << "Particle Groups:\n";
  for(unsigned int i = 0; i < result.size(); i++) {
    for(unsigned int j = 0; j < result[i].size(); j++) {
      libmesh_assert(result[i][j]->get_value() == 10*(*result[i][j])(0));
    }
    sout << "  ";
    print_coords(sout, particles[i], dim);
    sout << ": ";
    print_coords(sout, result[i], dim);
    sout << "\n";
  }

  std::string text_str = sout.str();
  std::vector<char> text(text_str.begin(), text_str.end());
  text.push_back('\0');
  init.comm().gather(0, text);
  int ci = 0;
  while(ci < (int)text.size()) {
    std::cout << &text[ci] << std::endl;
    while(text[ci++] != '\0');
  }

  serializer.delete_received_particles();
  delete hm;

  return 0;
}

