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
    void read(std::istream& stream, Point*& buffer) const {
      buffer = new Particle();
      stream.read((char*)buffer, sizeof(Particle));
    }
    void write(std::ostream& stream, Point* const & buffer) const {
      stream.write((char*)buffer, sizeof(Particle));
    }
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

void print_x_coords(std::ostream& os, const std::vector<Particle*> points) {
  unsigned int max_printed = 15;
  std::vector<Real> coords;
  for(unsigned int i = 0; i < points.size(); i++) {
    if(i >= max_printed) break;
    coords.push_back((*points[i])(0));
  }
  os << coords;
  if(points.size() > max_printed) {
    os << " ( + " << (points.size() - max_printed) << " more...)";
  }
}

Real time_diff(timeval start, timeval end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1e-6;
}

int main(int argc, char** argv) {
  LibMeshInit init(argc, argv);
  
  if (argc != 5) {
    libmesh_error_msg("Usage: " << argv[0]
        << " width halo_pad num_particles num_reps");
  }
  int width = std::atoi(argv[1]);
  Real halo_pad = std::atof(argv[2]);
  int num_particles = std::atoi(argv[3]);
  int num_reps = std::atoi(argv[4]);
  
  std::ostringstream sout;
  ParallelMesh mesh(init.comm());
  mesh.partitioner().reset(new CentroidPartitioner(CentroidPartitioner::X));
  MeshTools::Generation::build_cube(mesh, width, 1, 1, 0, width, 0, 1, 0, 1);
  mesh.print_info();
  Particle::PSerializer serializer;
  std::vector<Particle*> particles;

  BoundingBox processor_box
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  double minX = processor_box.min()(0);
  double maxX = processor_box.max()(0);
  double interval = width/(double)num_particles;
  int minI = (int)ceil(minX/interval - .5);
  int maxI = (int)ceil(maxX/interval - .5);
  if(maxX == width) maxI = num_particles;
  for(int i = minI; i < maxI; i++) {
    Point point((i + .5)*interval, .5, .5);
    particles.push_back(new Particle(point, point(0)*10));
  }

  std::vector<Particle*> inbox;
  std::vector<std::vector<Particle*> > result;

  HaloManager* hm = NULL;
  for(int c = 0; c < num_reps; c++) {
    if(hm != NULL) delete hm;
    for(unsigned int i = 0; i < inbox.size(); i++) {
      delete inbox[i];
    }
    inbox.clear();
    result.clear();
    hm = new HaloManager(mesh, halo_pad);
    hm->set_serializer(serializer);
    hm->find_particles_in_halos(
        reinterpret_cast<std::vector<Point*>& >(particles),
        reinterpret_cast<std::vector<Point*>& >(inbox),
        reinterpret_cast<std::vector<std::vector<Point*> >& >(result));
  }
  
  sout << "======== Processor " << mesh.processor_id() << " ========\n";
  sout << "Processor Box: " << processor_box << "\n";
  sout << "Halo Pad: " << halo_pad << "\n";
  sout << "Neighbors: " << hm->neighbor_processors() << "\n";
  sout << "Halo Neighbors: " << hm->box_halo_neighbor_processors() << "\n";
  sout << "Particles Inbox: ";
  print_x_coords(sout, inbox);
  sout << "\n";
  sout << "Particle Groups:\n";
  for(unsigned int i = 0; i < result.size(); i++) {
    for(unsigned int j = 0; j < result[i].size(); j++) {
      libmesh_assert(result[i][j]->get_value() == 10*(*result[i][j])(0));
    }
    sout << "  " << (*particles[i])(0) << ": ";
    print_x_coords(sout, result[i]);
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

  for(unsigned int i = 0; i < inbox.size(); i++) {
    delete inbox[i];
  }
  delete hm;

  return 0;
}

