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
#include <iostream>
#include <istream>
#include <ostream>
#include <vector>
#include <sstream>
#include <sys/time.h>

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
  
  Real getValue() const {return value;}
  
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

std::ostream& operator<<(std::ostream& os, const Particle* particle) {
  os << "(";
  for(unsigned int i = 0; i < LIBMESH_DIM; i++) {
    os << (*particle)(i);
    if(i < LIBMESH_DIM - 1) os << ", ";
  }
  os << ")";
  return os;
}

void print_x_coords(std::ostream& os, const std::vector<Particle*> points) {
  std::vector<Real> coords;
  for(unsigned int i = 0; i < points.size(); i++) {
    coords.push_back((*points[i])(0));
  }
  os << coords;
}

Real time_diff(timeval start, timeval end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1e-6;
}

int main(int argc, char** argv) {
  int num_reps = 100;
  timeval time1, time2, time3;
  gettimeofday(&time1, NULL);
  //timeval time1 = std::time(NULL);
  LibMeshInit init(argc, argv);
  
  if (argc != 3) {
    libmesh_error_msg("Usage: " << argv[0] << " width particles_per_cell");
  }
  unsigned int width = std::atoi(argv[1]);
  unsigned int particles_per_cell = std::atoi(argv[2]);
  
  std::ostringstream sout;
  ParallelMesh mesh(init.comm());
  MeshTools::Generation::build_cube(mesh, width, 1, 1, -.5,
      width - .5, 0, 1, 0, 1);
  mesh.print_info();
  Real haloPad = 7.1;
  HaloManager hm(mesh, haloPad);
  std::vector<Particle*> particles;
  typedef MeshBase::element_iterator ElemIter_t;
  for(ElemIter_t it = mesh.local_elements_begin();
      it != mesh.local_elements_end(); it++)
  {
    Point centroid = (*it)->centroid();
    for(unsigned int i = 0; i < particles_per_cell; i++) {
      Real y = .5 + (i - .5*(particles_per_cell - 1))*.8/particles_per_cell;
      Point point(centroid(0), y, y);
      particles.push_back(new Particle(point, point(0)*10));
    }
    //particles.push_back(new Particle(centroid, centroid(0)*10));
  }
  std::vector<Particle*> inbox;
  std::vector<std::vector<Particle*> > result;
  Particle::PSerializer serializer;
  //std::time_t time2 = std::time(NULL);
  gettimeofday(&time2, NULL);
  for(int c = 0; c < num_reps; c++) {
    for(unsigned int i = 0; i < inbox.size(); i++) {
      delete inbox[i];
    }
    inbox.clear();
    result.clear();
    hm.find_particles_in_halos(
        reinterpret_cast<std::vector<Point*>& >(particles),
        serializer,
        reinterpret_cast<std::vector<Point*>& >(inbox),
        reinterpret_cast<std::vector<std::vector<Point*> >& >(result));
  }
  //std::time_t time3 = std::time(NULL);
  gettimeofday(&time3, NULL);
  
  //sout << time1 << " " << time2 << " " << time3 << " " << CLOCKS_PER_SEC << "\n";
  Real setup_time = time_diff(time1, time2);//(time2 - time1)/((Real)CLOCKS_PER_SEC);
  Real comm_time = time_diff(time2, time3)/num_reps;//(time3 - time2)/((Real)CLOCKS_PER_SEC);
  
  sout << "======== Processor " << mesh.processor_id() << " ========\n";
  BoundingBox processor_box
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  sout << "Processor Box: " << processor_box << "\n";
  sout << "Halo Pad: " << haloPad << "\n";
  sout << "Neighbors: " << hm.neighbor_processors() << "\n";
  sout << "Halo Neighbors: " << hm.box_halo_neighbor_processors() << "\n";
  sout << "Particles Inbox: " << inbox;
  //print_x_coords(sout, inbox);
  sout << "\n";
  sout << "Particle Groups:\n";
  for(unsigned int i = 0; i < result.size(); i++) {
    for(unsigned int j = 0; j < result[i].size(); j++) {
      libmesh_assert(result[i][j]->getValue() == 10*(*result[i][j])(0));
    }
    sout << "  " << (*particles[i])(0) << ": " << result[i];
    //print_x_coords(sout, result[i]);
    sout << "\n";
  }
  sout << "Setup time: " << setup_time << " seconds\n";
  sout << "Communication time: " << comm_time << " seconds\n";

  std::string textStr = sout.str();
  std::vector<char> text(textStr.begin(), textStr.end());
  text.push_back('\0');
  init.comm().gather(0, text);
  int ci = 0;
  while(ci < (int)text.size()) {
    std::cout << &text[ci] << std::endl;
    while(text[ci++] != '\0');
  }
}

