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

std::ostream& operator<<(std::ostream& os, const Particle* particle) {
  os << "(";
  for(unsigned int i = 0; i < LIBMESH_DIM; i++) {
    os << (*particle)(i);
    if(i < LIBMESH_DIM - 1) os << ",";
  }
  os << ")";
  return os;
}

Real time_diff(timeval start, timeval end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1e-6;
}

int main(int argc, char** argv) {
  int num_reps = 100;
  timeval time1, time2, time3;
  gettimeofday(&time1, NULL);
  LibMeshInit init(argc, argv);
  
  std::ostringstream sout;
  ParallelMesh mesh(init.comm());
  mesh.read("tile.e");
  mesh.print_info();
  Real halo_pad = 1;
  Particle::PSerializer serializer;
  HaloManager hm(mesh, halo_pad);
  hm.set_serializer(serializer);
  std::vector<Particle*> particles;
  processor_id_type pid = mesh.processor_id();
  typedef MeshBase::element_iterator ElemIter_t;
  typedef Elem::side_iterator SideIter_t;
  for(ElemIter_t elem_it = mesh.local_elements_begin();
      elem_it != mesh.local_elements_end(); elem_it++)
  {
    Elem* elem = *elem_it;
    for(SideIter_t it = elem->boundary_sides_begin();
        it != elem->boundary_sides_end(); it++)
    {
      if((*it)->processor_id() != pid) continue;
      Point centroid = (*it)->centroid();
      particles.push_back(new Particle(centroid, centroid(0)*10));
    }
  }
  std::vector<Particle*> inbox;
  std::vector<std::vector<Particle*> > result;
  gettimeofday(&time2, NULL);
  for(int c = 0; c < num_reps; c++) {
    for(unsigned int i = 0; i < inbox.size(); i++) {
      delete inbox[i];
    }
    inbox.clear();
    result.clear();
    hm.find_particles_in_halos(
        reinterpret_cast<std::vector<Point*>& >(particles),
        reinterpret_cast<std::vector<Point*>& >(inbox),
        reinterpret_cast<std::vector<std::vector<Point*> >& >(result));
  }
  gettimeofday(&time3, NULL);
  
  Real setup_time = time_diff(time1, time2);
  Real comm_time = time_diff(time2, time3)/num_reps;
  
  sout << "======== Processor " << mesh.processor_id() << " ========\n";
  BoundingBox processor_box
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  sout << "Processor Box: " << processor_box << "\n";
  sout << "Halo Pad: " << halo_pad << "\n";
  sout << "Neighbors: " << hm.neighbor_processors() << "\n";
  sout << "Halo Neighbors: " << hm.box_halo_neighbor_processors() << "\n";
  sout << "Particles Inbox: " << inbox;
  sout << "\n";
  sout << "Particle Groups:\n";
  for(unsigned int i = 0; i < result.size(); i++) {
    for(unsigned int j = 0; j < result[i].size(); j++) {
      libmesh_assert(result[i][j]->get_value() == 10*(*result[i][j])(0));
    }
    sout << "  " << particles[i] << ": " << result[i];
    sout << "\n";
  }
  sout << "Setup time: " << setup_time << " seconds\n";
  sout << "Halo finding time: " << comm_time << " seconds\n";

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

  return 0;
}

