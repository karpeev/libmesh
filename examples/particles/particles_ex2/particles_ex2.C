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

void print_x_coords(std::ostream& os, const std::vector<Particle*> points) {
  std::vector<Real> coords;
  for(unsigned int i = 0; i < points.size(); i++) {
    coords.push_back((*points[i])(0));
  }
  os << coords;
}

int main(int argc, char** argv) {
  LibMeshInit init(argc, argv);
  std::ostringstream sout;
  ParallelMesh mesh(init.comm());
  MeshTools::Generation::build_cube(mesh, 60, 1, 1, -.5, 59.5, 0, 1, 0, 1);
  mesh.print_info();
  Real haloPad = 7.1;
  HaloManager hm(mesh, haloPad);
  std::vector<Particle*> particles;
  typedef MeshBase::element_iterator ElemIter_t;
  for(ElemIter_t it = mesh.local_elements_begin();
      it != mesh.local_elements_end(); it++)
  {
    Point centroid = (*it)->centroid();
    particles.push_back(new Particle(centroid, centroid(0)*10));
  }
  std::vector<Particle*> inbox;
  std::vector<std::vector<Particle*> > result;
  Particle::PSerializer serializer;
  hm.find_particles_in_halos(
      reinterpret_cast<std::vector<Point*>& >(particles),
      serializer,
      reinterpret_cast<std::vector<Point*>& >(inbox),
      reinterpret_cast<std::vector<std::vector<Point*> >& >(result));
  
  sout << "======== Processor " << mesh.processor_id() << " ========\n";
  BoundingBox processor_box
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  sout << "Processor Box: " << processor_box << "\n";
  sout << "Halo Pad: " << haloPad << "\n";
  sout << "Neighbors: " << hm.neighbor_processors() << "\n";
  sout << "Halo Neighbors: " << hm.box_halo_neighbor_processors() << "\n";
  sout << "Particles Inbox: ";
  print_x_coords(sout, inbox);
  sout << "\n";
  sout << "Particle Groups:\n";
  for(unsigned int i = 0; i < result.size(); i++) {
    for(unsigned int j = 0; j < result[i].size(); j++) {
      libmesh_assert(result[i][j]->getValue() == 10*(*result[i][j])(0));
    }
    sout << "  " << (*particles[i])(0) << ": ";
    print_x_coords(sout, result[i]);
    sout << "\n";
  }

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

