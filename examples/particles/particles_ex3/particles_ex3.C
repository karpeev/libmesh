#include "libmesh/libmesh_common.h"
#include "libmesh/libmesh.h"
#include "libmesh/halo_manager.h"
#include "libmesh/serializer.h"
#include "libmesh/mesh_generation.h"
#include <iostream>
#include <istream>
#include <ostream>
#include <vector>

class MyParticle : public Point {
public:
  MyParticle(Point& point, double value) : Particle(point), value(value)
  {}

  MyParticle() {}

  class PSerializer : public Serializer<Point*> {
  public:
    void read(std::istream& stream, Point*& buffer) const {
      buffer = new MyParticle();
      stream.read((char*)buffer, sizeof(*buffer));
    }
    void write(std::ostream& stream, const Point*& buffer) const {
      stream.write((char*)buffer, sizeof(*buffer));
    }
  }

private:
  double val;
};

int main(int argc, char** argv) {
  LibMeshInit init(argc, argv);
  ParallelMesh mesh(init.comm());
  MeshTools::Generation::build_cube(mesh, 360, 1, 1, 0, 360, 0, 1, 0, 1);
  mesh.print_info();
  Real haloPad = 85;
  HaloManager hm(mesh, haloPad);
  std::vector<Particle*> particles;
  typedef MeshBase::element_iterator ElemIter_t;
  for(ElemIter_t it = mesh.local_elements_begin();
      it != mesh.local_elements_end(); it++)
  {
    particles.push_back(new MyParticle(it->centroid(), it->unique_id()));
  }
  std::vector<MyParticle*> inbox;
  std::vector<std::vector<MyParticle*> > result;
  PSerializer serializer;
  hm.find_particles_in_halos(particles, serializer,
      (std::vector<Point*>)inbox,
      (std::vector<std::vector<Point*> >)result);
}

