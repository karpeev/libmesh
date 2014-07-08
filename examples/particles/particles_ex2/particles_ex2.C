#include "libmesh/libmesh_common.h"
#include "libmesh/libmesh.h"
#include "libmesh/point_tree.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/serial_mesh.h"
#include <iostream>
#include <sstream>
#include <vector>

using namespace libMesh;
using MeshTools::BoundingBox;

std::ostream& operator<<(std::ostream& os, const BoundingBox& box) {
  os << "(" << box.min() << ", " << box.max() << ")";
  return os;
}

int main(int argc, char** argv) {
  LibMeshInit init(argc, argv);
  SerialMesh mesh(init.comm());
  MeshTools::Generation::build_cube(mesh, 360, 1, 1, 0, 360, 0, 1, 0, 1);
  mesh.print_info();
  BoundingBox halo;
  halo.min() = Point(159.5, -1, -1);
  halo.max() = Point(200.5, 2, 2);
  std::cout << "Halo: " << halo << std::endl;
  PointTree tree;
  typedef MeshBase::node_iterator iter_t;
  for(iter_t it = mesh.nodes_begin(); it != mesh.nodes_end(); it++) {
    tree.insert((Point*)*it);
  }
  std::cout << std::endl;
  std::vector<Point*> result;
  tree.find(halo, result);
  std::cout << "Halo Nodes: " << std::endl;
  for(unsigned int i = 0; i < result.size(); i++) {
    std::cout << "  " << *result[i] << std::endl;
  }
}

