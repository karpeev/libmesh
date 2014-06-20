#include "libmesh/libmesh.h"
#include "libmesh/parallel_mesh.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/mesh_generation.h"
#include <iostream>
#include <sstream>
#include <vector>

using namespace libMesh;
using MeshTools::BoundingBox;

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

int main(int argc, char** argv) {
  LibMeshInit init(argc, argv);
  std::ostringstream sout;
  ParallelMesh mesh(init.comm());
  MeshTools::Generation::build_cube(mesh, 360, 1, 1, 0, 360, 0, 1, 0, 1);
  mesh.print_info();
  sout << "======== Processor " << mesh.processor_id() << " ========\n";
  BoundingBox halo
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  sout << "Processor Box: " << halo << "\n";
  double haloPad = 85;
  sout << "Halo Pad: " << haloPad << "\n";
  std::vector<int> neighbors;
  MeshTools::find_neighbor_proc_ids(mesh, neighbors);
  sout << "Neighbors: " << neighbors << "\n";
  std::vector<int> haloPids;
  MeshTools::parallel_find_box_halo_proc_ids(mesh, haloPad, haloPids);
  sout << "Halo Processors: " << haloPids << "\n";

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

