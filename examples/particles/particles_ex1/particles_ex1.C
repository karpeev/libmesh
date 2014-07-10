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

