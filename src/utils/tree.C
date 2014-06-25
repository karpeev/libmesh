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



// C++ includes

// Local includes
#include "libmesh/tree.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/mesh_base.h"

namespace libMesh
{



// ------------------------------------------------------------
// Tree class method

// constructor
template <unsigned int N>
Tree<N>::Tree (const MeshBase& m,
               unsigned int target_bin_size,
               const Trees::BuildType bt) :
  TreeBase(m),
  root(m,target_bin_size),
  build_type(bt)
{
  // Set the root node bounding box equal to the bounding
  // box for the entire domain.
  root.set_bounding_box (MeshTools::bounding_box(mesh));

  if (build_type == Trees::NODES)
    {
      // Add all the nodes to the root node.  It will
      // automagically build the tree for us.
      MeshBase::const_node_iterator       it  = mesh.nodes_begin();
      const MeshBase::const_node_iterator end = mesh.nodes_end();

      for (; it != end; ++it)
        root.insert (*it);

      // Now the tree contains the nodes.
      // However, we want element pointers, so here we
      // convert between the two.
      std::vector<std::vector<const Elem*> > nodes_to_elem;

      MeshTools::build_nodes_to_elem_map (mesh, nodes_to_elem);
      root.transform_nodes_to_elements (nodes_to_elem);
    }

  else if (build_type == Trees::ELEMENTS)
    {
      // Add all active elements to the root node.  It will
      // automatically build the tree for us.
      MeshBase::const_element_iterator       it  = mesh.active_elements_begin();
      const MeshBase::const_element_iterator end = mesh.active_elements_end();

      for (; it != end; ++it)
        root.insert (*it);
    }

  else if (build_type == Trees::LOCAL_ELEMENTS)
    {
      // Add all active, local elements to the root node.  It will
      // automatically build the tree for us.
      MeshBase::const_element_iterator       it  = mesh.active_local_elements_begin();
      const MeshBase::const_element_iterator end = mesh.active_local_elements_end();

      for (; it != end; ++it)
        root.insert (*it);
    }

  else
    libmesh_error_msg("Unknown build_type = " << build_type);
}

namespace {
  template<unsigned int N>
  MeshTools::BoundingBox find_bounding_box(
      MeshBase::const_node_iterator nodes_begin,
      MeshBase::const_node_iterator nodes_end)
  {
    unsigned int dim = 3;
    switch(N) {
      case 2: dim = 1; break;
      case 4: dim = 2; break;
      case 8: dim = 3; break;
    }
    MeshTools::BoundingBox box;
    MeshBase::const_node_iterator it = nodes_begin;
    for(; it != nodes_end; it++) {
      const Point& p = *(Point*)*it;
      for(unsigned int d = 0; d < dim; d++) {
        box.min()(d) = std::min(box.min()(d), p(d));
        box.max()(d) = std::max(box.max()(d), p(d));
      }
    }
    return box;
  }
}

template <unsigned int N>
Tree<N>::Tree (const MeshBase& m,
      unsigned int target_bin_size,
      MeshBase::const_node_iterator nodes_begin,
      MeshBase::const_node_iterator nodes_end) :
  TreeBase(m),
  root(m,target_bin_size),
  build_type(Trees::INVALID_BUILD_TYPE)
{
  root.set_bounding_box(find_bounding_box<N>(nodes_begin, nodes_end));
  
  MeshBase::const_node_iterator it = nodes_begin;
  for (; it != nodes_end; ++it) {
    root.insert(*it);
  }
}


// copy-constructor is not implemented
template <unsigned int N>
Tree<N>::Tree (const Tree<N>& other_tree) :
  TreeBase   (other_tree),
  root       (other_tree.root),
  build_type (other_tree.build_type)
{
  libmesh_not_implemented();
}






template <unsigned int N>
void Tree<N>::print_nodes(std::ostream& my_out) const
{
  my_out << "Printing nodes...\n";
  root.print_nodes(my_out);
}



template <unsigned int N>
void Tree<N>::print_elements(std::ostream& my_out) const
{
  my_out << "Printing elements...\n";
  root.print_elements(my_out);
}



template <unsigned int N>
const Elem* Tree<N>::find_element(const Point& p) const
{
  return root.find_element(p);
}



template <unsigned int N>
const Elem* Tree<N>::operator() (const Point& p) const
{
  return this->find_element(p);
}

template <unsigned int N>
void Tree<N>::find_nodes (const std::pair<Point, Point>& box,
                 std::vector<const Node*>& result) const
{
  return root.find_nodes(box, result);
}


// ------------------------------------------------------------
// Explicit Instantiations
template class Tree<2>;
template class Tree<4>;
template class Tree<8>;

} // namespace libMesh
