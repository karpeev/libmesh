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


#ifndef LIBMESH_POINT_TREE_H
#define LIBMESH_POINT_TREE_H

// C++ Includes   -----------------------------------
#include <vector>

namespace libMesh
{

// forward declarations

class Point;

namespace MeshTools {
  class BoundingBox;
}

/**
 * This is the \p PointTree class.  This class is used to lookup points
 * efficiently.  It is implemented as a k-d tree.
 *
 * \author  Matthew D. Michelotti
 */

class PointTree {

public:

  /**
   * Constructor.  Creates an empty tree.
   */
  PointTree();

  /**
   * Destructor.
   */
  ~PointTree();

  /**
   * Inserts a single \p point into the tree.  Tree leaves will not
   * be refined until needed (e.g. when find(...) is called).
   */
  void insert(Point* point);

  /**
   * Inserts contents of the \p points vector into the tree.
   * Tree leaves will not be refined until needed (e.g. when find(...)
   * is called).
   */
  void insert(const std::vector<Point*>& points);

  /**
   * Efficiently finds all points in the tree that are contained within
   * the given \p box.  These points are placed in the \p result vector.
   */
  void find(const MeshTools::BoundingBox& box,
      std::vector<Point*>& result);

  /**
   * Prints the contents of the tree in a hierarchical manner.
   */
  void print() const;
  
private:

  /**
   * Overridden copy constructor that should never be used.
   */
  PointTree(const PointTree& other);

  /**
   * Overridden assignment operator that should never be used.
   */
  PointTree& operator=(const PointTree& other);
  
  //forward declaration
  class PTNode;

  /**
   * A pointer to the root node of the tree.
   */
  PTNode* root;
};

} // namespace libMesh

#endif // LIBMESH_POINT_TREE_H
