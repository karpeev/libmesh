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


// Local Includes -----------------------------------
#include "libmesh/point_tree.h"

// C++ Includes   -----------------------------------
#include <algorithm>

namespace libMesh {

using MeshTools::BoundingBox;

namespace { // anonymous namespace for helper classes/functions

Real sq_dist(const Point& a, const Point& b) {
  Real result = 0.0;
  for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
    Real diff = a(d) - b(d);
    result += diff*diff;
  }
  return result;
}

/**
 * A comparator used for ordering points along a given axis.
 *
 * \author  Matthew D. Michelotti
 */
class PointComp {
public:
  PointComp(int axis) : axis(axis) {}
  bool operator()(const Point* a, const Point* b) {
    return (*a)(axis) < (*b)(axis);
  }
private:
  int axis;
};

class BallPredicate {
public:
  BallPredicate(Point* center, Real radius, bool include_center)
      : center(center), sq_rad(radius*radius), include_center(include_center)
  {
  }

  bool test(const Point* point) {
    if(!include_center && point == center) return false;
    return sq_dist(*point, *center) <= sq_rad;
  }

private:
  Point* center;
  Real sq_rad;
  bool include_center;
};

class TruePredicate {
public:
  bool test(const Point* point) {
    (void)point;
    return true;
  }
};

/**
 * @returns a pivot at the median of of the \p points vector along the
 * \p axis.  The \p points are assumed to be sorted in increasing order
 * along the \p axis, and we also assume that not all points have
 * the same value on that axis. Will not choose a pivot that has no effect
 * (i.e. after splitting along pivot, at least one point will fall on either
 * side of the pivot).
 */
Real select_pivot(const std::vector<Point*>& points, int axis) {
  libmesh_assert(points.size() > 0);
  libmesh_assert((*points[0])(axis) < (*points.back())(axis));
  unsigned int i = points.size()/2;
  while((*points[i])(axis) == (*points[0])(axis)) i++;
  while((*points[i])(axis) == (*points[i-1])(axis)) i--;
  Real lo = (*points[i-1])(axis);
  Real hi = (*points[i])(axis);
  Real result = .5*(lo + hi);
  if(result > lo && result < hi) return result;
  return hi;
}

} // end anonymous namespace


/**
 * This is the \p PTNode class.  Objects of this class represent
 * nodes in a PointTree.  This is a k-d tree.
 *
 * \author  Matthew D. Michelotti
 */
class PointTree::PTNode {
public:
  /**
   * Constructor.  The \p split_counts array is the number of splits
   * that have occurred along each axis in this node's ancestors.
   */
  PTNode(const int* split_counts, unsigned int max_points_in_leaf);
  
  /**
   * Destructor.
   */
  ~PTNode();
  
  /**
   * Inserts a single \p point into this node.  Tree leaves will not be
   * refined until needed.
   */
  void insert(Point* point);
  
  /**
   * Efficiently finds all points in the tree that are contained within
   * the given \p box.  These points are placed in the \p result vector.
   */
  template<class T>
  void find(const BoundingBox& box, T& predicate,
      std::vector<Point*>& result);
      
  void to_vector(std::vector<Point*>& result);
  
  /**
   * Prints the contents of the subtree in a hierarchical manner.
   * The \p depth input is the depth of this node.
   */
  void print(int depth) const;

private:

  /**
   * Overridden copy constructor that should never be used.
   */
  PTNode(const PTNode& other) {
    libmesh_error();
    (void)other;
  }

  /**
   * Overridden assignment operator that should never be used.
   */
  PTNode& operator=(const PTNode& other) {
    libmesh_error();
    (void)other;
  }
  
  /**
   * If this is a leaf node and has more points than max_points_in_leaf,
   * then refines this leaf node into two new leaf node children.
   * Splits along median of points for the selected axis.
   * Will not refine new child nodes.
   */
  void refine_leaf();
  
  /**
   * @returns the axis with the fewest number of splits so far.
   * If all points have the same value for some axis, will not return
   * that axis.  If there is no suitable axis, returns LIBMESH_DIM.
   */
  int select_axis() const;
  
  /**
   * Returns true if this node is a leaf node.
   */
  bool is_leaf() const;
  
  /**
   * The number of splits that have occurred along each axis in
   * this node's ancestors.
   */
  int split_counts[LIBMESH_DIM];
  
  /**
   * The axis of this node's split (if this is not a leaf node).
   */
  int axis;
  
  /**
   * The location of this node's split (if this is not a leaf node).
   */
  Real pivot;
  
  /**
   * The first child of this node (NULL if this is a leaf node).
   */
  PTNode* lo_child;
  
  /**
   * The second child of this node (NULL if this is a leaf node).
   */
  PTNode* hi_child;
  
  /**
   * The points contained in this node (if this is a leaf node).
   */
  std::vector<Point*> points;

  /**
   * The maximum capacity of a leaf PTNode before it should be refined
   */
  unsigned int max_points_in_leaf;
};

PointTree::PTNode::PTNode(const int* split_counts,
    unsigned int max_points_in_leaf)
    : lo_child(NULL), hi_child(NULL)
{
  if(max_points_in_leaf <= 0) this->max_points_in_leaf = 16;
  else this->max_points_in_leaf = max_points_in_leaf;
  for(int i = 0; i < LIBMESH_DIM; i++) {
    this->split_counts[i] = split_counts[i];
  }
}

PointTree::PTNode::~PTNode() {
  if(lo_child != NULL) delete lo_child;
  if(hi_child != NULL) delete hi_child;
}

void PointTree::PTNode::insert(Point* point) {
  if(is_leaf()) {
    points.push_back(point);
  }
  else {
    if((*point)(axis) < pivot) lo_child->insert(point);
    else hi_child->insert(point);
  }
}

template<class T>
void PointTree::PTNode::find(const BoundingBox& box, T& predicate,
    std::vector<Point*>& result)
{
  refine_leaf();
  if(is_leaf()) {
    typedef std::vector<Point*>::iterator iter_t;
    for(iter_t it = points.begin(); it != points.end(); it++) {
      if(box.contains_point(**it) && predicate.test(*it)) {
        result.push_back(*it);
      }
    }
  }
  else {
    if(box.min()(axis) < pivot) lo_child->find(box, predicate, result);
    if(box.max()(axis) >= pivot) hi_child->find(box, predicate, result);
  }
}

void PointTree::PTNode::to_vector(std::vector<Point*>& result) {
  if(is_leaf()) {
    result.insert(result.end(), points.begin(), points.end());
  }
}

void PointTree::PTNode::print(int depth) const {
  for(int c = 0; c < depth; c++) libMesh::out << "  ";
  if(is_leaf()) {
    libMesh::out << "[";
    for(unsigned int i = 0; i < points.size(); i++) {
      libMesh::out << "(";
      for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
        libMesh::out << (*points[i])(d);
        if(d < LIBMESH_DIM - 1) libMesh::out << ", ";
      }
      libMesh::out << ")";
      if(i < points.size() - 1) libMesh::out << ", ";
    }
    libMesh::out << "]" << std::endl;
  }
  else {
    libmesh_assert(points.size() == 0);
    libMesh::out << "axis " << axis << " pivot " << pivot << std::endl;
    lo_child->print(depth + 1);
    hi_child->print(depth + 1);
  }
}

void PointTree::PTNode::refine_leaf() {
  if(!is_leaf()) return;
  if(points.size() <= max_points_in_leaf) return;
  axis = select_axis();
  if(axis == LIBMESH_DIM) return;
  std::sort(points.begin(), points.end(), PointComp(axis));
  pivot = select_pivot(points, axis);
  int new_split_counts[LIBMESH_DIM];
  for(int i = 0; i < LIBMESH_DIM; i++) new_split_counts[i] = split_counts[i];
  new_split_counts[axis]++;
  lo_child = new PTNode(new_split_counts, max_points_in_leaf);
  hi_child = new PTNode(new_split_counts, max_points_in_leaf);
  for(unsigned int i = 0; i < points.size(); i++) insert(points[i]);
  points.clear();
}

int PointTree::PTNode::select_axis() const {
  //determine which axes we are allowed to split along
  //(cannot split along an axes in which the split will have no effect)
  bool allowed_map[LIBMESH_DIM];
  for(int i = 0; i < LIBMESH_DIM; i++) allowed_map[i] = false;
  for(unsigned int i = 1; i < points.size(); i++) {
    for(unsigned int axis = 0; axis < LIBMESH_DIM; axis++) {
      if(allowed_map[axis]) continue;
      if((*points[i])(axis) != (*points[0])(axis)) allowed_map[axis] = true;
    }
  }

  //choose the allowed axis with the fewest splits so far
  int result = LIBMESH_DIM;
  for(int i = 0; i < LIBMESH_DIM; i++) {
    if(!allowed_map[i]) continue;
    if(result == LIBMESH_DIM || split_counts[i] < split_counts[result]) {
      result = i;
    }
  }
  return result;
}

bool PointTree::PTNode::is_leaf() const {
  return lo_child == NULL;
}

PointTree::PointTree(unsigned int max_points_in_leaf) {
  int split_counts[LIBMESH_DIM];
  for(int i = 0; i < LIBMESH_DIM; i++) split_counts[i] = 0;
  root = new PTNode(split_counts, max_points_in_leaf);
}

PointTree::~PointTree() {
  delete root;
}

void PointTree::insert(Point* point) {
  root->insert(point);
  for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
    bounding_box.min()(d) = std::min(bounding_box.min()(d), (*point)(d));
    bounding_box.max()(d) = std::max(bounding_box.max()(d), (*point)(d));
  }
}

void PointTree::insert(const std::vector<Point*>& points) {
  for(unsigned int i = 0; i < points.size(); i++) insert(points[i]);
}

void PointTree::find_box(const BoundingBox& box, std::vector<Point*>& result) {
  TruePredicate predicate;
  root->find(box, predicate, result);
}

void PointTree::find_ball(Point* center, Real radius,
      std::vector<Point*>& result, bool include_center)
{
  BoundingBox box;
  for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
    box.min()(d) = (*center)(d) - radius;
    box.max()(d) = (*center)(d) + radius;
  }
  BallPredicate predicate(center, radius, include_center);
  root->find(box, predicate, result);
}
      
void PointTree::to_vector(std::vector<Point*>& result) {
  return root->to_vector(result);
}

const MeshTools::BoundingBox& PointTree::get_bounding_box() {
  return bounding_box;
}

void PointTree::print() const {
  root->print(0);
}

PointTree::PointTree(const PointTree& other) {
  libmesh_error();
  (void)other;
}

PointTree& PointTree::operator=(const PointTree& other) {
  libmesh_error();
  (void)other;
}

} // end namespace libMesh
