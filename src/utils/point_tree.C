#include "libmesh/point_tree.h"
#include "libmesh/libmesh_common.h"
#include "libmesh/mesh_tools.h"
#include <algorithm>

using namespace libMesh;
using MeshTools::BoundingBox;

namespace { // anonymous namespace for helper classes/functions

const unsigned int max_points_in_leaf = 16;

class PointComp {
public:
  PointComp(int axis) : axis(axis) {}
  bool operator()(const Point* a, const Point* b) {
    return (*a)(axis) < (*b)(axis);
  }
private:
  int axis;
};

} // end anonymous namespace

class PointTree::PTNode {
public:
  PTNode(const int* splitCounts);
  ~PTNode();
  void insert(Point* point);
  void find(const BoundingBox& box, std::vector<Point*>& result);
  void print(int depth) const;

private:
  PTNode(const PTNode& other) {libmesh_error();}
  PTNode& operator=(const PTNode& other) {libmesh_error();}
  
  void refine_leaf();
  int select_axis() const;
  bool is_leaf() const;
  
  int splitCounts[LIBMESH_DIM];
  int axis;
  Real pivot;
  PTNode* loChild;
  PTNode* hiChild;
  std::vector<Point*> points;
};

PointTree::PTNode::PTNode(const int* splitCounts)
  : loChild(NULL), hiChild(NULL)
{
  for(int i = 0; i < LIBMESH_DIM; i++) this->splitCounts[i] = splitCounts[i];
}

PointTree::PTNode::~PTNode() {
  if(loChild != NULL) delete loChild;
  if(hiChild != NULL) delete hiChild;
}

void PointTree::PTNode::insert(Point* point) {
  if(is_leaf()) {
    points.push_back(point);
  }
  else {
    if((*point)(axis) < pivot) loChild->insert(point);
    else hiChild->insert(point);
  }
}

void PointTree::PTNode::find(const BoundingBox& box,
    std::vector<Point*>& result)
{
  refine_leaf();
  if(is_leaf()) {
    typedef std::vector<Point*>::iterator iter_t;
    for(iter_t it = points.begin(); it != points.end(); it++) {
      if(box.contains_point(**it)) result.push_back(*it);
    }
  }
  else {
    if(box.min()(axis) < pivot) loChild->find(box, result);
    if(box.max()(axis) >= pivot) hiChild->find(box, result);
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
    loChild->print(depth + 1);
    hiChild->print(depth + 1);
  }
}

void PointTree::PTNode::refine_leaf() {
  if(!is_leaf()) return;
  if(points.size() <= max_points_in_leaf) return;
  axis = select_axis();
  if(axis == LIBMESH_DIM) return;
  std::sort(points.begin(), points.end(), PointComp(axis));
  pivot = (*points[points.size()/2])(axis);
  int newSplitCounts[LIBMESH_DIM];
  for(int i = 0; i < LIBMESH_DIM; i++) newSplitCounts[i] = splitCounts[i];
  newSplitCounts[axis]++;
  loChild = new PTNode(newSplitCounts);
  hiChild = new PTNode(newSplitCounts);
  for(unsigned int i = 0; i < points.size(); i++) insert(points[i]);
  points.clear();
}

int PointTree::PTNode::select_axis() const {
  bool allowedMap[LIBMESH_DIM];
  for(int i = 0; i < LIBMESH_DIM; i++) allowedMap[i] = false;
  for(unsigned int i = 1; i < points.size(); i++) {
    for(unsigned int axis = 0; axis < LIBMESH_DIM; axis++) {
      if(allowedMap[axis]) continue;
      if((*points[i])(axis) != (*points[0])(axis)) allowedMap[axis] = true;
    }
  }

  int result = LIBMESH_DIM;
  for(int i = 0; i < LIBMESH_DIM; i++) {
    if(!allowedMap[i]) continue;
    if(result == LIBMESH_DIM || splitCounts[i] < splitCounts[result]) {
      result = i;
    }
  }

  return result;
}

bool PointTree::PTNode::is_leaf() const {
  return loChild == NULL;
}

PointTree::PointTree() {
  int splitCounts[LIBMESH_DIM];
  for(int i = 0; i < LIBMESH_DIM; i++) splitCounts[i] = 0;
  node = new PTNode(splitCounts);
}

PointTree::~PointTree() {
  delete node;
}

void PointTree::insert(Point* point) {
  node->insert(point);
}

void PointTree::insert(std::vector<Point*>& points) {
  for(unsigned int i = 0; i < points.size(); i++) insert(points[i]);
}

void PointTree::find(const BoundingBox& box, std::vector<Point*>& result) {
  node->find(box, result);
}

void PointTree::print() const {
  node->print(0);
}

PointTree::PointTree(const PointTree& other) {libmesh_error();}
PointTree& PointTree::operator=(const PointTree& other) {libmesh_error();}
