#include "libmesh/point_tree.h"
#include "libmesh/libmesh_common.h"
#include "libmesh/mesh_tools.h"
#include <algorithm>

using namespace libMesh;
using MeshTools::BoundingBox;

namespace { // anonymous namespace for helper classes/functions

const int max_points_in_leaf = 16;

class PointComp {
public:
  PointComp(int axis) : axis(axis) {}
  bool operator()(const Point* a, const Point* b) {
    return (*a)(axis) < (*b)(axis);
  }
private:
  int axis;
};

class AxisComp {
public:
  PointComp(const int* splitCounts) : splitCounts(splitCounts) {}
  bool operator()(int a, int b) {
    if(splitCounts[a] == splitCounts[b]) return a < b;
    return splitCounts[a] < splitCounts[b];
  }
private:
  const int* splitCounts;
};

}; // end anonymous namespace

class PointTree::PTNode {
public:
  PTNode(int splitCounts[LIBMESH_DIM]);
  ~PTNode();
  void insert(Point* point);
  void find(const BoundingBox& box, std::vector<Point*>& result);

private:
  PointTree(const PointTree& other) {libmesh_error();}
  PointTree& operator=(const PointTree& other) {libmesh_error();}
  
  void refine_leaf();
  int select_axis() const;
  bool is_leaf();
  
  int splitCounts[LIBMESH_DIM];
  int axis;
  Real pivot;
  PTNode* loChild;
  PTNode* hiChild;
  std::vector<Point*> points;
};

PointTree::PTNode::PTNode(int splitCounts[LIBMESH_DIM])
  : splitCounts(splitCounts), loChild(NULL), hiChild(NULL)
{
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
    if(box.min()(axis) < pivot) loChild.find(box, result);
    if(box.max()(axis) >= pivot) hiChild.find(box, result);
  }
}

void PointTree::PTNode::refine_leaf() {
  if(!is_leaf()) return;
  if(points.size() <= max_points_in_leaf) return;
  axis = select_axis();
  if(axis == LIBMESH_DIM) return;
  std::sort(points.begin(), points.end(), PointComp(axis));
  pivot = (*points[points.size()/2])(axis);
  int newSplitCounts[LIBMESH_DIM] = splitCounts;
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

  std::vector<int> choices();
  for(int i = 0; i < LIBMESH_DIM; i++) {
    if(allowedMap[i]) choices.push_back(i);
  }
  std::sort(choices.begin(), choices.end(), AxisComp(splitCounts));
  for(int i = 0; i < LIBMESH_DIM; i++) {
    if(!on_same_plane(choices[i])) return choices[i];
  }
  return LIBMESH_DIM;
}

bool PointTree::PTNode::is_leaf() {
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

void PointTree::insert(vector<Point*>& points) {
  for(unsigned int i = 0; i < points.size(); i++) insert(points[i]);
}

void PointTree::find(const BoundingBox& box, std::vector<Point*>& result) const {
  node->find(box, result);
}

PointTree::PointTree(const PointTree& other) {libmesh_error();}
PointTree::PointTree& operator=(const PointTree& other) {libmesh_error();}
