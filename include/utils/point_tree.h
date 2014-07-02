#ifndef LIBMESH_POINT_TREE_H
#define LIBMESH_POINT_TREE_H

#include <vector>

namespace libMesh
{

class Point;

namespace MeshTools {
  class BoundingBox;
}

class PointTree {

public:
  PointTree();
  ~PointTree();
  void insert(Point* point);
  void insert(vector<Point*>& points);
  void find(const MeshTools::BoundingBox& box,
      std::vector<Point*>& result) const;
  
private:
  PointTree(const PointTree& other);
  PointTree& operator=(const PointTree& other);
  
  class PTNode;
  PTNode* node;
};

} // namespace libMesh

#endif // LIBMESH_POINT_TREE_H