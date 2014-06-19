#include <iostream>
#include <sstream>
#include "libmesh/libmesh.h"
#include "libmesh/mesh_base.h"
#include "libmesh/mesh.h"
#include "libmesh/parallel.h"
#include "libmesh/parallel_mesh.h"
#include "libmesh/elem.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/neighbors_extender.h"

using namespace libMesh;
using MeshTools::BoundingBox;
using Parallel::NeighborsExtender;

//TODO move HaloNeighborsExtender into its own files in LibMesh...

class HaloNeighborsExtender : public NeighborsExtender {
  public:
    HaloNeighborsExtender(const MeshBase* mesh);
    void resolve(const BoundingBox& halo, std::vector<int>& result);
    inline const std::vector<int> getNeighbors() {return neighbors;}
    
  protected:
    void testInit(int root, int testDataSize, const char* testData);
    bool testEdge(int neighbor);
    inline bool testNode() {return true;}
    void testClear();
    
  private:
    std::vector<const Elem*> ghostElems;
    std::set<int> testEdges;
    std::vector<int> neighbors;
};

HaloNeighborsExtender::HaloNeighborsExtender(const MeshBase* mesh) {
  std::set<int> neighborSet;
  processor_id_type pid = mesh->processor_id();
  MeshBase::const_element_iterator it = mesh->not_local_elements_begin();
  for(; it != mesh->not_local_elements_end(); it++) {
    const Elem* elem = *it;
    if(elem->is_semilocal(pid)) {
      ghostElems.push_back(elem);
      neighborSet.insert(elem->processor_id());
    }
  }
  std::copy(neighborSet.begin(), neighborSet.end(),
            std::back_inserter(neighbors));
  setNeighbors(neighbors);
}

void HaloNeighborsExtender::resolve(const BoundingBox& halo,
    std::vector<int>& result)
{
  NeighborsExtender::resolve(sizeof(BoundingBox), (const char*)&halo, result);
}

void HaloNeighborsExtender::testInit(int root, int testDataSize,
    const char* testData)
{
  const BoundingBox& box = *(const BoundingBox*)testData;
  for(int i = 0; i < (int)ghostElems.size(); i++) {
    const Elem* elem = ghostElems[i];
    BoundingBox elemBox;
    for(int j = 0; j < (int)elem->n_nodes(); j++) {
      const Point& point = elem->point(j);
      elemBox.min()(0) = std::min(elemBox.min()(0), point(0));
      elemBox.min()(1) = std::min(elemBox.min()(1), point(1));
      elemBox.min()(2) = std::min(elemBox.min()(2), point(2));
      elemBox.max()(0) = std::max(elemBox.max()(0), point(0));
      elemBox.max()(1) = std::max(elemBox.max()(1), point(1));
      elemBox.max()(2) = std::max(elemBox.max()(2), point(2));
    }
    if(box.intersect(elemBox)) testEdges.insert(elem->processor_id());
  }
}

bool HaloNeighborsExtender::testEdge(int neighbor) {
  return testEdges.count(neighbor) > 0;
}

void HaloNeighborsExtender::testClear() {
  testEdges.clear();
}

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
  Mesh mesh(init.comm());
  MeshTools::Generation::build_cube(mesh, 360, 1, 1, 0, 360, 0, 1, 0, 1);
  mesh.print_info();
  BoundingBox halo = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  sout << "======== Processor " << mesh.processor_id() << " ========\n";
  sout << "Processor Box: " << halo << "\n";
  double haloPad = 85;
  for(int i = 0; i < 3; i++) {
    halo.min()(i) -= haloPad;
    halo.max()(i) += haloPad;
  }
  sout << "Halo Box:      " << halo << "\n";
  HaloNeighborsExtender extender(&mesh);
  std::vector<int> result;
  extender.resolve(halo, result);
  sout << "Neighbors: " << extender.getNeighbors() << "\n";
  sout << "Extended Neighbors: " << result << "\n";

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

