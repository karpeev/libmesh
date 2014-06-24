#include "libmesh/halo.h"
#include "libmesh/elem.h"
#include "libmesh/mesh_base.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/neighbors_extender.h"

#include <set>

using namespace libMesh;
using MeshTools::BoundingBox;
using Parallel::NeighborsExtender;

namespace { // anonymous namespace for helper classes/functions

BoundingBox bounding_box(const Elem* elem) {
  BoundingBox box;
  for(unsigned int i = 0; i < elem->n_nodes(); i++) {
    const Point& point = elem->point(i);
    for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
      box.min()(d) = std::min(box.min()(d), point(d));
      box.max()(d) = std::max(box.max()(d), point(d));
    }
  }
  return box;
}

class HaloNeighborsExtender : public NeighborsExtender {
  public:
    HaloNeighborsExtender(const MeshBase* mesh)
        : NeighborsExtender(mesh->comm())
    {
      processor_id_type pid = mesh->processor_id();
      MeshBase::const_element_iterator it = mesh->not_local_elements_begin();
      for(; it != mesh->not_local_elements_end(); it++) {
        const Elem* elem = *it;
        if(elem->is_semilocal(pid)) ghostElems.push_back(elem);
      }
      std::vector<int> neighbors;
      Halo::find_neighbor_proc_ids(*mesh, neighbors);
      setNeighbors(neighbors);
    }
    
    void resolve(const BoundingBox& halo, std::vector<int>& result) {
      NeighborsExtender::resolve(sizeof(BoundingBox),
          (const char*)&halo, result);
    }
    
  protected:
    void testInit(int root, int testDataSize, const char* testData) {
      const BoundingBox& box = *(const BoundingBox*)testData;
      for(int i = 0; i < (int)ghostElems.size(); i++) {
        const Elem* elem = ghostElems[i];
        BoundingBox elemBox = bounding_box(elem);
        if(box.intersect(elemBox)) testEdges.insert(elem->processor_id());
      }
    }
    
    bool testEdge(int neighbor) {return testEdges.count(neighbor) > 0;}
    inline bool testNode() {return true;}
    void testClear() {testEdges.clear();}
    
  private:
    std::vector<const Elem*> ghostElems;
    std::set<int> testEdges;
};

} // end anonymous namespace

void Halo::find_neighbor_proc_ids(const MeshBase &mesh,
    std::vector<int> &result)
{
  std::set<int> neighborSet;
  processor_id_type pid = mesh.processor_id();
  MeshBase::const_element_iterator it = mesh.not_local_elements_begin();
  for(; it != mesh.not_local_elements_end(); it++) {
    const Elem* elem = *it;
    if(elem->is_semilocal(pid)) neighborSet.insert(elem->processor_id());
  }
  std::copy(neighborSet.begin(), neighborSet.end(),
            std::back_inserter(result));
}

//TODO make sure result of this function is symmetric, i.e. if processor x has neighbor processor y, then processor y has neighbor processor x
void Halo::parallel_find_bounding_box_halo_proc_ids(const MeshBase& mesh,
    Real halo_pad, std::vector<int>& result)
{
  BoundingBox halo
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  for(int d = 0; d < LIBMESH_DIM; d++) {
    halo.min()(d) -= halo_pad;
    halo.max()(d) += halo_pad;
  }
  HaloNeighborsExtender extender(&mesh);
  extender.resolve(halo, result);
}
