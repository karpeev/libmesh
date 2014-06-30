
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

//FIXME make sure box_halo_neighbors is symmetric across processors, i.e. if processor x has neighbor processor y, then processor y has neighbor processor x
HaloManager::HaloManager(const MeshBase& mesh, double halo_pad)
    : found_box_halo_neighbors(false), halo_pad(halo_pad)
{
  find_neighbor_processors(mesh, neighbors);
  
  BoundingBox halo
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  for(int d = 0; d < LIBMESH_DIM; d++) {
    halo.min()(d) -= halo_pad;
    halo.max()(d) += halo_pad;
  }
  HaloNeighborsExtender extender(&mesh);
  extender.resolve(halo, box_halo_neighbors);
}

const std::vector<int>& HaloManager::neighbor_processors() const {
  return neighbors;
}

const std::vector<int>& HaloManager::box_halo_neighbor_processors() const {
  return box_halo_neighbors;
}

void HaloManager::find_particles_in_halos(
    const std::vector<Particle*> particles,
    std::vector<Particle*> particle_inbox,
    Particle* (*constructor)(),
    std::vector<std::vector<Particle*> > result) const
{
  BoundingBox pointsHalo = find_bounding_box(particles);
  Request reqs[box_halo_neighbors.size()];
  const Communicator& comm;
  for(unsigned int i = 0; i < box_halo_neighbors.size(); i++) {
    comm.send(box_halo_neighbors[i], pointsHalo, reqs[i], requestTag);
  }
  //FIXME left off here...
}

void HaloManager::find_neighbor_processors(const MeshBase& mesh,
    std::vector<int>& result)
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

BoundingBox HaloManager::find_bounding_box(
    const std::vector<Particle*> particles) const
{
  if(particles.empty()) return BoundingBox(Point(0,0,0), Point(0,0,0));
  BoundingBox box(*(Point*)particles[0], *(Point*)particles[0]);
  for(unsigned int i = 0; i < particles.size(); i++) {
    for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
      box.min()(d) = std:min(box.min()(d), particles[i](d));
      box.max()(d) = std:max(box.max()(d), particles[i](d));
    }
  }
  return box;
}