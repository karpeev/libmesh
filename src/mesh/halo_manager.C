#include "libmesh/halo_manager.h"
#include "libmesh/mesh_base.h"
#include "libmesh/point_tree.h"
#include <algorithm>

using namespace libMesh;
using MeshTools::BoundingBox;

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

BoundingBox find_bounding_box(const std::vector<Point*> particles) const
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

double distance(const Point* a, const Point* b) {
  double sqDist = 0.0;
  for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
    diff = a(d) - b(d);
    sqDist += diff*diff;
  }
  return std::sqrt(sqDist);
}

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
    const std::vector<Point*>& particles,
    const Serializer<Point*>& particle_serializer,
    std::vector<Point*>& particle_inbox,
    std::vector<std::vector<Point*> >& result) const
{
  particle_inbox.clear();
  result.clear();
  PointTree tree();
  tree.insert((std::vector<Point*>)particles);
  comm_particles(particles, tree, particle_serializer, particles_inbox);
  tree.insert((std::vector<Point*>)particle_inbox);
  vector<Point*> buffer;
  for(unsigned int i = 0; i < particles.size(); i++) {
    BoundingBox box;
    box.min() = *particles[i];
    box.max() = *particles[i];
    pad_box(box);
    tree.find(box, buffer);
    for(unsigned int j = 0; j < buffer.size(); j++) {
      Point* point = buffer[j];
      if(point == particles[i]) continue;
      if(distance(point, particles[i]) >= halo_pad) continue;
      result[i].push_back(point);
    }
  }
}

void HaloManager::comm_particles(
    const std::vector<Point*>& particles,
    const PointTree& tree,
    const Serializer<Point*>& particle_serializer,
    std::vector<Point*>& particle_inbox)
{
  BoundingBox pointsHalo = find_bounding_box(particles);
  pad_box(pointsHalo);
  Request dummyReq;
  for(unsigned int i = 0; i < box_halo_neighbors.size(); i++) {
    comm.send(box_halo_neighbors[i], pointsHalo, dummyReq, tagRequest);
    //NOTE: we do not need to call wait on these requests,
    //      because we recieve responses from these processors
  }
  std::string outboxes[box_halo_neighbors.size()];
  Request reqs[box_halo_neighbors.size()];
  for(unsigned int c = 0; c < box_halo_neighbors.size(); c++) {
    BoundingBox halo;
    int source = comm().receive(
        Parallel::any_source, halo, tagRequest).source();
    std::vector<Particle*> particles_buffer;
    if(halo.min()(0) != halo.max()(0)) {
      tree.find_nodes(halo, particles_buffer);
    }
    std::ostringstream stream;
    int size = particles.size();
    stream.write((char*)&size, sizeof(size));
    for(unsigned int i = 0; i < size; i++) {
      particle_serializer.write(stream, particles[i]);
    }
    outboxes[c] = stream.str();
    comm().send(source, outboxes[c], reqs[c], tagResponse);
  }
  for(unsigned int c = 0; c < box_halo_neighbors.size(); c++) {
    std::string buffer;
    comm().receive(Parallel::any_source, buffer, tagResponse);
    std::istringstream stream(buffer);
    int size;
    stream.read((char*)&size, sizeof(size));
    unsigned int i = particle_inbox.size();
    particle_inbox.resize(i + size);
    for(; i < particle_inbox.size(); i++) {
      particle_serializer.read(stream, particle_inbox[i]);
    }
  }
  for(unsigned int i = 0 i < box_halo_neighbors.size(); i++) {
    reqs[i].wait();
  }
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

void HaloManager::pad_box(BoundingBox& box) const {
  for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
    box.min()(d) -= halo_pad;
    box.max()(d) += halo_pad;
  }
}
