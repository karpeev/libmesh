#include "libmesh/halo_manager.h"
#include "libmesh/mesh_base.h"
#include "libmesh/elem.h"
#include "libmesh/point_tree.h"
#include "libmesh/neighbors_extender.h"
#include "libmesh/point.h"
#include "libmesh/parallel.h"

#include <algorithm>

using namespace libMesh;
using MeshTools::BoundingBox;
using Parallel::Request;
using Parallel::MessageTag;

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

class HaloNeighborsExtender : public Parallel::NeighborsExtender {
  public:
    HaloNeighborsExtender(const MeshBase* mesh,
        const std::vector<int>& neighbors) : NeighborsExtender(mesh->comm())
    {
      pid = mesh->processor_id();
      MeshBase::const_element_iterator it = mesh->not_local_elements_begin();
      for(; it != mesh->not_local_elements_end(); it++) {
        const Elem* elem = *it;
        if(elem->is_semilocal(pid)) ghostElems.push_back(elem);
      }
      setNeighbors(neighbors);
    }
    
    void resolve(const BoundingBox& halo, std::vector<int>& result) {
      std::vector<int> inNeighbors;
      NeighborsExtender::resolve(sizeof(BoundingBox),
          (const char*)&halo, result, inNeighbors);
      NeighborsExtender::intersect(result, inNeighbors);
      result.erase(std::find(result.begin(), result.end(), pid));
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
    processor_id_type pid;
};

BoundingBox find_bounding_box(const std::vector<Point*> particles) {
  if(particles.empty()) return BoundingBox(Point(0,0,0), Point(0,0,0));
  BoundingBox box(*(Point*)particles[0], *(Point*)particles[0]);
  for(unsigned int i = 0; i < particles.size(); i++) {
    for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
      box.min()(d) = std::min(box.min()(d), (*particles[i])(d));
      box.max()(d) = std::max(box.max()(d), (*particles[i])(d));
    }
  }
  return box;
}

double distance(const Point* a, const Point* b) {
  Real sqDist = 0.0;
  for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
    Real diff = (*a)(d) - (*b)(d);
    sqDist += diff*diff;
  }
  return std::sqrt(sqDist);
}

} // end anonymous namespace

HaloManager::HaloManager(const MeshBase& mesh, Real halo_pad)
    : halo_pad(halo_pad), comm(mesh.comm()),
      tagRequest(mesh.comm().get_unique_tag(15382)),
      tagResponse(mesh.comm().get_unique_tag(15383))
{
  find_neighbor_processors(mesh, neighbors);
  
  BoundingBox halo
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  pad_box(halo);
  HaloNeighborsExtender extender(&mesh, neighbors);
  //TODO if we are using a SerialMesh, do not communicate with
  //     other processors to find box_halo_neighbors
  extender.resolve(halo, box_halo_neighbors);
}

const std::vector<int>& HaloManager::neighbor_processors() const {
  return neighbors;
}

const std::vector<int>& HaloManager::box_halo_neighbor_processors() const {
  return box_halo_neighbors;
}

void HaloManager::find_particles_in_halos(
    std::vector<Point*>& particles,
    const Serializer<Point*>& particle_serializer,
    std::vector<Point*>& particle_inbox,
    std::vector<std::vector<Point*> >& result) const
{
  particle_inbox.clear();
  result.clear();
  result.resize(particles.size());
  PointTree tree;
  tree.insert(particles);
  comm_particles(particles, tree, particle_serializer, particle_inbox);
  tree.insert(particle_inbox);
  std::vector<Point*> buffer;
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
    buffer.clear();
  }
}

void HaloManager::comm_particles(
    std::vector<Point*>& particles,
    const PointTree& tree,
    const Serializer<Point*>& particle_serializer,
    std::vector<Point*>& particle_inbox) const
{
  BoundingBox pointsHalo = find_bounding_box(particles);
  pad_box(pointsHalo);
  std::vector<char> pointsHaloBuffer(sizeof(BoundingBox));
  (*((BoundingBox*)&pointsHaloBuffer[0])) = pointsHalo;
  Request dummyReq;
  for(unsigned int i = 0; i < box_halo_neighbors.size(); i++) {
    comm.send(box_halo_neighbors[i], pointsHaloBuffer, dummyReq, tagRequest);
    //NOTE: we do not need to call wait on these requests,
    //      because we recieve responses from these processors
  }
  std::string outboxes[box_halo_neighbors.size()];
  Request reqs[box_halo_neighbors.size()];
  std::vector<char> haloBuffer(sizeof(BoundingBox));
  for(unsigned int c = 0; c < box_halo_neighbors.size(); c++) {
    BoundingBox halo;
    int source = comm.receive(
        Parallel::any_source, haloBuffer, tagRequest).source();
    halo = *((BoundingBox*)&haloBuffer[0]);
    std::vector<Point*> particles_buffer;
    if(halo.min()(0) != halo.max()(0)) {
      tree.find(halo, particles_buffer);
    }
    std::ostringstream stream;
    unsigned int size = particles_buffer.size();
    stream.write((char*)&size, sizeof(size));
    for(unsigned int i = 0; i < size; i++) {
      particle_serializer.write(stream, particles_buffer[i]);
    }
    outboxes[c] = stream.str();
    comm.send(source, outboxes[c], reqs[c], tagResponse);
    haloBuffer.clear();
  }
  for(unsigned int c = 0; c < box_halo_neighbors.size(); c++) {
    std::string buffer;
    comm.receive(Parallel::any_source, buffer, tagResponse);
    std::istringstream stream(buffer);
    unsigned int size;
    stream.read((char*)&size, sizeof(size));
    unsigned int i = particle_inbox.size();
    particle_inbox.resize(i + size);
    for(; i < particle_inbox.size(); i++) {
      particle_serializer.read(stream, particle_inbox[i]);
    }
  }
  for(unsigned int i = 0; i < box_halo_neighbors.size(); i++) {
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
