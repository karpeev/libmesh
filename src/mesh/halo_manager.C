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
#include "libmesh/halo_manager.h"
#include "libmesh/mesh_base.h"
#include "libmesh/elem.h"
#include "libmesh/point_tree.h"
#include "libmesh/neighbors_extender.h"
#include "libmesh/point.h"
#include "libmesh/parallel.h"

// C++ Includes   -----------------------------------
#include <algorithm>

namespace libMesh {

using MeshTools::BoundingBox;
using Parallel::Request;
using Parallel::MessageTag;

namespace { // anonymous namespace for helper classes/functions

/**
 * @returns a box that bounds the given \p elem
 */
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

/**
 * @returns a box that bounds all points in the \p particles vector
 */
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

/**
 * @returns the distance between points \p a and \p b
 */
Real distance(const Point* a, const Point* b) {
  Real sqDist = 0.0;
  for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
    Real diff = (*a)(d) - (*b)(d);
    sqDist += diff*diff;
  }
  return std::sqrt(sqDist);
}

/**
 * This is the \p HaloNeighborsExtender class.  It is used to form a set
 * of extended neighbors corresponding to which processors are touching this
 * processor's box halo (the bounding box of this processor expanded by
 * a certain amount).
 *
 * \author  Matthew D. Michelotti
 */
class HaloNeighborsExtender : public Parallel::NeighborsExtender {

  public:
  
    /**
     * Constructor.  Uses \p mesh for its Communicator and ghost elements.
     * The \p neighbors vector contains the processor IDs of the immediate
     * neighbors, and it should be consistent with the mesh.
     */
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
    
    /**
     * Finds the processor IDs of processors that overlap the given \p halo,
     * and places these values in the \p result vector.  This is a
     * collective operation.
     */
    void resolve(const BoundingBox& halo, std::vector<int>& result) {
      std::vector<int> inNeighbors;
      NeighborsExtender::resolve(sizeof(BoundingBox),
          (const char*)&halo, result, inNeighbors);
      NeighborsExtender::intersect(result, inNeighbors);
      result.erase(std::find(result.begin(), result.end(), pid));
    }
    
  protected:
    void testInit(int root, int testDataSize, const char* testData) {
      (void)root;
      (void)testDataSize;
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
  
    /**
     * Elements belonging to neighboring processors.
     */
    std::vector<const Elem*> ghostElems;
    
    /**
     * Set of processors that have ghost elements on this processor that
     * overlap the halo given in testInit.
     */
    std::set<int> testEdges;
    
    /**
     * The processor ID of this processor.
     */
    processor_id_type pid;
};

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
  //clear result vectors
  particle_inbox.clear();
  result.clear();
  result.resize(particles.size());
  
  //form tree and communicate particles between processors
  PointTree tree;
  tree.insert(particles);
  comm_particles(particles, tree, particle_serializer, particle_inbox);
  tree.insert(particle_inbox);
  
  //find particles in each particle's halo, using tree for efficiency
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
    PointTree& tree,
    const Serializer<Point*>& particle_serializer,
    std::vector<Point*>& particle_inbox) const
{
  //send requests, giving halo to other processors
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
  
  //receive requests and send responses, giving particles to other processors
  std::vector<std::string> outboxes(box_halo_neighbors.size());
  std::vector<Request> reqs(box_halo_neighbors.size());
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
  
  //receive responses
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
  
  //wait for all response sends to finish
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

} // end namespace libMesh
