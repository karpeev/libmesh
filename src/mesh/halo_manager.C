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
#include "libmesh/libmesh_logging.h"

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
  Real sq_dist = 0.0;
  for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
    Real diff = (*a)(d) - (*b)(d);
    sq_dist += diff*diff;
  }
  return std::sqrt(sq_dist);
}

/**
 * Writes the \p points vector to the \p bytes buffer using \p serializer.
 */
void write(const std::vector<Point*>& points, std::string& bytes,
    const Serializer<Point*>& serializer)
{
  std::ostringstream stream;
  unsigned int size = points.size();
  stream.write((char*)&size, sizeof(size));
  for(unsigned int i = 0; i < size; i++) {
    serializer.write(stream, points[i]);
  }
  bytes = stream.str();
}

/**
 * Reads the \p points vector from the \p bytes buffer using \p serializer.
 */
void read(std::vector<Point*>& points, const std::string& bytes,
    const Serializer<Point*>& serializer)
{
  std::istringstream stream(bytes);
  unsigned int size;
  stream.read((char*)&size, sizeof(size));
  unsigned int i = points.size();
  points.resize(i + size);
  for(; i < points.size(); i++) {
    serializer.read(stream, points[i]);
  }
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
        if(elem->is_semilocal(pid)) ghost_elems.push_back(elem);
      }
      set_neighbors(neighbors);
    }
    
    /**
     * Finds the processor IDs of processors that overlap the given \p halo,
     * and places these values in the \p result vector.  This is a
     * collective operation.
     */
    void resolve(const BoundingBox& halo, std::vector<int>& result) {
      std::vector<int> in_neighbors;
      NeighborsExtender::resolve(sizeof(BoundingBox),
          (const char*)&halo, result, in_neighbors);
      NeighborsExtender::intersect(result, in_neighbors);
      result.erase(std::find(result.begin(), result.end(), pid));
    }
    
  protected:
    void test(int root, int test_data_size, const char* test_data,
        bool& node_pass, std::set<int>& neighbors_pass)
    {
      (void)root;
      (void)test_data_size;
      node_pass = true;
      const BoundingBox& box = *(const BoundingBox*)test_data;
      for(int i = 0; i < (int)ghost_elems.size(); i++) {
        const Elem* elem = ghost_elems[i];
        BoundingBox elem_box = bounding_box(elem);
        if(box.intersect(elem_box)) {
          neighbors_pass.insert(elem->processor_id());
        }
      }
    }
    
  private:
  
    /**
     * Elements belonging to neighboring processors.
     */
    std::vector<const Elem*> ghost_elems;
    
    /**
     * The processor ID of this processor.
     */
    processor_id_type pid;
};

} // end anonymous namespace

HaloManager::HaloManager(const MeshBase& mesh, Real halo_pad)
    : halo_pad(halo_pad), serializer(NULL), mesh(mesh), comm(mesh.comm()),
      tag_request(mesh.comm().get_unique_tag(15382)),
      tag_response(mesh.comm().get_unique_tag(15383))
{
  START_LOG("constructor", "HaloManager");

  find_neighbor_processors(mesh, neighbors);
  
  BoundingBox halo
      = MeshTools::processor_bounding_box(mesh, mesh.processor_id());
  pad_box(halo);
  HaloNeighborsExtender extender(&mesh, neighbors);
  //TODO If we are using a SerialMesh, perhaps do not communicate with
  //     other processors to find box_halo_neighbors?
  extender.resolve(halo, box_halo_neighbors);
  
  STOP_LOG("constructor", "HaloManager");
}

void HaloManager::set_serializer(const Serializer<Point*>& serializer) {
  this->serializer = &serializer;
}

const std::vector<int>& HaloManager::neighbor_processors() const {
  return neighbors;
}

const std::vector<int>& HaloManager::box_halo_neighbor_processors() const {
  return box_halo_neighbors;
}

void HaloManager::find_particles_in_halos(
    const std::vector<Point*>& halo_centers,
    const std::vector<Point*>& particles,
    std::vector<Point*>& particle_inbox,
    std::vector<std::vector<Point*> >& result) const
{
  START_LOG("find_particles_in_halos", "HaloManager");

  //clear result vectors
  particle_inbox.clear();
  result.clear();
  result.resize(halo_centers.size());
  
  //form tree and communicate particles between processors
  PointTree tree;
  tree.insert(particles);
  BoundingBox box_halo = find_bounding_box(halo_centers);
  pad_box(box_halo);
  comm_particles(box_halo, tree, particle_inbox);
  tree.insert(particle_inbox);
  std::set<Point*> inbox_set(particle_inbox.begin(), particle_inbox.end());
  
  //find particles in each particle's halo, using tree for efficiency
  std::set<Point*> inbox_used;
  std::vector<Point*> buffer;
  for(unsigned int i = 0; i < halo_centers.size(); i++) {
    BoundingBox box;
    box.min() = *halo_centers[i];
    box.max() = *halo_centers[i];
    pad_box(box);
    tree.find(box, buffer);
    for(unsigned int j = 0; j < buffer.size(); j++) {
      Point* point = buffer[j];
      if(point == halo_centers[i]) continue;
      if(distance(point, halo_centers[i]) >= halo_pad) continue;
      result[i].push_back(point);
      if(inbox_set.count(point)) inbox_used.insert(point);
    }
    buffer.clear();
  }
  
  //only return the used particles in the particle_inbox
  for(unsigned int i = 0; i < particle_inbox.size(); i++) {
    if(inbox_used.count(particle_inbox[i]) == 0) delete particle_inbox[i];
  }
  particle_inbox.clear();
  std::copy(inbox_used.begin(), inbox_used.end(),
      std::back_inserter(particle_inbox));
      
  STOP_LOG("find_particles_in_halos", "HaloManager");
}

void HaloManager::find_particles_in_halos(
    const std::vector<Point*>& particles,
    std::vector<Point*>& particle_inbox,
    std::vector<std::vector<Point*> >& result) const
{
  find_particles_in_halos(particles, particles, particle_inbox, result);
}
      
void HaloManager::redistribute_particles(std::vector<Point*>& particles,
    const std::vector<int>& destinations)
{
  libmesh_assert(particles.size() == destinations.size());
  if(neighbors.empty()) return;
  std::vector<Point*> new_particles;
  std::map<int, std::vector<Point*> > outboxes;
  for(unsigned int i = 0; i < neighbors.size(); i++) outboxes[neighbors[i]];
  for(unsigned int i = 0; i < particles.size(); i++) {
    if(destinations[i] == mesh.processor_id()) {
      new_particles.push_back(particles[i]);
    }
    else if(outboxes.count(destinations[i]) > 0) {
      outboxes[destinations[i]].push_back(particles[i]);
    }
    else {
      libMesh::err << "ERROR, pid " << destinations[i]
          << " is not a neighbor processor of pid " << mesh.processor_id()
          << std::endl;
      libmesh_error();
    }
  }
  std::vector<std::string> buffers(neighbors.size());
  std::vector<Request> reqs(neighbors.size());
  for(unsigned int i = 0; i < neighbors.size(); i++) {
    write(outboxes[neighbors[i]], buffers[i], *serializer);
    comm.send(neighbors[i], buffers[i], reqs[i], tag_redistribute);
  }
  for(unsigned int c = 0; c < neighbors.size(); c++) {
    std::string buffer;
    comm.receive(Parallel::any_source, buffer, tag_redistribute);
    read(new_particles, buffer, *serializer);
  }
  particles.swap(new_particles);
  for(unsigned int i = 0; i < reqs.size(); i++) {
    reqs[i].wait();
  }
}
      
void HaloManager::redistribute_particles(std::vector<Point*>& particles)
{
  std::vector<int> destinations(particles.size());
  for(unsigned int i = 0; i < particles.size(); i++) {
    const Elem* elem = mesh.point_locator()(*particles[i]);
    if(elem == NULL) {
      libMesh::err << "No element at point " << *particles[i] << std::endl;
    }
    destinations[i] = elem->processor_id();
  }
  redistribute_particles(particles, destinations);
}

void HaloManager::comm_particles(BoundingBox box_halo, PointTree& tree,
    std::vector<Point*>& particle_inbox) const
{
  //send requests, giving halo to other processors
  std::vector<char> points_halo_buffer(sizeof(BoundingBox));
  (*((BoundingBox*)&points_halo_buffer[0])) = box_halo;
  Request dummy_req;
  for(unsigned int i = 0; i < box_halo_neighbors.size(); i++) {
    comm.send(box_halo_neighbors[i], points_halo_buffer, dummy_req,
        tag_request);
    //NOTE: we do not need to call wait on these requests,
    //      because we recieve responses from these processors
  }
  
  //receive requests and send responses, giving particles to other processors
  std::vector<std::string> outboxes(box_halo_neighbors.size());
  std::vector<Request> reqs(box_halo_neighbors.size());
  std::vector<char> halo_buffer(sizeof(BoundingBox));
  for(unsigned int c = 0; c < box_halo_neighbors.size(); c++) {
    BoundingBox halo;
    int source = comm.receive(
        Parallel::any_source, halo_buffer, tag_request).source();
    halo = *((BoundingBox*)&halo_buffer[0]);
    std::vector<Point*> particles_buffer;
    if(halo.min()(0) != halo.max()(0)) {
      tree.find(halo, particles_buffer);
    }
    write(particles_buffer, outboxes[c], *serializer);
    comm.send(source, outboxes[c], reqs[c], tag_response);
    halo_buffer.clear();
  }
  
  //receive responses
  for(unsigned int c = 0; c < box_halo_neighbors.size(); c++) {
    std::string buffer;
    comm.receive(Parallel::any_source, buffer, tag_response);
    read(particle_inbox, buffer, *serializer);
  }
  
  //wait for all response sends to finish
  for(unsigned int i = 0; i < reqs.size(); i++) {
    reqs[i].wait();
  }
}

void HaloManager::find_neighbor_processors(const MeshBase& mesh,
    std::vector<int>& result)
{
  std::set<int> neighbor_set;
  processor_id_type pid = mesh.processor_id();
  MeshBase::const_element_iterator it = mesh.not_local_elements_begin();
  for(; it != mesh.not_local_elements_end(); it++) {
    const Elem* elem = *it;
    if(elem->is_semilocal(pid)) neighbor_set.insert(elem->processor_id());
  }
  std::copy(neighbor_set.begin(), neighbor_set.end(),
            std::back_inserter(result));
}

void HaloManager::pad_box(BoundingBox& box) const {
  for(unsigned int d = 0; d < LIBMESH_DIM; d++) {
    box.min()(d) -= halo_pad;
    box.max()(d) += halo_pad;
  }
}

} // end namespace libMesh
