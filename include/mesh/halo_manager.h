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


#ifndef LIBMESH_HALO_MANAGER_H
#define LIBMESH_HALO_MANAGER_H

// Local Includes -----------------------------------
#include "libmesh/libmesh_common.h"
#include "libmesh/serializer.h"
#include "libmesh/mesh_tools.h"

// C++ Includes   -----------------------------------
#include <vector>
#include <string>

namespace libMesh {

// forward declarations

class PointTree;
class MeshBase;
class Point;

namespace Parallel {
  class Communicator;
  class MessageTag;
}

/**
 * This is the \p HaloManager class.  This class is used to find points
 * that are within a spherical halo around a point.  Communication
 * with other processors takes place when necessary.  Works with
 * SerialMesh or ParallelMesh.
 *
 * \author  Matthew D. Michelotti
 */

class HaloManager {

public:

  class Opts {
  public:
    Opts();
    bool form_halo_neighbors;
    bool a2a_form_halo_neighbors;
    bool a2a_send_particles;
    bool use_kd_tree;
    bool send_all_particles;
  };

  /**
   * Constructor.  Uses \p mesh to determine connectivity between
   * processors.  \p halo_pad is the radius of the halos.
   */
  HaloManager(const MeshBase& mesh, Real halo_pad, Opts opts=Opts());
  
  /**
   * Set the \p serializer used to read/write particles
   * from/to a buffer for communication between processors.
   */
  void set_serializer(const Serializer<Point*>& serializer);

  /**
   * @returns the processor IDs that are immediately neighboring
   * this processor, determined using the ghost elements of the mesh
   */
  const std::vector<int>& neighbor_processors() const;

  /**
   * @returns the processor IDs for processors whose mesh is at
   * most distance halo_pad from this processor's mesh.
   * May include extra processor IDs.
   */
  const std::vector<int>& box_halo_neighbor_processors() const;

  const Real get_halo_pad() const;

  void comm_particles(PointTree& tree,
      MeshTools::BoundingBox box=MeshTools::BoundingBox());

  void comm_particles(std::vector<Point*>& particles,
      MeshTools::BoundingBox box=MeshTools::BoundingBox());

  /**
   * For each point in the \p halo_centers vector, finds all other points
   * in the \p particles vector that are a distance of at most halo_pad
   * from the given point, and stores those values in \p result at the same
   * index.  The \p particle_inbox vector will be filled with points that
   * were received from nearby processors and were newly allocated.
   * HaloManager is not responsible for deleting these points.
   * In case the particle class used is a subclass of Point,
   * the \p serializer is used to read and write particles
   * to a buffer for communication between processors.
   */
  void find_particles_in_halos(
      const std::vector<Point*>& halo_centers,
      const std::vector<Point*>& particles,
      std::vector<std::vector<Point*> >& result) const;
  
  /**
   * Same as other find_particles_in_halos method, except
   * uses the same vector for particles and halo_centers.
   */
  void find_particles_in_halos(
      const std::vector<Point*>& particles,
      std::vector<std::vector<Point*> >& result) const;
  
  /**
   * Transfers particles between neighboring processors.
   * The \p particles vector is the local set of particles.
   * The \p destinations vector is the set of destination
   * processor IDs for each particle (same size as particles vector).
   * Each of these IDs must correspond either to a neighbor
   * processor or this processor.  The outgoing particles will
   * be removed from the \p particles vector, and incoming particles
   * will be added to the \p particles vector.
   * In case the particle class used is a subclass of Point,
   * the \p serializer is used to read and write particles
   * to a buffer for communication between processors.
   */
  void redistribute_particles(std::vector<Point*>& particles,
      const std::vector<int>& destinations);
      
  /**
   * Same as other redistribute_particles method, but uses
   * the processor IDs of mesh elements to determine the destinations
   * of the particles.  Each particle must be within some element
   * on the local mesh (could be a ghost element).
   */
  void redistribute_particles(std::vector<Point*>& particles);

  /**
   * Finds the processor IDs that are immediately neighboring
   * this processor, determined using the ghost elements of \p mesh.
   * These IDs are placed into the \p result vector.
   */
  static void find_neighbor_processors(const MeshBase& mesh,
      std::vector<int>& result);

private:

  void a2a_form_halo_neighbors(const MeshTools::BoundingBox& halo);

  /**
   * Receives points (from other processors) that are within
   * the given \p box_halo.  The received particles are placed
   * into the \p inbox vector.  The \p tree should contain
   * the same points as in the \p particles vector, and is used
   * for quick lookup of points in box halos to send to other processors.
   * In case the particle class used is a subclass of Point,
   * the \p particle_serializer is used to read and write particles
   * to a buffer for communication between processors.
   */
  //void comm_particles(MeshTools::BoundingBox box_halo, PointTree& tree,
  //    std::vector<Point*>& particle_inbox) const;

  /**
   * Extends the thickness of the /p box by halo_pad in each dimension.
   */
  void pad_box(MeshTools::BoundingBox& box) const;

  Opts opts;

  /**
   * Vector of processor IDs that are immediately neighboring this processor.
   */
  std::vector<int> neighbors;

  /**
   * Vector of processor IDs that might contain points within the halo
   * of this processor.
   */
  std::vector<int> box_halo_neighbors;

  /**
   * The radius of the halos around points.
   */
  Real halo_pad;
  
  /**
   * The serializer used to read/write particles
   * from/to a buffer for communication between processors.
   */
  const Serializer<Point*>* serializer;
  
  /**
   * The mesh used for looking up elements.
   */
  const MeshBase& mesh;

  /**
   * Communicator used to send particles between processors.
   */
  const Parallel::Communicator& comm;

  /**
   * Communication tag used for requesting particles from another processor.
   */
  const Parallel::MessageTag tag_request;

  /**
   * Communication tag used for sending particles to another processor.
   */
  const Parallel::MessageTag tag_response;

  /**
   * Communication tag used for redistributing particles.
   */
  const Parallel::MessageTag tag_redistribute;
};

} // end namespace libMesh

#endif // LIBMESH_HALO_MANAGER_H
