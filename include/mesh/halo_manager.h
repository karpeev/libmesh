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

  /**
   * A list of options for constructing a HaloManager.
   *
   * \author  Matthew D. Michelotti
   */
  class Opts {
  public:
    /**
     * Constructor.
     */
    Opts();

    /**
     * If true, will use an MPI allgather operation to redundantly store
     * all particles on each processor when comm_particles is called.
     * If false, will build up a set of nearby processors at the start
     * (the box_halo_neighbor_processors) and only send particles between
     * these processors, and only those particles that overlap the halo
     * of the bounding box of the receiving processor.  Default value
     * is false.
     */
    bool use_all_gather;

    /**
     * If true, will perform local search of which particles are in which
     * halos using a PointTree.  If false, will perform this local search
     * using the inefficient naive approach of comparting each particle
     * to each other particle directly.  Default value is true.
     */
    bool use_point_tree;
  };

  /**
   * Constructor.  Uses \p mesh to determine connectivity between
   * processors.  \p halo_pad is the radius of the halos.
   * \p opts specifies options for constructing the HaloManager.
   */
  HaloManager(const MeshBase& mesh, Real halo_pad, Opts opts=Opts());
  
  /**
   * Set the \p serializer used to read/write particles
   * from/to a buffer for communication between processors.
   */
  void set_serializer(Serializer<Point*>& serializer);

  /**
   * @returns the processor IDs that are immediately neighboring
   * this processor, determined using the ghost elements of the mesh
   */
  const std::vector<int>& neighbor_processors() const;

  /**
   * @returns the processor IDs for processors whose mesh is at
   * most distance halo_pad from this processor's mesh.
   * May include extra processor IDs.  If the HaloManager was
   * constructed using the use_all_gather option, this will
   * just return an empty array.
   */
  const std::vector<int>& box_halo_neighbor_processors() const;

  /**
   * @returns the halo_pad, i.e. the radius of the halos
   */
  Real get_halo_pad() const;

  /**
   * Given a \p tree containing all local particles, will receive
   * particles from other processors that lie within the halo of the
   * bounding box of all the local particles (or all non-local particles
   * if the use_all_gather option was used).  These particles will
   * be added to the tree.  This method is not responsible for deleting
   * points allocated by the given serializer: the serializer should
   * track these.
   */
  void comm_particles(PointTree& tree) const;

  /**
   * Same as other comm_particles method, except parameter \p particles
   * is a vector instead of a tree.
   */
  void comm_particles(std::vector<Point*>& particles) const;

  /**
   * For each point in the \p halo_centers vector, finds all other points
   * in the \p particles vector that are a distance of at most halo_pad
   * from the given point, and stores those values in \p result at the same
   * index.  This method is not responsible for deleting points allocated
   * by the given serializer: the serializer should track these.
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
   * be removed from the \p particles vector and deleted using
   * the serializer's free method, and incoming particles
   * will be added to the \p particles vector.
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

  /**
   * Extends the thickness of the /p box by halo_pad in each dimension.
   */
  void pad_box(MeshTools::BoundingBox& box) const;
  
  /**
   * Implementation of comm_particles method assuming use_all_gather
   * option is true.
   */
  void comm_particles_w_all_gather(std::vector<Point*>& particles) const;
  
  /**
   * Implementation of comm_particles method assuming use_all_gather
   * option is false, where received particles are placed in the \p inbox
   * vector instead of the \p tree.
   */
  void comm_particles_w_sends(PointTree& tree, std::vector<Point*>& inbox)
      const;

  /**
   * The options specified at construction.
   */
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
  Serializer<Point*>* serializer;
  
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
