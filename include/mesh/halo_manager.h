#ifndef LIBMESH_HALO_MANAGER_H
#define LIBMESH_HALO_MANAGER_H

#include "libmesh/libmesh_common.h"
#include "libmesh/serializer.h"
#include "libmesh/mesh_tools.h"
#include <vector>

namespace libMesh {

class PointTree;
class MeshBase;
class Point;

namespace Parallel {
  class Communicator;
  class MessageTag;
}

class HaloManager {

public:
  HaloManager(const MeshBase& mesh, Real halo_pad);
  const std::vector<int>& neighbor_processors() const;
  const std::vector<int>& box_halo_neighbor_processors() const;
  void find_particles_in_halos(std::vector<Point*>& particles,
      const Serializer<Point*>& particle_serializer,
      std::vector<Point*>& particle_inbox,
      std::vector<std::vector<Point*> >& result) const;
      
  static void find_neighbor_processors(const MeshBase& mesh,
      std::vector<int>& result);

private:
  void comm_particles(
      std::vector<Point*>& particles,
      const PointTree& tree,
      const Serializer<Point*>& particle_serializer,
      std::vector<Point*>& particle_inbox) const;

  void pad_box(MeshTools::BoundingBox& box) const;

  std::vector<int> neighbors;
  std::vector<int> box_halo_neighbors;
  Real halo_pad;
  const Parallel::Communicator& comm;
  const Parallel::MessageTag tagRequest;
  const Parallel::MessageTag tagResponse;
};

} // end namespace libMesh

#endif // LIBMESH_HALO_MANAGER_H
