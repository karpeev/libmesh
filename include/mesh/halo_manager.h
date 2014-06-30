
#ifndef LIBMESH_HALO_MANAGER_H
#define LIBMESH_HALO_MANAGER_H

namespace libMesh {

class HaloManager {

public:
  HaloManager(const MeshBase& mesh, double halo_pad);
  const std::vector<int>& neighbor_processors() const;
  const std::vector<int>& box_halo_neighbor_processors() const;
  void find_particles_in_halos(const std::vector<Particle*> particles,
      std::vector<Particle*> particle_inbox,
      Particle* (*constructor)(),
      std::vector<std::vector<Particle*> > result) const;
      
  static void find_neighbor_processors(const MeshBase& mesh,
      std::vector<int>& result)

private:
  BoundingBox find_bounding_box(const std::vector<Particle*> particles) const;

  std::vector<int> neighbors;
  std::vector<int> box_halo_neighbors;
  double halo_pad;

};

} // end namespace libMesh

#endif // LIBMESH_HALO_MANAGER_H
