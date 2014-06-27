

namespace libMesh {

class HaloManager {

public:
  HaloManager(const MeshBase& mesh, double haloPad);
  HaloManager(const std::vector<int>& neighbors, double haloPad);
  const std::vector<int>& neighbor_processors() const;
  const std::vector<int>& box_halo_neighbor_processors();
  void find_particles_in_halos(std::vector<Particle*> particles,
      std::vector<std::vector<Particle*> > result);
};

} // end namespace libMesh

