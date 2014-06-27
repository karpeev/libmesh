//NOTE: Particle will extend Point and have a virtual serialize function...
//      Serialize function handles input AND output.
//      Serializer will have a flag saying whether it's in input or output mode...


class MyParticle : public Particle {
  public:
    MyParticle(Point& point, double val) : Particle(point) {
      data.push_back(val);
    }

    void serialize(Serializer& s) {
      Particle::serialize(s);
      s << data;
    }

  private:
    std::vector<double> data;
};

int main(int argc, char** argv) {
  LibMeshInit init(argc, argv);
  ParallelMesh mesh(init.comm());
  MeshTools::Generation::build_cube(mesh, 360, 1, 1, 0, 360, 0, 1, 0, 1);
  mesh.print_info();
  Real haloPad = 85;
  std::vector<int> neighbors;
  Halo::find_neighbor_proc_ids(mesh, neighbors);
  std::vector<int> haloPids;
  Halo::parallel_find_bounding_box_halo_proc_ids(mesh, haloPad, haloPids);
  std::vector<Particle*> particles;
  typedef MeshBase::element_iterator ElemIter_t;
  for(ElemIter_t it = mesh.local_elements_begin();
      it != mesh.local_elements_end(); it++)
  {
    particles.push_back(new MyParticle(it->centroid(), it->unique_id()));
  }
  std::vector<std::vector<Particle*> > result;
  Halo::parallel_find_particles_in_halos(particles, haloPids, haloPad,
      (std::vector<std::vector<Particle*> >)result);
}
