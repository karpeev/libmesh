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
  HaloManager hm(mesh, haloPad);
  std::vector<Particle*> particles;
  typedef MeshBase::element_iterator ElemIter_t;
  for(ElemIter_t it = mesh.local_elements_begin();
      it != mesh.local_elements_end(); it++)
  {
    particles.push_back(new MyParticle(it->centroid(), it->unique_id()));
  }
  std::vector<std::vector<Particle*> > result;
  hm.find_particles_in_halos(particles,
      (std::vector<std::vector<Particle*> >)result);
}


//NOTE: finding neighbor processor id's and halo processor id's will
//      be handled in the HaloManager class when first needed

