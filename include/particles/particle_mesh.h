#ifndef LIBMESH_PARTICLE_MESH
#define LIBMESH_PARTICLE_MESH

#include "libmesh/libmesh_common.h"

namespace libMesh{
  template<Particle>
  class ParticleMesh : public ParallelObject {
  ParticleMesh(const MeshBase& mesh) : ParallelObject(mesh.comm()), _mesh(mesh){}
    // TODO: check compatibility of communicators

    const MeshBase& mesh()  const { return _mesh; }


    void set_local_particles(const std::vector<Particle>& particles);
    void set_halo(Real halo);
    Real get_halo() const;

    void  set_local_anchor_elem_ids(const std::vector<dof_id_type>& anchor_elems);
    const std::vector<dof_id_type>& local_anchor_elem_ids() const {return _anchors; }

    unsigned int    num_local_particles()           const { return _num_local_particles; }
    unsigned int    num_ghost_particles()           const { return _num_ghost_particles; }
    const Particle& local_particle(unsigned int i)  const { return _particles[i]; }
    const Particle& ghost_particle(unsigned int i)  const { return _particles[_num_local_particles+i]; }
    const Particle& particle(unsigned int i)        const { return _particles[i]; }

    const std::vector<unsigned int> local_particle_halo(const unsigned int i) const;
    const std::vector<unsigned int> local_anchor_elem_halo(const unsigned int i) const;


    const std::vector<dof_id_type>&  local_elems_with_particles() const;
    const std::vector<dof_id_type>&  local_particle_elem_ids(unsigned int i) const;
    const std::vector<unsigned int>& local_elem_particles(unsigned int) const;

    // Start translation of particles, possibly, across processor boundaries.
    // Here we might, for example, move all of the local particles, figure out
    // which ones are moving off the process and post the sends/receives.
    void translation_begin(const std::vector<Point>& translation_vectors);
    // Finish translating the particles.
    void translation_end(const std::vector<Point>& translation_vectors);

    // What's the advantage of the translation_begin()/translation_end() splitting?
    // After translation_begin(), for those particles staying within the processor
    // we might repartition them into halos and start computing their local field
    // contributions.  This would require updating the halos twice:
    // after translation_begin() and after translation_end(),
    // as well as splitting halos into local halos and ghost halos.

    /*
    AutoPtr<NumericVector<Number> >       create_global_particle_vector() const;
    AutoPtr<NumericVector<Number> >       create_ghosted_particle_vector() const;
    AutoPtr<SparseMatrix<Number> >        create_particle_particle_matrix() const;
    */

    void setup();
  protected:
    const MeshBase& _mesh;

  };// class ParticleMesh

}

#endif //LIBMESH_PARTICLE_MESH
