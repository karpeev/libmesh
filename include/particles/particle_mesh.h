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

    // We could use local Nodes as points to evaluate the influence of particles on the Nodes.
    // The correspondence between the Nodes and the points is through the indices.
    void set_local_points(const std::vector<Point>& points);
    void set_halo(Real halo);
    Real get_halo() const;



    unsigned int    num_local_particles()           const { return _num_local_particles; }
    const Particle& local_particle(unsigned int i)  const { return _particles[i]; }
    /* unsigned int    num_ghost_particles()           const { return _num_ghost_particles; } */
    /* const Particle& ghost_particle(unsigned int i)  const { return _particles[_num_local_particles+i]; } */
    const Particle& particle(unsigned int i)        const { return _particles[i]; }

    // This routine will calculate all of the particle-element relations once the local particles have beeen set.
    void setup();

    /*
    unsigned int    num_local_points()              const { return _local_points.size(); }
    const Point&    local_point(unsigned int i)     const { return _local_points[i]; }
    */

    // A list of particle indices within a disk of radius 'halo' of the i-th local point
    /* const std::vector<unsigned int>& local_point_halo(const unsigned int i) const; */

    // A list of particle indices within a disk of raidus 'halo' of the i-th local particle
    /*const std::vector<unsigned int>& local_particle_halo(const unsigned int i) const;*/

    // TODO: allow for turning on and off the halos of particles and points to save work
    // when only particles or points need a halo.

    // Elements that contain the i-th local particle (typically just 1 element)
    const std::vector<dof_id_type>&  local_particle_elem_ids(unsigned int i) const;

    // Definition: The local id of an element is defined as the position of that element
    // when traversing the local elements using mesh's active_local_elements iterators.
    // List of local element ids for the elements that contain particles.
    const std::vector<dof_id_type>&  local_elems_with_particles() const;
    // List of particles for local element with _local_ id e
    const std::vector<unsigned int>& local_elem_particles(dof_id_type e) const;

    // Translation of particles, possibly, across processor boundaries.
    // Here we might, for example, move all of the local particles, figure out
    // which ones are moving off the process and post the sends/receives.
    // To update the particle-element relations call setup() after calling translate_local_particles().
    void translate_local_particles(const std::vector<Point>& translation_vectors);

  protected:
    const MeshBase& _mesh;

  };// class ParticleMesh

}

#endif //LIBMESH_PARTICLE_MESH
