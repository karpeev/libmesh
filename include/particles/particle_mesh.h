#ifndef LIBMESH_PARTICLE_MESH
#define LIBMESH_PARTICLE_MESH

#include <ostream>

#include "libmesh/libmesh_common.h"
#include "libmesh/mesh_base.h"
#include "libmesh/equation_systems.h"
#include "libmesh/dof_object.h"
#include "libmesh/scatter.h"



namespace libMesh {


  class ParticleMesh : public ParallelPrinter, public EquationSystems, public Scatter::Packer, public Scatter::Unpacker
  {
  public:

    class ParticlePacker
    {
    public:
      virtual void pack(Scatter::OutBuffer& buf,const Point& point,const dof_id_type& id) = 0;
    };

    class ParticleUnPacker
    {
    public:
      virtual void unpack(Scatter::InBuffer& buf,Point& point,dof_id_type& id) = 0;
    };

    // Move impls to .C
    ParticleMesh(MeshBase& mesh, MeshData* mesh_data=NULL);
    ~ParticleMesh();

    void                                setup();
    bool                                is_setup(){return _setup;};
    void                                print_info(bool elem_info=false) const;

    unsigned int                        add_particle(const Point& point, dof_id_type id) {
      if (!_particles.get()) _particles = AutoPtr<Particles>(new Particles);
      _particles->push_back(Particle(point,id));
      return _particles->size()-1;
    };
    unsigned int                        num_particles() const {if (_particles.get()) {return _particles->size();} else {return 0;}}
    const std::pair<Point,dof_id_type>  get_particle(unsigned int i)  const {
      if (_particles.get()) {
	return (*_particles)[i];
      } else {
	libmesh_error_msg("No particles have been added.");
      }
    };

   // List of particle indices for global element id e: each returned i can be fed into get_particle(i)
    // to get the corresponding Point and id.
    const std::vector<unsigned int>&    elem_particles(dof_id_type e) const;

    void                                translate_particles(const std::vector<Point>& shifts,ParticlePacker& packer,ParticleUnPacker& unpacker);


  protected:
    typedef std::pair<Point,dof_id_type> Particle;
    typedef std::vector<Particle>        Particles;

    bool                                                   _setup;

    void                                                   __map_particles(const Particles& particles,std::map<dof_id_type,std::vector<unsigned int> >& elem_particle_ids);
    // Particles -- some of which might not be local, if we are in the middle of translation.
    AutoPtr<Particles>                                     _particles;

    // TODO: optimize away this temporary particle buffer.  It holds translated particles before they have
    // been scattered to their owning processors.  This could be done by splitting scatter into 'scatter_begin()'
    // and 'scatter_end()' so that at the end of 'scatter_begin()' the particles have been moved to communication
    // buffers and the particle array can be cleared out.
    // Another option is to introduce 'postpack' and/or 'preunpack' hooks.
    AutoPtr<Particles>                                     _tparticles;

    // Might as well use a map to attach particles to elements:
    // we only have sequential, not random, access to the local elements (through the iterator),
    // which likely involves an std::set with similar complexity to std::map.
    // TODO: investigate ways of switching to an array instead using a local renumbering of the ghosts
    // FIXME: should be 'indices', not 'ids'
    std::map<dof_id_type,std::vector<unsigned int> >      _elem_particle_ids;

    std::map<dof_id_type,std::vector<unsigned int> >      _telem_particle_ids;


    AutoPtr<ScatterDistributed>                           _ghost_element_scatter; // ghost elements to their real counterparts

    // packer/unpacker for individual particles
    ParticlePacker*                          _ppacker;
    ParticleUnPacker*                        _punpacker;
    void prepack(int /*ochannel*/,int /*source*/){};
    void pack(int ochannel,int /*source*/,Scatter::OutBuffer& obuffer);
    void unpack(int ichannel,int /*source*/,Scatter::InBuffer& ibuffer);

 };// class ParticleMesh
}// namespace libMesh

#endif //LIBMESH_PARTICLE_MESH
