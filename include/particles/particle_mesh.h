#ifndef LIBMESH_PARTICLE_MESH
#define LIBMESH_PARTICLE_MESH

#include <ostream>

#include "libmesh/libmesh_common.h"
#include "libmesh/mesh_base.h"
#include "libmesh/equation_systems.h"
#include "libmesh/dof_object.h"
#include "libmesh/scatter.h"



namespace libMesh{


  class ParticleMesh : public EquationSystems, Scatter::Packer, Scatter::Unpacker {
  public:
    // Particles container must implement
    //  - translation by a vector of Points (const std::vector<Point>&)
    //  - location of the i-th particle:                        const Point&  operator()(unsigned int i)
    //  - packing of i-th particle to an OutBuffer:             void          write(Scatter::OutBuffer&,unsigned int i)
    //  - unpacking of i-th particle from an InBuffer:          void          read(Scatter::InBuffer&,unsigned int& i)
    //  -                                                       unsigned int  size()
    //  -                                                       void          clear()
    //  - clone:                                                Particles*    clone()
    //  - viewing to a stream:                                  void          view(std::ostream& os)
    class Particles {
    public:
      virtual void         translate(const std::vector<Point>& shift) = 0;
      virtual void         write(Scatter::OutBuffer& obuffer,unsigned int i) const = 0;
      virtual void         read(Scatter::InBuffer& ibuffer,unsigned int& i) = 0;
      virtual const Point& operator()(unsigned int i) const = 0;
      virtual unsigned int size() const = 0;
      virtual Particles*   clone() const = 0;
      virtual void         clear() = 0;
      virtual void         view(std::ostream& os) const = 0;
      virtual ~Particles(){};
    };

    ParticleMesh(MeshBase& mesh, MeshData* mesh_data=NULL);
    ~ParticleMesh();

    void                     set_local_particles(AutoPtr<Particles> particles);
    void                     setup();
    bool                     is_setup(){return _setup;};
    // FIXME: is there a clean way to return const AutoPtr<Particles> for symmetry with the set_xxx?
    const Particles&         get_local_particles()  const { return *_local_particles; };
    void                     print_info() const;


    // List of particles for global element id e
    const std::vector<unsigned int>& local_elem_particles(dof_id_type e) const;

    // Translation of particles, possibly, across processor boundaries.
    // Here we might, for example, move all of the local particles, figure out
    // which ones are moving off the process and post the sends/receives.
    // To update the particle-element relations call setup() after calling translate_local_particles().
    void translate_local_particles(const std::vector<Point>& shifts);

  protected:
#ifdef DEBUG
    std::vector<std::string> _vdebug;
    bool __debug(const char*s) {return std::find(_vdebug.begin(),_vdebug.end(),std::string(s)) != _vdebug.end();};
#endif
    std::ostream& __rankprint(std::ostream& os) {os << "["<<this->comm().rank()<<"|"<<this->comm().size()<<"]: "; return os;}
    bool _setup;

    // Local particles -- some of which might not be local, if we are in the middle of translation.
    AutoPtr<Particles>                                     _local_particles;

    // TODO: optimize away this temporary particle buffer.  It holds translated particles before they have
    // been scattered to their owning processors.  This could be done by splitting scatter into 'scatter_begin()'
    // and 'scatter_end()' so that at the end of 'scatter_begin()' the particles have been moved to communication
    // buffers and the particle array can be cleared out.
    // Another option is to introduce 'postpack' and/or 'preunpack' hooks.
    AutoPtr<Particles>                                     _tmp_particles;

    // Might as well use a map to attach particles to elements:
    // we only have sequential, not random, access to the local elements (through the iterator),
    // which likely involves an std::set with similar complexity to std::map.
    // TODO: investigate ways of switching to an array instead using a local renumbering of the ghosts
    std::map<dof_id_type,std::vector<unsigned int> >      _local_elem_particle_ids;

    AutoPtr<ScatterDistributed>                           _ghost_element_scatter; // ghost elements to their real counterparts

    // This routine will calculate all of the particle-element relations and the particle scatter
    // once the local particles have beeen set.
    void map_particles();

    // TODO: factor this out into a separate packer operating on Particles
    void prepack(int /*ochannel*/,int /*source*/){};
    void pack(int ochannel,int /*source*/,Scatter::OutBuffer& obuffer);
    void unpack(int ichannel,int /*source*/,Scatter::InBuffer& ibuffer);

    // TODO: factor this out into a separate Rendezvous operating on MeshBase
    void __create_scatter();

    void __add_local_elem_particle_id(dof_id_type e, unsigned int i);
 };// class ParticleMesh
}// namespace libMesh

#endif //LIBMESH_PARTICLE_MESH
