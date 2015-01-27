#include <set>

#include "libmesh/parallel_object.h"
#include "libmesh/particle_mesh.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/elem_ghost_rendezvous.h"



using namespace libMesh;


ParticleMesh::ParticleMesh(MeshBase& mesh, MeshData* mesh_data)
  : ParallelPrinter(mesh.comm()),EquationSystems(mesh,mesh_data), _setup(false)
{
}

ParticleMesh::~ParticleMesh()
{
}



void ParticleMesh::setup()
{
  if (_setup) return;
  if (!_ghost_element_scatter.get()) {
    _ghost_element_scatter = ElemGhostRendezvous(_mesh).createElemGhostScatter();
    _ghost_element_scatter->setup();
  }
  if (_particles.get())
    __map_particles(*_particles,_elem_particle_ids);
  _setup = true;
}

// Translation of particles, possibly, across processor boundaries.
// Here we might, for example, move all of the local particles, figure out
// which ones are moving off the process and post the sends/receives.
// To update the particle-element relations call setup() after calling translate_particles().
// TODO: should we make shifts into a pair of iterators? This might save memory, for example,
//       when shifting the system uniformly.  However, (a) this use case is rather synthetic,
//       (b) the memory/time savings would probably be in the noise for any real use case.
void ParticleMesh::translate_particles(const std::vector<Point>& shifts,ParticlePacker& packer,ParticleUnPacker& unpacker)
{
  libmesh_assert_msg(_setup, "Cannot translate_particles: ParticleMesh not set up.");
  std::ostringstream err;
  err << "Invalid number of shift vectors " << shifts.size() << ", should be " << _particles->size();
  libmesh_assert_msg(shifts.size()==_particles->size(),err.str());
  AutoPtr<Particles> tmp = _tparticles;
  _tparticles = _particles;
  _particles = tmp;
  if (!_particles.get()) {
    _particles = AutoPtr<Particles>(new Particles);
  } else {
    _particles->clear();
  }
  for (unsigned int i = 0; i < _tparticles->size(); ++i) {
    (*_tparticles)[i].first += shifts[i];
  }
  _elem_particle_ids.clear();

  _telem_particle_ids.clear();  // FIXME: this might be redundant, since we clear at the end of translation; just in case for now.
  __map_particles(*_tparticles,_telem_particle_ids);
  _ppacker   = &packer;
  _punpacker = &unpacker;
  _ghost_element_scatter->scatter(*this,*this);
  _tparticles->clear();
  _telem_particle_ids.clear();
}

#undef  __FUNCT__
#define __FUNCT__ "ParticleMesh::__map_particles"
void ParticleMesh::__map_particles(const Particles& particles,std::map<dof_id_type,std::vector<unsigned int> >& elem_particle_ids)
{
#ifdef DEBUG
  if (debug(__FUNCT__)) {
    rank_print(std::cout) << ": " << __FUNCT__ << ": mapping " << particles.size() << " particles\n";
  }
#endif
  elem_particle_ids.clear();
  for (unsigned int i = 0; i < particles.size(); ++i) {
    const Point& q    = particles[i].first;
    const Elem*     elem = _mesh.point_locator()(q);
    if(elem == NULL) {
#ifdef DEBUG
    const dof_id_type& id = particles[i].second;
    if (debug(__FUNCT__)) {
      rank_print(std::cout) << ": " << __FUNCT__ << ": \tparticle <(" << q(0) << "," << q(1) << "," << q(2) << ")," << id << "> with local index " << i << " mapped to no element" << std::endl;
    }
#endif
      libMesh::err << "No element at point " << q << std::endl;
      //libmesh_error();
    } else {
#ifdef DEBUG
      const dof_id_type& id = particles[i].second;
      if (debug(__FUNCT__)) {
	rank_print(std::cout) << ": " << __FUNCT__ << ": \tparticle <(" << q(0) << "," << q(1) << "," << q(2) << ")," << id << "> with local index " << i << " mapped to element " << elem->id() << std::endl;
      }
#endif
      dof_id_type e = elem->id();
      if (elem_particle_ids.find(e) == elem_particle_ids.end()) {
	elem_particle_ids[e] = std::vector<unsigned int>();
      }
      elem_particle_ids[e].push_back(i);
    }
  }
}

#undef  __FUNCT__
#define __FUNCT__ "ParticleMesh::pack"
void ParticleMesh::pack(int src, int /*elemid*/,Scatter::OutBuffer& obuffer) {
#ifdef DEBUG
  if (!_tparticles.get()) libmesh_error_msg("NULL _tmp_particle array pointer.");
#endif
  // Pack all of the _tparticles belonging to element with id src into a corresponding buffer
  const std::vector<unsigned int>& src_particle_ids = _telem_particle_ids[src];
  unsigned int src_size = src_particle_ids.size();
#ifdef DEBUG
  if (debug(__FUNCT__)) {
    rank_print(std::cout,": ") << __FUNCT__ << ": packing " << src_size << " particles from source " << src << std::endl;
  }
#endif
  obuffer.write(src_size);
  for (unsigned int i = 0; i < src_size; ++i) {
    unsigned int idx;
    idx = src_particle_ids[i];
#ifdef DEBUG
    if (idx >= _tparticles->size()) {
      std::ostringstream err;
      err << "Cannot pack particle with index " << idx << ": _tmp_particle array has size " << _tparticles->size();
      libmesh_error_msg(err.str());
    }
#endif
    Point& point = (*_tparticles)[idx].first;
    dof_id_type id;
    id  = (*_tparticles)[idx].second;
#ifdef DEBUG
    if (debug(__FUNCT__)) {
      rank_print(std::cout,": ") << __FUNCT__ << ": \tpacking particle  <(" << point(0) << "," << point(1) << "," << point(2) << "),"<< id << "> with local index " << idx << std::endl;
    }
#endif
    _ppacker->pack(obuffer,point,id);
  }
}


#undef  __FUNCT__
#define __FUNCT__ "ParticleMesh::unpack"
void ParticleMesh::unpack(int src, int/*elemid*/,Scatter::InBuffer& ibuffer) {
  // Unpack all of the received particles and add them as local (should we check that they are?)
  unsigned int src_size;
  ibuffer.read(src_size);
#ifdef DEBUG
  if (debug(__FUNCT__)) {
    rank_print(std::cout,": ") << __FUNCT__ << ": unpacking " << src_size << " particles from source " << src << std::endl;
  }
#endif
  for (unsigned int i = 0; i < src_size; ++i) {
    dof_id_type p_id;
    Point       point;
    // FIXME: p_id is the 'particle id' output parameter; should return it through the return value. [Why?]
    _punpacker->unpack(ibuffer,point,p_id);
    unsigned idx;
    if (!_particles.get()) _particles = AutoPtr<Particles>(new Particles);
    idx = _particles->size();
    _particles->push_back(Particle(point,p_id));

#ifdef DEBUG
    if (debug(__FUNCT__)) {
      rank_print(std::cout,": ") << __FUNCT__ << ": \tunpacked particle <(" <<  point(0) << "," << point(1) << "," << point(2) << ")," << p_id << "> at local index " << idx << std::endl;
    }
#endif
    if (_elem_particle_ids.find(src) == _elem_particle_ids.end()) {
      _elem_particle_ids[src] = std::vector<unsigned int>();
    }
    _elem_particle_ids[src].push_back(idx);
  }
}

void ParticleMesh::print_info(bool elem_info) const {
#ifdef DEBUG
  int rank = comm().rank();
  int size = comm().size();
#endif
  MeshTools::BoundingBox bb = MeshTools::processor_bounding_box(_mesh,processor_id());
  std::ostringstream sout;
  if (!processor_id()) {
    sout << "ParticleMesh: >>>>>\n";
  }
  rank_print(sout,": ");
  Point min = bb.min(), max = bb.max();
  sout << "bounding box:    [(" << min(0) << ","<<min(1)<<","<<min(2) << ") - (" << max(0) << ","<<max(1)<<","<<max(2) << ")]\n";
  unsigned int particles_size;
  if (!_particles.get()) {
    sout << "no particles\n";
  } else {
    particles_size = _particles->size();
    sout << particles_size << " particles: [ ";
    for (unsigned int i = 0; i < _particles->size(); ++i) {
      const Point& p = (*_particles)[i].first;
      const dof_id_type id = (*_particles)[i].second;
      sout << "<(" << p(0) << "," << p(1) << "," << p(2) << "),"<< id << "> ";
    }
    sout << "]\n";
  }

  if (elem_info) {
    unsigned int ecount = 0;
    unsigned int esize = _elem_particle_ids.size();
    for (std::map<dof_id_type,std::vector<dof_id_type> >::const_iterator elem_particle_ids_it = _elem_particle_ids.begin(); elem_particle_ids_it != _elem_particle_ids.end(); ++elem_particle_ids_it,++ecount) {
      dof_id_type e = elem_particle_ids_it->first;
      const Elem* elem = _mesh.elem(e);
      rank_print(sout,": ") << "Elem " << e << ": ";
      elem->print_info(sout);
      const std::vector<dof_id_type>& pp = elem_particle_ids_it->second;
      unsigned int pp_size = pp.size();
      rank_print(sout,": ") << "Elem " << e << " has " << pp_size << " particles: [ ";
      for (unsigned int i = 0; i < pp_size; ++i) {
	unsigned int idx = pp[i];
	const Point point = (*_particles)[idx].first;
	const dof_id_type id = (*_particles)[idx].second;
	sout << idx<<":<(" << point(0) << "," << point(1) << "," << point(2)<<"),"<<id<<"> ";
      }
      sout << "]\n";
    }
  }
  gather_print(sout);
  if (_ghost_element_scatter.get()) {
    if (!processor_id()) {
      std::cout << "ghost_element_scatter:\n";
    }
    _ghost_element_scatter->print_info();
  }

  if (!processor_id()) {
    std::cout << "ParticleMesh: <<<<<\n";
  }
}
