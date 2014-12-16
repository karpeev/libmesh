#include <set>

#include "libmesh/parallel_object.h"
#include "libmesh/particle_mesh.h"
#include "libmesh/mesh_tools.h"



using namespace libMesh;


ParticleMesh::ParticleMesh(MeshBase& mesh, MeshData* mesh_data)
  : EquationSystems(mesh,mesh_data), _setup(false)
{
}

ParticleMesh::~ParticleMesh()
{
}

void ParticleMesh::setup()
{
  if (_setup) return;
  if (!_ghost_element_scatter.get()) __create_scatter();
  _setup = true;
}

void ParticleMesh::translate_local_particles(const std::vector<Point>& shifts)
{
  if (!_local_particles.get()) return;
  if (!_tmp_particles.get()) {
    _tmp_particles = AutoPtr<Particles>(_local_particles->clone());
  } else {
    // TODO: can we optimize away this copy?  Should we?
    *_tmp_particles = *_local_particles;
  }
  _tmp_particles->translate(shifts);
  _ghost_element_scatter->scatter(*this,*this);
  _tmp_particles->clear();
}


void ParticleMesh::__create_scatter()
{
  // TODO: make this into a MeshGhostElemRendezvous.
  AutoPtr<ScatterDistributed> scatter = AutoPtr<ScatterDistributed>(new ScatterDistributedMPI(_mesh.comm()));
  // FIXME: which iterators actually include ghost elements?
  MeshBase::const_element_iterator it  = _mesh.active_local_elements_begin();
  MeshBase::const_element_iterator end = _mesh.active_local_elements_end();
  std::map<dof_id_type,std::set<dof_id_type> > ghosts;
  for (; it != end; ++it) {
    if ((*it)->processor_id() == _mesh.processor_id()) {
      // local element
      scatter->add_ochannel((*it)->id());
      scatter->add_ochannel_source((*it)->id(),(*it)->id());
      scatter->rank_add_ochannel(_mesh.processor_id(),(*it)->id());

    }
    int rank = (*it)->processor_id();
    int elem = (*it)->id();
    ghosts[rank].insert(elem);
  }
  for (std::map<dof_id_type,std::set<dof_id_type> >::const_iterator git = ghosts.begin(); git != ghosts.end(); ++git) {
    dof_id_type              rank  = git->first;
    std::set<dof_id_type>    r_ghost_elems = git->second;
    for (std::set<dof_id_type>::const_iterator r_ghost_elem_it = r_ghost_elems.begin(); r_ghost_elem_it != r_ghost_elems.end(); ++r_ghost_elem_it) {
      // Each ghost element is its own outchannel to the owning proc;
      // On the ghosting proc it is also a source that sends to the namesake channel.
      dof_id_type r_ghost_elem = *r_ghost_elem_it;
      scatter->add_ochannel(r_ghost_elem);
      scatter->add_ochannel_source(r_ghost_elem,r_ghost_elem);
      scatter->rank_add_ochannel(rank,r_ghost_elem);
    }
  }
  // Now outboxes are full, determine the inboxes and the rendezvous by communication.
  // Exchange channel data with the other processors.
  //
  std::vector<int> incoming;
  for (processor_id_type p=1; p != this->n_processors(); ++p) {
      // Trade my requests with processor procup and procdown
      processor_id_type procup = (this->processor_id() + p) % this->n_processors();
      processor_id_type procdown = (this->n_processors() + this->processor_id() - p) % this->n_processors();
      incoming.clear();
      // TODO: is there a way to structure the traversal of the comm to avoid the following lookup?
      // How much would we save by doing so?
      // TODO: zero-copy rank_get_ochannels.  In general, this should be in a rendezvous object, which
      // ought to be a friend of ScatterDistributedMPI.  Alternatively, the Rendezvous object allocates
      // this array and later passes the ownership to the Scatter via rank_set_ochannels(AutoPtr<...>)?
      std::vector<int> outgoing = scatter->rank_get_ochannels(procup);
      // TODO: reuse a tag to save a bit of time?  There generally isn't any danger of encountering loose
      // delayed messages since we expect to communicate with EVERY rank EVERY time a scatter is constructed.
      // Unless we construct many such scatters in quick succession, then messages from nearby constructions
      // might get mixed up.
      this->comm().send_receive(procup,outgoing,procdown,incoming);
      if (incoming.size()) {
	scatter->rank_add_ichannels(procdown,incoming);
	for(std::vector<int>::const_iterator r_ichannels_it = incoming.begin(); r_ichannels_it != incoming.end(); ++r_ichannels_it) {
	  scatter->add_ichannel(*r_ichannels_it);
	}
      }
  }
  // Finally, set up ichannel sources.  Each ichannel receives a single source that has the same id as the inchannel.
  // Thus, there is no need to send the sources in this case, but in general a source exchange has to occur as well.
  // In that case we might as well keep the per-rank ichannel sources (see item D in ScatterDistributedMPI::scatter()).
  for (Scatter::const_channel_sources_iterator ichannel_sources_it = scatter->ichannel_sources_begin(); ichannel_sources_it != scatter->ichannel_sources_end(); ++ichannel_sources_it) {
    int ichannel = ichannel_sources_it->first;
    scatter->add_ichannel_source(ichannel,ichannel);
  }
  _ghost_element_scatter = scatter;
}

void ParticleMesh::__add_local_elem_particle_id(dof_id_type e, unsigned int i)
{
  if (_local_elem_particle_ids.find(e) == _local_elem_particle_ids.end()) {
    _local_elem_particle_ids[e] = std::vector<unsigned int>();
  }
  _local_elem_particle_ids[e].push_back(i);
}

void ParticleMesh::pack(int src, int /*elemid*/,Scatter::OutBuffer& obuffer) {
  // Pack all of the _tmp_particles belonging to element with id src into a corresponding buffer
  const std::vector<unsigned int>& src_particle_ids = _local_elem_particle_ids[src];
  unsigned int src_size = src_particle_ids.size();
  obuffer.write(src_size);
  for (unsigned int i = 0; i < src_size; ++i) {
    unsigned int id = src_particle_ids[i];
    _tmp_particles->write(obuffer,id);
  }
}

void ParticleMesh::unpack(int src, int/*elemid*/,Scatter::InBuffer& ibuffer) {
  // Unpack all of the received particles and add them as local (should we check that they are?)
  unsigned int src_size;
  ibuffer.read(src_size);
  for (unsigned int i = 0; i < src_size; ++i) {
    unsigned int p_id;
    _local_particles->read(ibuffer,p_id);
    __add_local_elem_particle_id(src,p_id);
  }

}

void ParticleMesh::set_local_particles(AutoPtr<Particles> particles)
{
  for (unsigned int i = 0; i < particles->size(); ++i) {
    const Point& q    = (*particles)(i);
    const Elem*     elem = _mesh.point_locator()(q);
    int p = elem->processor_id();
    if(elem == NULL) {
      libMesh::err << "No element at point " << q << std::endl;
      libmesh_error();
    }
    if (p != processor_id()) {
      libMesh::err << "Found nonlocal particle " << q << std::endl;
      libmesh_error();
    }
    dof_id_type e = elem->id();
    __add_local_elem_particle_id(e,i);
  }
  _local_particles = particles;
}



void ParticleMesh::print_info() const
{
  MeshTools::BoundingBox bb = MeshTools::processor_bounding_box(_mesh,processor_id());
  std::ostringstream sout;
  if (!processor_id()) {
    std::cout << "ParticleMesh: >>>>>\n";
  }
  sout << "[" << processor_id() << "|" << comm().size() << "]\n";
  Point min = bb.min(), max = bb.max();
  sout << "bounding box:    [(" << min(0) << ","<<min(1)<<","<<min(2) << ") - (" << max(0) << ","<<max(1)<<","<<max(2) << ")]\n";
  sout << "local particles: ";
  const Particles& particles = get_local_particles();
  particles.view(sout);

  std::string text_str = sout.str();
  std::vector<char> text(text_str.begin(), text_str.end());
  text.push_back('\0');
  comm().gather(0, text);
  int ci = 0;
  while(ci < (int)text.size()) {
    std::cout << &text[ci] << std::endl;
    while(text[ci++] != '\0');
  }
  if (!processor_id()) {
    std::cout << "ParticleMesh: <<<<<\n";
  }
}
