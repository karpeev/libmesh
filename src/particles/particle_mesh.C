#include <set>

#include "libmesh/parallel_object.h"
#include "libmesh/particle_mesh.h"
#include "libmesh/mesh_tools.h"



using namespace libMesh;


ParticleMesh::ParticleMesh(MeshBase& mesh, MeshData* mesh_data)
  : EquationSystems(mesh,mesh_data), _setup(false)
{
#ifdef DEBUG
  command_line_vector("debug",_vdebug);
#endif
}

ParticleMesh::~ParticleMesh()
{
}

void ParticleMesh::__gatherprint(const std::ostringstream& sout, std::ostream& os) const {
  libmesh_parallel_only(comm());
  std::string text_str = sout.str();
  std::vector<char> text(text_str.begin(), text_str.end());
  text.push_back('\0');
  comm().gather(0, text);
  if (!processor_id()) {
    int ci = 0;
    while(ci < (int)text.size()) {
      os << &text[ci] << std::endl;
      while(text[ci++] != '\0');
    }
  }
}


void ParticleMesh::setup()
{
  if (_setup) return;
  if (!_ghost_element_scatter.get()) __create_scatter();
  _ghost_element_scatter->setup();
  _setup = true;
}

void ParticleMesh::translate_particles(const std::vector<Point>& shifts)
{
  libmesh_assert_msg(_setup, "Cannot translate_local_particles: ParticleMesh not set up.");
  AutoPtr<Particles> tmp = _tmp_particles;
  _tmp_particles = _particles;
  _particles = tmp;
  if (!_particles.get()) {
    _particles = AutoPtr<Particles>(_tmp_particles->clone());
  } else {
    _particles->clear();
  }
  _elem_particle_ids.clear();
  _tmp_particles->translate(shifts);
  __map_particles(*_tmp_particles);
  _ghost_element_scatter->scatter(*this,*this);
  _tmp_particles->clear();
}


#undef  __FUNCT__
#define __FUNCT__ "ParticleMesh::__create_scatter"
void ParticleMesh::__create_scatter()
{
  // TODO: make this into a MeshGhostElemRendezvous.
  AutoPtr<ScatterDistributed> scatter = AutoPtr<ScatterDistributed>(new ScatterDistributedMPI(_mesh.comm()));
  // FIXME: which iterators actually include ghost elements? NONE!
  MeshBase::const_element_iterator it  = _mesh.active_local_elements_begin();
  MeshBase::const_element_iterator end = _mesh.active_local_elements_end();
  // TODO: Use __gatherprint() for debugging info.
  std::map<dof_id_type,std::set<dof_id_type> > ghosts;
  for (; it != end; ++it) {
    Elem *elem = *it;
    int rank = (*it)->processor_id();
    int e = elem->id();
    // local element
#ifdef DEBUG
    if (__debug(__FUNCT__)) {
      __rankprint(std::cout);
      std::cout << "DEBUG: " << __FUNCT__ << ": Local element: " << e << "; adding source " << e << " to channel " << e << "; ";
      std::cout << "channel  " << e << " to rank " << rank << std::endl;
    }
#endif
    scatter->add_ochannel(e);
    scatter->add_ochannel_source(e,e);
    scatter->rank_add_ochannel(rank,e);
    // Look for neighboring ghosts
    for (unsigned int j = 0; j < elem->n_neighbors(); ++j) {
      Elem *neigh = elem->neighbor(j);
      if (neigh && neigh->processor_id() != this->processor_id()) {
	dof_id_type g = neigh->id();
	int grank = neigh->processor_id();
#ifdef DEBUG
    if (__debug(__FUNCT__)) {
      __rankprint(std::cout);
      std::cout << "DEBUG: " << __FUNCT__ << ": Element " << e << ": neighbor " << j << " is a ghost element " << g << " on rank " << grank << std::endl;
    }
#endif
	ghosts[grank].insert(g);
      }
    }
  }
  for (std::map<dof_id_type,std::set<dof_id_type> >::const_iterator git = ghosts.begin(); git != ghosts.end(); ++git) {
    dof_id_type              rank  = git->first;
    std::set<dof_id_type>    r_ghost_elems = git->second;
    for (std::set<dof_id_type>::const_iterator r_ghost_elem_it = r_ghost_elems.begin(); r_ghost_elem_it != r_ghost_elems.end(); ++r_ghost_elem_it) {
      // Each ghost element is its own outchannel to the owning proc;
      // On the ghosting proc it is also a source that sends to the namesake channel.
      dof_id_type g = *r_ghost_elem_it;
#ifdef DEBUG
      if (__debug(__FUNCT__)) {
	__rankprint(std::cout);
	std::cout <<  "DEBUG: " << __FUNCT__ << ": Ghost element: " << g << "; adding source " << g << " to  channel  " << g << " to rank " << rank << std::endl;
      }
#endif
      scatter->add_ochannel(g);
      scatter->add_ochannel_source(g,g);
      scatter->rank_add_ochannel(rank,g);
    }
  }
  // Now outboxes are full, determine the inboxes and the rendezvous by communication.
  // Copy the local channels and exchange channel data with the other processors.
  //
  std::vector<int> outgoing, incoming;
  outgoing = scatter->rank_get_ochannels(_mesh.processor_id());
#ifdef DEBUG
  if (__debug(__FUNCT__)) {
    std::ostringstream sout;
    __rankprint(sout);
    sout << "DEBUG: " << __FUNCT__ << ": ";
    sout << "Copying local outgoing channels to incoming: [ ";
    for (unsigned int i = 0; i < outgoing.size(); ++i) {
      sout << outgoing[i] << " ";
    }
    sout << "]\n";
    std::cout << sout.str();
  }
#endif
  scatter->add_ichannels(outgoing);
  scatter->rank_add_ichannels(_mesh.processor_id(),outgoing);
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
      outgoing = scatter->rank_get_ochannels(procup);
#ifdef DEBUG
      if (__debug(__FUNCT__)) {
	std::ostringstream sout;
	__rankprint(sout);
	sout << "DEBUG: " << __FUNCT__ << ": ";
	sout << "Channels going to rank " << procup << ": [ ";
	for (unsigned int i = 0; i < outgoing.size(); ++i) {
	  sout << outgoing[i] << " ";
	}
	sout << "]\n";
	std::cout << sout.str();
      }
#endif
      // TODO: reuse a tag to save a bit of time?  There generally isn't any danger of encountering loose
      // delayed messages since we expect to communicate with EVERY rank EVERY time a scatter is constructed.
      // Unless we construct many such scatters in quick succession, then messages from nearby constructions
      // might get mixed up.
      this->comm().send_receive(procup,outgoing,procdown,incoming);
#ifdef DEBUG
      if (__debug(__FUNCT__)) {
	std::ostringstream sout;
	__rankprint(sout);
	sout << "DEBUG: " << __FUNCT__ << ": ";
	sout << "Channels received from rank " << procdown << ": [ ";
	for (unsigned int i = 0; i < incoming.size(); ++i) {
	  sout << incoming[i] << " ";
	}
	sout << "]\n";
	std::cout << sout.str();
      }
#endif
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
#ifdef DEBUG
      if (__debug(__FUNCT__)) {
	__rankprint(std::cout);
	std::cout << "DEBUG: " << __FUNCT__ << ": ";
	std::cout << "Adding source " << ichannel << " to channel " << ichannel << std::endl;
      }
#endif
    scatter->add_ichannel_source(ichannel,ichannel);
  }
  _ghost_element_scatter = scatter;
}

void ParticleMesh::__add_elem_particle_id(dof_id_type e, unsigned int i)
{
  if (_elem_particle_ids.find(e) == _elem_particle_ids.end()) {
    _elem_particle_ids[e] = std::vector<unsigned int>();
  }
  _elem_particle_ids[e].push_back(i);
}

void ParticleMesh::pack(int src, int /*elemid*/,Scatter::OutBuffer& obuffer) {
  // Pack all of the _tmp_particles belonging to element with id src into a corresponding buffer
  const std::vector<unsigned int>& src_particle_ids = _elem_particle_ids[src];
  unsigned int src_size = src_particle_ids.size();
  obuffer.write(src_size);
  for (unsigned int i = 0; i < src_size; ++i) {
    unsigned int id;
    id = src_particle_ids[i];
    _tmp_particles->write(obuffer,id);
  }
}

#undef  __FUNCT__
#define __FUNCT__ "ParticleMesh::unpack"
void ParticleMesh::unpack(int src, int/*elemid*/,Scatter::InBuffer& ibuffer) {
#ifdef DEBUG
  std::vector<std::string> vdebug;
  command_line_vector("debug",vdebug);
  bool debug = std::find(vdebug.begin(),vdebug.end(),std::string(__FUNCT__)) != vdebug.end();
#endif
  // Unpack all of the received particles and add them as local (should we check that they are?)
  unsigned int src_size;
  ibuffer.read(src_size);
#ifdef DEBUG
  if (debug) {
    std::cout << "[" << this->comm().rank()<<"|"<< this->comm().size()<<"]: " << __FUNCT__ << ": unpacking " << src_size << " particles from source " << src << std::endl;
  }
#endif
  for (unsigned int i = 0; i < src_size; ++i) {
    unsigned int p_id;
    // FIXME: p_id is the 'particle id' output parameter; should return it through the return value
    _particles->read(ibuffer,p_id);
    __add_elem_particle_id(src,p_id);
  }

}

#undef  __FUNCT__
#define __FUNCT__ "ParticleMesh::__map_particles"
void ParticleMesh::__map_particles(const Particles& particles)
{
#ifdef DEBUG
  std::vector<std::string> vdebug;
  command_line_vector("debug",vdebug);
  bool debug = std::find(vdebug.begin(),vdebug.end(),std::string(__FUNCT__)) != vdebug.end();
  if (debug) {
    std::cout << "[" << this->comm().rank()<<"|"<< this->comm().size()<<"]: " << __FUNCT__ << ": mapping " << particles.size() << " particles\n";
  }
#endif

  for (unsigned int i = 0; i < particles.size(); ++i) {
    const Point& q    = (particles)(i);
    const Elem*     elem = _mesh.point_locator()(q);
    if(elem == NULL) {
      libMesh::err << "No element at point " << q << std::endl;
      //libmesh_error();
    } else {
      dof_id_type e = elem->id();
      __add_elem_particle_id(e,i);
    }
  }
}

#undef  __FUNCT__
#define __FUNCT__ "ParticleMesh::set_particles"
void ParticleMesh::set_particles(AutoPtr<Particles> particles)
{
#ifdef DEBUG
  std::vector<std::string> vdebug;
  command_line_vector("debug",vdebug);
  bool debug = std::find(vdebug.begin(),vdebug.end(),std::string(__FUNCT__)) != vdebug.end();
  if (debug) {
    std::cout << "[" << this->comm().rank()<<"|"<< this->comm().size()<<"]: " << __FUNCT__ << ": adding " << particles->size() << " particles\n";
  }
#endif
  __map_particles(*particles);
  _particles = particles;
}



void ParticleMesh::print_info() const {
  MeshTools::BoundingBox bb = MeshTools::processor_bounding_box(_mesh,processor_id());
  std::ostringstream sout;
  if (!processor_id()) {
    sout << "ParticleMesh: >>>>>\n";
  }
  sout << "[" << processor_id() << "|" << comm().size() << "]\n";
  Point min = bb.min(), max = bb.max();
  sout << "bounding box:    [(" << min(0) << ","<<min(1)<<","<<min(2) << ") - (" << max(0) << ","<<max(1)<<","<<max(2) << ")]\n";
  sout << "particles: ";
  const Particles& particles = get_particles();
  particles.view(sout);

//   for (std::map<dof_id_type,std::vector<dof_id_type> >::const_iterator elem_particle_ids_it = _elem_particle_ids.begin(); elem_particle_ids_it != _elem_particle_ids.end(); ++elem_particle_ids_it) {
//     dof_id_type e = elem_particle_ids_it->first;
//     const Elem* elem = _mesh.elem(e);
//     sout << "Elem " << e << ": ";
//     elem->print_info(sout);
//     sout << "particles: [ ";
//     const std::vector<dof_id_type>& pp = elem_particle_ids_it->second;
//     for (unsigned int i = 0; i < pp.size(); ++i) {
//       dof_id_type pid = pp[i];
//       const Point& point = (*_particles)(pid);
//       sout << "(" << point(0) << "," << point(1) << "," << point(2)<<") ";
//     }
//     sout << "]\n";
//   }

  __gatherprint(sout);
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
