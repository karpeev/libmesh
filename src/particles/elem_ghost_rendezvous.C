#include <set>

#include "libmesh/parallel_object.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/elem_ghost_rendezvous.h"



using namespace libMesh;

ElemGhostRendezvous::ElemGhostRendezvous(MeshBase& mesh)  : ParallelObject(mesh.comm()), ParallelPrinter(mesh.comm()), _mesh(mesh)
{
}

ElemGhostRendezvous::~ElemGhostRendezvous(){}


#undef  __FUNCT__
#define __FUNCT__ "ElemGhostRendezvous::createElemGhostScatter"
AutoPtr<ScatterDistributed> ElemGhostRendezvous::createElemGhostScatter()
{
  AutoPtr<ScatterDistributed> scatter = AutoPtr<ScatterDistributed>(new ScatterDistributedMPI(_mesh.comm()));
  // FIXME: which iterators actually include ghost elements? NONE!
  MeshBase::const_element_iterator it  = _mesh.active_local_elements_begin();
  MeshBase::const_element_iterator end = _mesh.active_local_elements_end();
  std::map<dof_id_type,std::set<dof_id_type> > ghosts;
  for (; it != end; ++it) {
    Elem *elem = *it;
    int rank = (*it)->processor_id();
    int e = elem->id();
    // local element
#ifdef DEBUG
    if (debug(__FUNCT__)) {
      rank_print(std::cout,": ");
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
	if (debug(__FUNCT__)) {
	  rank_print(std::cout,": ");
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
      if (debug(__FUNCT__)) {
	rank_print(std::cout,": ");
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
  if (debug(__FUNCT__)) {
    std::ostringstream sout;
    rank_print(sout,": ");
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
      if (debug(__FUNCT__)) {
	std::ostringstream sout;
	rank_print(sout,": ");
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
      if (debug(__FUNCT__)) {
	std::ostringstream sout;
	rank_print(sout,": ");
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
      if (debug(__FUNCT__)) {
	rank_print(std::cout,": ");
	std::cout << "DEBUG: " << __FUNCT__ << ": ";
	std::cout << "Adding source " << ichannel << " to channel " << ichannel << std::endl;
      }
#endif
    scatter->add_ichannel_source(ichannel,ichannel);
  }
  scatter->setup();
  return scatter;
}
