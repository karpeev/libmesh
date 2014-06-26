
#ifndef LIBMESH_HALO_H
#define LIBMESH_HALO_H

#include "libmesh/libmesh_common.h"
#include "libmesh/parallel.h"
#include "libmesh/point_locator_base.h"
#include "libmesh/elem.h"

#include <vector>
#include <map>

namespace libMesh {

class MeshBase;

namespace Halo {

void find_neighbor_proc_ids(const MeshBase& mesh, std::vector<int>& result);

void parallel_find_bounding_box_halo_proc_ids(const MeshBase& mesh,
    Real halo_pad, std::vector<int>& result);

template <class T>
void send_to_neighbors(const Parallel::Communicator& comm,
    std::map<int, std::vector<T> > data, std::vector<T> result)
{
  Parallel::MessageTag tag = comm.get_unique_tag(29417);
  typedef typename std::map<int, std::vector<T> >::iterator iter_t;
  Parallel::Request commReqs[data.size()];
  unsigned int i = 0;
  for (iter_t it = data.begin(); it != data.end(); it++) {
    comm.send(it->first, it->second, commReqs[i++], tag);
  }
  std::vector<T> buffer;
  for(i = 0; i < data.size(); i++) {
    comm.receive(Parallel::any_source, buffer, tag);
    result.insert(result.end(), buffer.begin(), buffer.end());
    buffer.clear();
  }
  for(i = 0; i < data.size(); i++) commReqs[i].wait();
}

template <class T>
void redistribute_particles(Parallel::Communicator& comm,
    std::vector<T>& particles, const PointLocatorBase& pointLocator,
    const std::vector<int>& neighbors)
{
  typedef typename std::vector<T>::iterator Titer_t;
  typedef typename std::vector<int>::iterator intiter_t;
  if(neighbors.empty()) return;
  std::vector<T> newParticles;
  std::map<int, std::vector<T> > outboxes;
  for(intiter_t it = neighbors.begin(); it != neighbors.end(); it++) {
    outboxes[*it];
  }
  for(Titer_t it = particles.begin(); it != particles.end(); it++) {
    int proc_id = pointLocator(*it)->processor_id();
    if(proc_id == comm.rank()) newParticles.push_back(*it);
    else if(outboxes.count(proc_id) > 0) outboxes[proc_id].push_back(*it);
    else {
      libMesh::err << "ERROR, unexpected element processor id" << std::endl;
      libmesh_error();
    }
  }
  send_to_neighbors(comm, outboxes, newParticles);
  particles.swap(newParticles);
}

} // end namespace Halo
} // end namespace libMesh

#endif // LIBMESH_HALO_H
