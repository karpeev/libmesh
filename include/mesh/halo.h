
#ifndef LIBMESH_HALO_H
#define LIBMESH_HALO_H

#include "libmesh/libmesh_common.h"

#include <vector>

namespace libMesh {

class MeshBase;

namespace Halo {

void find_neighbor_proc_ids(const MeshBase& mesh, std::vector<int>& result);

void parallel_find_bounding_box_halo_proc_ids(const MeshBase& mesh,
    Real halo_pad, std::vector<int>& result);

} // end namespace Halo
} // end namespace libMesh

#endif // LIBMESH_HALO_H
