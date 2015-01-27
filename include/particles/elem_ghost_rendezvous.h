#ifndef LIBMESH_ELEM_GHOST_RENDEZVOUS
#define LIBMESH_ELEM_GHOST_RENDEZVOUS

#include <ostream>

#include "libmesh/libmesh_common.h"
#include "libmesh/mesh_base.h"
#include "libmesh/equation_systems.h"
#include "libmesh/dof_object.h"
#include "libmesh/scatter.h"



namespace libMesh {


  class ElemGhostRendezvous : public ParallelObject, ParallelPrinter
  {
  public:
    ElemGhostRendezvous(MeshBase& mesh);
    ~ElemGhostRendezvous();
    AutoPtr<ScatterDistributed> createElemGhostScatter();
  protected:
    MeshBase& _mesh;
 };// class ElemGhostRendezvous
}// namespace libMesh

#endif //LIBMESH_ELEM_GHOST_RENDEZVOUS
