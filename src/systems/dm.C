#include <libmesh/dm.h>
#include <libmesh/system.h>

namespace libMesh {
  DM::DM(EquationSystems& es, const std::string& sys_name) : ParallelObject(es.comm()),_es(es),_sys_name(sys_name),_sys(es.get_system<System>(sys_name)) {}

  DM::~DM(){}

  void DM::initVector(NumericVector<Number>& vec, const ParallelType ptype) {/* TODO */}

  void DM::initMatrix(SparseMatrix<Number>& vec) {/* TODO */}


  UniquePtr<DM> DM::coarsen(){
    /* Migrate stuff from dm_sample and mg_tool here. */
    return UniquePtr<DM>(new DM(_es,""));
  }

  void DM::assembleInterpolation(const DM& fine, SparseMatrix<Number> &interp) {
    /* Migrate stuff from dm_sample and mg_tool here. */
  }

}// namespace libMesh
