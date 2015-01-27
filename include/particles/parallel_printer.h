#ifndef LIBMESH_PARALLEL_PRINTER
#define LIBMESH_PARALLEL_PRINTER

#include <ostream>
#include <vector>
#include <string>
#include <algorithm>

#include "libmesh/libmesh.h"
#include "libmesh/libmesh_common.h"
#include "libmesh/parallel_object.h"




namespace libMesh {

  // For meshes with enough ghosting
  class ParallelPrinter {
  public:
    ParallelPrinter(const Parallel::Communicator& comm) : _comm(comm) {
#ifdef DEBUG
      command_line_vector("debug",_vdebug);
#endif
    }
    ~ParallelPrinter(){};
#ifdef DEBUG
    bool          debug(const char* s) const {return std::find(_vdebug.begin(),_vdebug.end(),std::string(s)) != _vdebug.end();};
#endif
    std::ostream& rank_print(std::ostream& os,const std::string& suffix="") const {
      os << "["<<_comm.rank()<<"|"<<_comm.size()<<"]"<<suffix;
      return os;
    }
    void          gather_print(const std::ostringstream& sout, std::ostream& os = std::cout) const;
  protected:
    const Parallel::Communicator& _comm;
#ifdef DEBUG
    std::vector<std::string> _vdebug;
#endif

 };// class ParallelPrinter
}// namespace libMesh

#endif //LIBMESH_PARALLEL_PRINTER
