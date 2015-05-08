#include "libmesh/ts_system.h"

namespace libMesh
{

TSSystem::TSSystem(EquationSystems& es,
                   const std::string& name_in,
                   const unsigned int number_in):
  Parent (es, name_in, number_in)
{
  // do nothing right now
}

TSSystem::~TSSystem ()
{
}

}
