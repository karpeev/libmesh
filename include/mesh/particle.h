
#ifndef LIBMESH_PARTICLE_H
#define LIBMESH_PARTICLE_H

#include "libmesh/point.h"
#include "libmesh/serializable.h;

namespace libMesh {

class Particle : public Point, Serializable {
  inline void serialize(Serializer& serializer) {
    for(unsigned int i = 0; i < LIBMESH_DIM; i++) {
      serializer.add(this->(i));
    }
  }
};

} // end namespace libMesh

#endif // LIBMESH_PARTICLE_H
