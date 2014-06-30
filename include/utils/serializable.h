
#ifndef LIBMESH_SERIALIZABLE_H
#define LIBMESH_SERIALIZABLE_H

class Serializer;

namespace libMesh {

class Serializable {

public:
  inline virtual ~Serializable() {}
  virtual void serialize(Serializer& serializer);
};

}

#endif // LIBMESH_SERIALIZABLE_H
