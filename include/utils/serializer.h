#ifndef LIBMESH_SERIALIZER_H
#define LIBMESH_SERIALIZER_H

#include <istream>
#include <ostream>

namespace libMesh {

template <class T>
class Serializer {

public:
  virtual void read(std::istream& stream, T& buffer) const = 0;
  virtual void write(std::ostream& stream, const T& buffer) const = 0;

};

} // namespace libMesh

#endif // LIBMESH_SERIALIZER_H
