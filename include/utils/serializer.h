#ifndef LIBMESH_SERIALIZER_H
#define LIBMESH_SERIALIZER_H

#include <istream>
#include <ostream>

namespace libMesh {

template <class T>
class Serializer {

public:
  read(std::istream stream, T& buffer);
  write(std::ostream stream, const T& buffer);

} // namespace libMesh


#endif // LIBMESH_SERIALIZER_H
