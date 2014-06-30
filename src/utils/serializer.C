#include "libmesh/libmesh_common.h"
#include "libmesh/serializer.h"
#include "libmesh/serializable.h"

namespace libMesh {

Serializer::Serializer(std::istream& input) : in(&input), out(NULL) {}
Serializer::Serializer(std::ostream& output) : in(NULL), out(&output) {}

bool Serializer::isReading() {return out == NULL;}
bool Serializer::isWriting() {return in == NULL;}

void Serializer::add(Serializable& data) {
  data.serialize(*this);
}

void Serializer::add(int dataSize, void* data) {
  if(in != NULL) in->read((char*)data, dataSize);
  else out->write((char*)data, dataSize);
#ifndef NDEBUG
  std::ios* ios;
  if(in != NULL) ios = in;
  else ios = out;
  if(!ios->good()) {
    if(ios->eof()) libMesh::err << "End Of File encountered during serialization" << std::endl;
    else libMesh::err << "IO Error encountered during serialization" << std::endl;
    libmesh_error();
  }
#endif
}

} // namespace libMesh
