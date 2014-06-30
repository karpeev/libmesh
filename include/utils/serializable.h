
#ifndef SERIALIZABLE_H
#define SERIALIZABLE_H

class Serializer;

namespace libMesh {

class Serializable {

public:
  inline virtual ~Serializable() {}
  virtual void serialize(Serializer& serializer);
};

}

#endif // SERIALIZABLE_H
