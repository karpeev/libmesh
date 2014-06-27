
namespace libMesh {

class Serializable {

public:
  inline virtual ~Serializable() {}
  virtual void serialize(Serializer& serializer) const;
};

}
