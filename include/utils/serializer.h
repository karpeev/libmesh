
namespace libMesh {

class Serializer {

public:
  bool isReading();
  bool isWriting();
  void add(const Serializeable& data);
  void add(int dataSize, void* data);

  //FIXME finish writing...
};

}

