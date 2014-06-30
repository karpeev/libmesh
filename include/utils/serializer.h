
#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <vector>
#include <cstdint>
#include <istream>
#include <ostream>

namespace libMesh {

class Serializable;

class Serializer {

public:
  Serializer(std::istream& input);
  Serializer(std::ostream& output);

  bool isReading();
  bool isWriting();
  void add(Serializable& data);
  void add(int dataSize, void* data);
  inline void add(std::int8_t& data) {add(sizeof(data), &data);}
  inline void add(std::int16_t& data) {add(sizeof(data), &data);}
  inline void add(std::int32_t& data) {add(sizeof(data), &data);}
  inline void add(std::int64_t& data) {add(sizeof(data), &data);}
  inline void add(std::uint8_t& data) {add(sizeof(data), &data);}
  inline void add(std::uint16_t& data) {add(sizeof(data), &data);}
  inline void add(std::uint32_t& data) {add(sizeof(data), &data);}
  inline void add(std::uint64_t& data) {add(sizeof(data), &data);}
  inline void add(float& data) {add(sizeof(data), &data);}
  inline void add(double& data) {add(sizeof(data), &data);}
  
  template <class T>
  void add(std::vector<T>& data) {
    typename std::vector<T>::size_type size = data.size();
    add(size);
    data.resize(size);
    for(unsigned int i = 0; i < size; i++) add(data[i]);
  }
  
private:
  std::istream* const in;
  std::ostream* const out;
};

} // namespace libMesh


#endif // SERIALIZER_H