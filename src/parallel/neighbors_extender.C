#include "libmesh/neighbors_extender.h"

//TODO can I use variable length arrays? or should I allocate memory for them?

namespace libMesh {
namespace Parallel {

NeighborsExtender::NeighborsExtender(const Communicator& comm)
    : ParallelObject(comm),
      tagRequest(comm.get_unique_tag(12753)),
      tagResponse(comm.get_unique_tag(12754)),
      tagNeighbor(comm.get_unique_tag(12755)),
      outNeighbors(NULL), inNeighbors(NULL)
{}

void NeighborsExtender::setNeighbors(const std::vector<int>& neighbors) {
  this->neighbors = neighbors;
  neighborMsgs.resize(neighbors.size());
}

void NeighborsExtender::resolve(int testDataSize, const char* testData,
    std::vector<int>& outNeighbors, std::vector<int>& inNeighbors)
{
  this->outNeighbors = &outNeighbors;
  this->inNeighbors = &inNeighbors;
  this->testData.clear();
  this->testData.reserve(testDataSize);
  for(int i = 0; i < testDataSize; i++) this->testData.push_back(testData[i]);
  requestSet.insert(comm().rank());
  requestLayer.push_back(comm().rank());
  responseSet.insert(comm().rank());
  bool isFirstIteration = true;
  do {
    int nextResponseLayerSize;
    if(isFirstIteration) nextResponseLayerSize = 1;
    else nextResponseLayerSize = commNeighbors();
    isFirstIteration = false;
    int numRequestsMade = requestLayer.size();
    commRequests(nextResponseLayerSize);
    commResponses(numRequestsMade);
  } while(!allProcessorsDone());
  requestSet.clear();
  responseSet.clear();
  this->outNeighbors = NULL;
  this->inNeighbors = NULL;
}

void NeighborsExtender::intersect(std::vector<int>& a,
    const std::vector<int>& b)
{
  std::set<int> mySet;
  mySet.insert(a.begin(), a.end());
  a.clear();
  for(unsigned int i = 0; i < b.size(); i++) {
    if(mySet.count(b[i]) > 0) a.push_back(b[i]);
  }
}

void NeighborsExtender::commRequests(int numRecvs) {
  Request commReqs[requestLayer.size()];
  for(int i = 0; i < (int)requestLayer.size(); i++) {
    comm().send(requestLayer[i], testData, commReqs[i], tagRequest);
  }
  for(int i = 0; i < (int)neighborMsgs.size(); i++) neighborMsgs[i].clear();
  for(; numRecvs > 0; numRecvs--) recvRequest();
  for(int i = 0; i < (int)requestLayer.size(); i++) commReqs[i].wait();
  requestLayer.clear();
}

void NeighborsExtender::recvRequest() {
  std::vector<char> buffer;
  int source = comm().receive(any_source, buffer, tagRequest).source();

  int layerI = responseLayer.size();
  responseLayer.push_back(source);
  if(responseMsgs.size() < responseLayer.size()) {
    responseMsgs.resize(responseLayer.size());
  }
  std::vector<int>& responseMsg = responseMsgs[layerI];
  responseMsg.clear();
  testInit(source, buffer.size(), &buffer[0]);
  if(testNode()) {
    responseMsg.push_back(1);
    inNeighbors->push_back(source);
    for(int i = 0; i < (int)neighbors.size(); i++) {
      if(neighbors[i] != source && testEdge(neighbors[i])) {
        responseMsg.push_back(neighbors[i]);
        neighborMsgs[i].push_back(source);
      }
    }
  }
  else {
    responseMsg.push_back(0);
  }
  testClear();
}

void NeighborsExtender::commResponses(int numRecvs) {
  Request commReqs[responseLayer.size()];
  for(int i = 0; i < (int)responseLayer.size(); i++) {
    comm().send(responseLayer[i], responseMsgs[i], commReqs[i], tagResponse);
  }
  for(; numRecvs > 0; numRecvs--) recvResponse();
  for(int i = 0; i < (int)responseLayer.size(); i++) commReqs[i].wait();
  responseLayer.clear();
}

void NeighborsExtender::recvResponse() {
  std::vector<int> buffer;
  int source = comm().receive(any_source, buffer, tagResponse).source();

  if(buffer[0]) outNeighbors->push_back(source);
  for(int i = 1; i < (int)buffer.size(); i++) {
    bool success = requestSet.insert(buffer[i]).second;
    if(success) requestLayer.push_back(buffer[i]);
  }
}

int NeighborsExtender::commNeighbors() {
  Request commReqs[neighbors.size()];
  for(int i = 0; i < (int)neighbors.size(); i++) {
    comm().send(neighbors[i], neighborMsgs[i], commReqs[i], tagNeighbor);
  }
  int result = 0;
  for(int i = 0; i < (int)neighbors.size(); i++) result += recvNeighbor();
  for(int i = 0; i < (int)neighbors.size(); i++) commReqs[i].wait();
  return result;
}

int NeighborsExtender::recvNeighbor() {
  std::vector<int> buffer;
  comm().receive(any_source, buffer, tagNeighbor);

  int result = 0;
  for(int i = 0; i < (int)buffer.size(); i++) {
    if(responseSet.insert(buffer[i]).second) result++;
  }
  return result;
}

bool NeighborsExtender::allProcessorsDone() {
  bool done = (requestLayer.empty() ? 1 : 0);
  comm().max(done);
  return done;
}

NeighborsExtender::NeighborsExtender(const NeighborsExtender& other)
    : ParallelObject(*(Communicator*)NULL)
{
  libmesh_error();
}

NeighborsExtender& NeighborsExtender::operator=(
    const NeighborsExtender& other)
{
  libmesh_error();
}

} // namespace Parallel
} // namespace libMesh

