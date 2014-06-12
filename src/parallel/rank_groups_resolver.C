#include "libmesh/rank_groups_resolver.h"

void RankGroupsResolver::setComm(MPI_Comm comm) {
  this->comm = comm;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
}

void RankGroupsResolver::setNeighbors(const std::vector<int>& neighbors) {
  this->neighbors = neighbors;
  neighborMsgs.resize(neighbors.size());
}

void RankGroupsResolver::resolve(int testDataSize, const char* testData,
    std::vector<int>& result)
{
  this->testDataSize = testDataSize;
  this->testData = testData;
  requestSet.insert(myRank);
  requestLayer.push_back(myRank);
  responseSet.insert(myRank);
  bool isFirstIteration = true;
  int nextResponseLayerSize = 1;
  do {
    int nextResponseLayerSize;
    if(isFirstIteration) nextResponseLayerSize = 1;
    else nextResponseLayerSize = commNeighbors();
    isFirstIteration = false;
    int numRequestsMade = requestLayer.size();
    commRequests(nextResponseLayerSize);
    commResponses(numRequestsMade);
  } while(!allProcessorsDone());
  result = contacts;
  contacts.clear();
  requestSet.clear();
  responseSet.clear();
}

RankGroupsResolver::RankGroupsResolver() {
  setComm(MPI_COMM_WORLD);
}

void RankGroupsResolver::commRequests(int numRecvs) {
  MPI_Request mpiReqs[requestLayer.size()];
  for(int i = 0; i < requestLayer.size(); i++) {
    //we assume MPI_Isend does not alter the contents of the buffer...
    MPI_Isend((void*)testData, testDataSize, MPI_CHAR, requestLayer[i],
        tagRequest, comm, &mpiReqs[i]);
  }
  for(int i = 0; i < neighborMsgs.size(); i++) neighborMsgs[i].clear();
  for(; numRecvs > 0; numRecvs--) recvRequest();
  MPI_Status mpiStats[requestLayer.size()];
  MPI_Waitall(requestLayer.size(), mpiReqs, mpiStats);
  requestLayer.clear();
}

void RankGroupsResolver::recvRequest() {
  int source, bufferSize;
  probe(tagRequest, MPI_CHAR, source, bufferSize);
  char buffer[bufferSize];
  recv(buffer, bufferSize, MPI_CHAR, source, tagRequest);

  int layerI = responseLayer.size();
  responseLayer.push_back(source);
  if(responseMsgs.size() < responseLayer.size()) {
    responseMsgs.resize(responseLayer.size());
  }
  std::vector<int>& responseMsg = responseMsgs[layerI];
  responseMsg.clear();
  testInit(source, bufferSize, buffer);
  if(testNode()) {
    responseMsg.push_back(1);
    for(int i = 0; i < neighbors.size(); i++) {
      if(testEdge(neighbors[i])) {
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

void RankGroupsResolver::commResponses(int numRecvs) {
  MPI_Request mpiReqs[responseLayer.size()];
  for(int i = 0; i < responseLayer.size(); i++) {
    MPI_Isend(&responseMsgs[i][0], responseMsgs[i].size(), MPI_INT,
        responseLayer[i], tagResponse, comm, &mpiReqs[i]);
  }
  for(; numRecvs > 0; numRecvs--) recvResponse();
  MPI_Status mpiStats[responseLayer.size()];
  MPI_Waitall(responseLayer.size(), mpiReqs, mpiStats);
  responseLayer.clear();
}

void RankGroupsResolver::recvResponse() {
  int source, bufferSize;
  probe(tagResponse, MPI_INT, source, bufferSize);
  int buffer[bufferSize];
  recv(buffer, bufferSize, MPI_INT, source, tagResponse);

  if(buffer[0]) contacts.push_back(source);
  for(int i = 1; i < bufferSize; i++) {
    bool success = requestSet.insert(buffer[i]).second;
    if(success) requestLayer.push_back(buffer[i]);
  }
}

int RankGroupsResolver::commNeighbors() {
  MPI_Request mpiReqs[neighbors.size()];
  for(int i = 0; i < neighbors.size(); i++) {
    MPI_Isend(&neighborMsgs[i][0], neighborMsgs[i].size(), MPI_INT,
        neighbors[i], tagNeighbor, comm, &mpiReqs[i]);
  }
  int result = 0;
  for(int i = 0; i < neighbors.size(); i++) result += recvNeighbor();
  MPI_Status mpiStats[neighbors.size()];
  MPI_Waitall(neighbors.size(), mpiReqs, mpiStats);
  return result;
}

int RankGroupsResolver::recvNeighbor() {
  int source, bufferSize;
  probe(tagNeighbor, MPI_INT, source, bufferSize);
  int buffer[bufferSize];
  recv(buffer, bufferSize, MPI_INT, source, tagNeighbor);

  int result = 0;
  for(int i = 0; i < bufferSize; i++) {
    if(responseSet.insert(buffer[i]).second) result++;
  }
  return result;
}

void RankGroupsResolver::probe(int tag, MPI_Datatype datatype,
    int& source, int& size)
{
  MPI_Status status;
  MPI_Probe(MPI_ANY_SOURCE, tag, comm, &status);
  MPI_Get_count(&status, datatype, &size);
  source = status.MPI_SOURCE;
}

void RankGroupsResolver::recv(void* buf, int count, MPI_Datatype datatype,
    int source, int tag)
{
  MPI_Status status;
  MPI_Recv(buf, count, datatype, source, tag, comm, &status);
}

bool RankGroupsResolver::allProcessorsDone() {
  char done = requestLayer.empty();
  char allDone;
  MPI_Allreduce(&done, &allDone, 1, MPI_CHAR, MPI_LAND, comm);
  return allDone;
}

const int RankGroupsResolver::tagRequest = 8147;
const int RankGroupsResolver::tagResponse = 8148;
const int RankGroupsResolver::tagNeighbor = 8149;

