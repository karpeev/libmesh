#include "libmesh/rank_groups_resolver.h"
#include <string>

void RankGroupsResolver::setComm(MPI_Comm comm) {
  this->comm = comm;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
}

void RankGroupsResolver::setNeighbors(const std::vector<int>& neighbors) {
  deleteOutboxes();
  std::vector<int>::const_iterator it;
  for(it = neighbors.begin(); it != neighbors.end(); it++) {
    outboxes[*it] = new std::stringstream();
  }
}

void RankGroupsResolver::resolve(int edgeTestDataLen,
    const char* edgeTestData, std::vector<int>& result)
{
  searchRequest(myRank, myRank, edgeTestDataLen, edgeTestData);
  do {
    std::string outboxStrs[outboxes.size()];
    MPI_Request mpiReqs[outboxes.size()];
    int i = 0;
    std::map<int, std::stringstream*>::iterator it;
    for(it = outboxes.begin(); it != outboxes.end(); it++) {
      std::string& outboxStr = outboxStrs[i];
      outboxStr = it->second->str();
      it->second->str("");
      MPI_Isend(&outboxStr[0], outboxStr.size(), MPI_CHAR, it->first, tag,
          comm, &mpiReqs[i]);
      i++;
    }
    for(int c = outboxes.size(); c > 0; c--) recvMsg();
    MPI_Status mpiStats[outboxes.size()];
    MPI_Waitall(outboxes.size(), mpiReqs, mpiStats);
  }while(!allProcessorsDone());
  result = searches[myRank]->contacts;
  delete searches[myRank];
  searches.clear();
}

RankGroupsResolver::RankGroupsResolver() {
  setComm(MPI_COMM_WORLD);
}

RankGroupsResolver::~RankGroupsResolver() {
  deleteOutboxes();
}

void RankGroupsResolver::deleteOutboxes() {
  std::map<int, std::stringstream*>::iterator it;
  for(it = outboxes.begin(); it != outboxes.end(); it++) {
    delete it->second;
  }
}

void RankGroupsResolver::recvMsg() {
  MPI_Status status;
  MPI_Probe(MPI_ANY_SOURCE, tag, comm, &status);
  int numBytes;
  MPI_Get_count(&status, MPI_CHAR, &numBytes);
  char buffer[numBytes];
  int source = status.MPI_SOURCE;
  MPI_Recv(buffer, numBytes, MPI_CHAR, source, tag, comm, &status);
  readMsgs(numBytes, buffer, source);
}

bool RankGroupsResolver::allProcessorsDone() {
  char done = (searches[myRank]->requestCount == 0);
  char allDone;
  MPI_Allreduce(&done, &allDone, 1, MPI_CHAR, MPI_LAND, comm);
  return allDone;
}

void RankGroupsResolver::searchRequest(int parent, int root,
    int edgeTestDataLen, const char* edgeTestData)
{
  if(searches.count(root) > 0) {
    std::vector<int> emptyVector;
    addSearchResponseMsg(root, emptyVector, outboxes[parent]);
    return;
  }
  Search* search = new Search();
  searches[root] = search;
  search->parent = parent;
  search->contacts.push_back(myRank);
  edgeTestInit(root, edgeTestDataLen, edgeTestData);
  std::map<int, std::stringstream*>::iterator it;
  for(it = outboxes.begin(); it != outboxes.end(); it++) {
    int rank = it->first;
    if(rank == root || rank == parent) continue;
    if(!edgeTest(rank)) continue;
    addSearchRequestMsg(root, edgeTestDataLen, edgeTestData, it->second);
    search->requestCount++;
  }
  edgeTestClear();
  if(search->requestCount == 0) finishSearch(root);
}

void RankGroupsResolver::finishSearch(int root) {
  if(root == myRank) return;
  Search*& search = searches[root];
  addSearchResponseMsg(root, search->contacts, outboxes[search->parent]);
  delete search;
  search = NULL;
}

void RankGroupsResolver::addSearchRequestMsg(int root,
    int edgeTestDataLen, const char* edgeTestData,
    std::stringstream* outbox)
{
  outbox->write(&code_msgRequest, 1);
  outbox->write((char*)&root, sizeof(int));
  outbox->write((char*)&edgeTestDataLen, sizeof(int));
  outbox->write(edgeTestData, edgeTestDataLen);
}

void RankGroupsResolver::addSearchResponseMsg(int root,
    std::vector<int>& contacts, std::stringstream* outbox)
{
  outbox->write(&code_msgResponse, 1);
  outbox->write((char*)&root, sizeof(int));
  int size = contacts.size();
  outbox->write((char*)&size, sizeof(int));
  outbox->write((char*)&contacts[0], sizeof(int)*size);
}

void RankGroupsResolver::readMsgs(int msgsLen, char* msgs, int source) {
  char* msgsEnd = msgs + msgsLen;
  while(msgs != msgsEnd) {
    char type = msgs[0];
    msgs++;
    if(type == code_msgRequest) msgs = readSearchRequestMsg(msgs, source);
    else msgs = readSearchResponseMsg(msgs); //type == code_msgResponse
  }
}

char* RankGroupsResolver::readSearchRequestMsg(char* msg, int source) {
  int root = *(int*)msg;
  msg += sizeof(int);
  int edgeTestDataLen = *(int*)msg;
  msg += sizeof(int);
  char* edgeTestData = msg;
  msg += edgeTestDataLen;
  searchRequest(source, root, edgeTestDataLen, edgeTestData);
  return msg;
}

char* RankGroupsResolver::readSearchResponseMsg(char* msg) {
  int root = *(int*)msg;
  msg += sizeof(int);
  int contactsLen = *(int*)msg;
  msg += sizeof(int);
  int* contacts = (int*)msg;
  msg += sizeof(int)*contactsLen;
  Search* search = searches[root];
  for(int i = 0; i < contactsLen; i++) {
    search->contacts.push_back(contacts[i]);
  }
  search->requestCount--;
  if(search->requestCount == 0) finishSearch(root);
  return msg;
}

const char RankGroupsResolver::code_msgRequest = 0;
const char RankGroupsResolver::code_msgResponse = 1;
const int RankGroupsResolver::tag = 8147;

