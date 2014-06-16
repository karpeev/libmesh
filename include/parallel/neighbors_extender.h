#include <vector>
#include <set>
#include "mpi.h"

namespace libMesh {
namespace Parallel {

//TODO replace MPI calls with calls to Libmesh's Communicator functions

class NeighborsExtender {
  public:
    void setComm(MPI_Comm comm);
    void setNeighbors(const std::vector<int>& neighbors);
    void resolve(int testDataSize, const char* testData,
        std::vector<int>& result);

  protected:
    NeighborsExtender();
    inline virtual ~NeighborsExtender() {}
    virtual void testInit(int root, int testDataSize,
        const char* testData) = 0;
    virtual bool testEdge(int neighbor) = 0;
    virtual bool testNode() = 0;
    virtual void testClear() = 0;

  private:
    void commRequests(int numRecvs);
    void recvRequest();
    void commResponses(int numRecvs);
    void recvResponse();
    int commNeighbors();
    int recvNeighbor();
    void probe(int tag, MPI_Datatype datatype, int& source, int& size);
    void recv(void* buf, int count, MPI_Datatype datatype, int source,
        int tag);
    bool allProcessorsDone();

    MPI_Comm comm;
    int myRank;
    std::vector<int> neighbors;
    
    int testDataSize;
    const char* testData;
    
    std::vector<int> contacts;
    std::set<int> requestSet;
    std::vector<int> requestLayer;
    std::set<int> responseSet;
    std::vector<int> responseLayer;
    std::vector<std::vector<int> > responseMsgs;
    std::vector<std::vector<int> > neighborMsgs;

    static const int tagRequest, tagResponse, tagNeighbor;
};

} // namespace Parallel
} // namespace libMesh

