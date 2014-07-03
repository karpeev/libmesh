#ifndef LIBMESH_NEIGHBORS_EXTENDER_H
#define LIBMESH_NEIGHBORS_EXTENDER_H

#include "libmesh/parallel.h"
#include "libmesh/parallel_object.h"
#include <vector>
#include <set>

namespace libMesh {
namespace Parallel {

class NeighborsExtender : public ParallelObject {
  protected:
    NeighborsExtender(const Communicator& comm);
    inline virtual ~NeighborsExtender() {}
    void setNeighbors(const std::vector<int>& neighbors);
    void resolve(int testDataSize, const char* testData,
        std::vector<int>& outNeighbors, std::vector<int>& inNeighbors);
    virtual void testInit(int root, int testDataSize,
        const char* testData) = 0;
    virtual bool testEdge(int neighbor) = 0;
    virtual bool testNode() = 0;
    virtual void testClear() = 0;

    static void intersect(std::vector<int>& a, const std::vector<int>& b);

  private:
    void commRequests(int numRecvs);
    void recvRequest();
    void commResponses(int numRecvs);
    void recvResponse();
    int commNeighbors();
    int recvNeighbor();
    bool allProcessorsDone();

    std::vector<int> neighbors;

    MessageTag tagRequest;
    MessageTag tagResponse;
    MessageTag tagNeighbor;
    
    std::vector<char> testData;

    std::vector<int>* outNeighbors;
    std::vector<int>* inNeighbors;
    
    std::set<int> requestSet;
    std::vector<int> requestLayer;
    std::set<int> responseSet;
    std::vector<int> responseLayer;
    std::vector<std::vector<int> > responseMsgs;
    std::vector<std::vector<int> > neighborMsgs;
};

} // namespace Parallel
} // namespace libMesh

#endif // LIBMESH_NEIGHBORS_EXTENDER_H

