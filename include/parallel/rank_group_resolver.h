#include <vector>
#include <map>
#include <sstream>
#include "mpi.h"

class RankGroupsResolver {
  public:
    void setComm(MPI_Comm comm);
    void setNeighbors(const std::vector<int>& neighbors);
    void resolve(int edgeTestDataLen, const char* edgeTestData,
        std::vector<int>& result);

  protected:
    RankGroupsResolver();
    virtual ~RankGroupsResolver();
    virtual void edgeTestInit(int root, int edgeTestDataLen,
        const char* edgeTestData) = 0;
    virtual bool edgeTest(int neighbor) = 0;
    virtual void edgeTestClear() = 0;

  private:
    void deleteOutboxes();
    void recvMsg();
    bool allProcessorsDone();
    void searchRequest(int parent, int root, int edgeTestDataLen,
        const char* edgeTestData);
    void finishSearch(int root);
    void addSearchRequestMsg(int root, int edgeTestDataLen,
        const char* edgeTestData, std::stringstream* outbox);
    void addSearchResponseMsg(int root, std::vector<int>& contacts,
        std::stringstream* outbox);
    void readMsgs(int msgsLen, char* msgs, int source);
    char* readSearchRequestMsg(char* msg, int source);
    char* readSearchResponseMsg(char* msg);

    struct Search {
      std::vector<int> contacts;
      int requestCount;
      int parent;

      inline Search() : requestCount(0) {}
    };

    MPI_Comm comm;
    int myRank;

    std::map<int, Search*> searches;
    std::map<int, std::stringstream*> outboxes;

    static const char code_msgRequest, code_msgResponse;
    static const int tag;
};

