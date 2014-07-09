// The libMesh Finite Element Library.
// Copyright (C) 2002-2014 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


#ifndef LIBMESH_NEIGHBORS_EXTENDER_H
#define LIBMESH_NEIGHBORS_EXTENDER_H

// Local Includes -----------------------------------
#include "libmesh/parallel.h"
#include "libmesh/parallel_object.h"

// C++ Includes   -----------------------------------
#include <vector>
#include <set>

namespace libMesh {
namespace Parallel {

/**
 * This is the \p NeighborsExtender class.  It is used to increase the
 * connectivity of the graph of processors in parallel.  In this graph,
 * the nodes are processors.  Starting at this processor, we will call
 * the methods testEdge and testNode to determine if the neighboring
 * edges/nodes should be followed, and then check the neighbors of the
 * new nodes, etc., building up an extended neighbor set.  This is a virtual
 * class.  A subclass of this class should implement the testInit, testEdge,
 * testNode, and testClear methods.
 *
 * \author  Matthew D. Michelotti
 */

class NeighborsExtender : public ParallelObject {

  protected:
    
    /**
     * Constructor.  Inputs the Communicator \p comm used for communicating
     * with other processors.
     */
    NeighborsExtender(const Communicator& comm);

    /**
     * Destructor.
     */
    inline virtual ~NeighborsExtender() {}

    /**
     * Set the immediate neighbors of this processor.  The \p neighbors
     * vector contains the processor IDs of these neighbors.  These
     * neighbors can be chosen however you like, perhaps by calling
     * HaloManager::find_neighbor_processors(...).
     */
    void setNeighbors(const std::vector<int>& neighbors);

    /**
     * Determine the extended neighbors of this processor based on a graph
     * traversal in parallel.  This is a collective operation.  The
     * \p testData block of memory of length \p testDataSize will be passed
     * to the testInit(...) function on other processors.  This method
     * will follow the nodes and edges allowed by testNode and testEdge
     * to increase the connectivity of this processor.  The \p outNeighbors
     * vector will be filled with the extended neighbors of this processor.
     * The \p inNeighbors vector will be filled with all processors who
     * have this processor as an extended neighbor.
     */
    void resolve(int testDataSize, const char* testData,
        std::vector<int>& outNeighbors, std::vector<int>& inNeighbors);

    /**
     * Prepare for testing nodes and edges.  Given the \p root processor
     * ID that is requesting the tests, as well as a \p testData block
     * of memory of length \p testDataSize from that root processor
     * that is important for the tests.
     */
    virtual void testInit(int root, int testDataSize,
        const char* testData) = 0;

    /**
     * @returns true if the edge between this processor and the processor
     * with processor ID specified by \p neighbor should be followed
     * for the current testData passed to testInit.
     */
    virtual bool testEdge(int neighbor) = 0;

    /**
     * @returns true if this processor should be accepted as an extended
     * neighbor of the processor specified in testInit.
     */
    virtual bool testNode() = 0;

    /**
     * Clear any data that was stored from a call to testInit.
     */
    virtual void testClear() = 0;

    /**
     * Computes the set intersection of the two vectors \p a and \p b,
     * and replaces \p a with the contents of that intersection.
     * Assumes vectors do not contain duplicate values.
     */
    static void intersect(std::vector<int>& a, const std::vector<int>& b);

  private:
  
    /**
     * Send and receive request messages.  A request message will send
     * the testData.  \p numRecvs is the number of request messages that
     * will be received from other processors.
     */
    void commRequests(int numRecvs);
    
    /**
     * Receive a single request message.
     */
    void recvRequest();
  
    /**
     * Send and receive response messages.  A response message will send
     * whether the source processor should be included in the dest processor's
     * extended neighbors, along with the neighbors of the source processor
     * that should be checked.  \p numRecvs is the number of response messages
     * that will be received from other processors.
     */
    void commResponses(int numRecvs);
    
    /**
     * Receive a single response message.
     */
    void recvResponse();
    
    /**
     * Send and receive messages from neighbors.  These neighbor messages
     * contain the processor IDs that the source processor referred the
     * destination processor to.  This is needed  to determine how many
     * requests will be received for the next commRequests call.
     */
    int commNeighbors();
    
    /**
     * Receive a single neighbor message.
     */
    int recvNeighbor();

    /**
     * Performs an all-reduce operation to determine if all processors
     * are finished forming their extended neighbors.
     */
    bool allProcessorsDone();

    /**
     * Overridden copy constructor that should never be used.
     */
    NeighborsExtender(const NeighborsExtender& other);

    /**
     * Overridden assignment operator that should never be used.
     */
    NeighborsExtender& operator=(const NeighborsExtender& other);

    /**
     * The processor IDs of the immediate neighbors of this processor.
     */
    std::vector<int> neighbors;

    /**
     * Communication tag used for requesting extended connectivity results
     * from another processor.
     */
    MessageTag tagRequest;

    /**
     * Communication tag for sending extended connectivity results
     * to another processor.
     */
    MessageTag tagResponse;

    /**
     * Communication tag for communicating next request counts to
     * immediate neighbors of this processor.
     */
    MessageTag tagNeighbor;

    /**
     * The testData originating from this processor (argument of
     * resolve(...)), stored as a vector for sending to other processors.
     */
    std::vector<char> testData;

    /**
     * The outgoing extended neighbors (argument of resolve(...)).
     */
    std::vector<int>* outNeighbors;

    /**
     * The incoming extended neighbors (argument of resolve(...)).
     */
    std::vector<int>* inNeighbors;
    
    /**
     * Processor IDs that requests were sent to, including
     * contents of requestLayer.
     */
    std::set<int> requestSet;
    
    /**
     * Processor IDs that requests should be sent to next.
     */
    std::vector<int> requestLayer;
    
    /**
     * Processor IDs that responses were sent to or are about to be sent to.
     */
    std::set<int> responseSet;
    
    /**
     * Processor IDs that responses should be sent to next.
     */
    std::vector<int> responseLayer;
    
    /**
     * Buffers for response messages.
     */
    std::vector<std::vector<int> > responseMsgs;
    
    /**
     * Buffers for neighbor messages.
     */
    std::vector<std::vector<int> > neighborMsgs;
};

} // namespace Parallel
} // namespace libMesh

#endif // LIBMESH_NEIGHBORS_EXTENDER_H

