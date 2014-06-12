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
     * HaloManager::find_neighbor_processors(...).  Will make an
     * independent copy of the given vector.
     */
    void set_neighbors(const std::vector<int>& neighbors);
    
    /**
     * @returns the neighbors vector that was set using set_neighbors.
     */
    const std::vector<int>& get_neighbors();

    /**
     * Determine the extended neighbors of this processor based on a graph
     * traversal in parallel.  This is a collective operation.  The
     * \p test_data block of memory of length \p test_data_size will be passed
     * to the testInit(...) function on other processors.  This method
     * will follow the nodes and edges allowed by testNode and testEdge
     * to increase the connectivity of this processor.  The \p out_neighbors
     * vector will be filled with the extended neighbors of this processor.
     * The \p in_neighbors vector will be filled with all processors who
     * have this processor as an extended neighbor.
     */
    void resolve(int test_data_size, const char* test_data,
        std::vector<int>& out_neighbors, std::vector<int>& in_neighbors);

    /**
     * Tests this node and its edges to determine which nodes/edges
     * should be followed.  Given the \p root processor ID that is
     * requesting the tests, as well as a \p test_data block of memory
     * of length \p test_data_size from that root processor that is important
     * for the tests.  Should set \p node_pass to be true if this
     * processor should be accepted as an extended neighbor of \p root.
     * Should add neighboring processor IDs to \p neighbors_pass for
     * each edge that should be followed (this should be a subset of
     * get_neighbors()).
     */
    virtual void test(int root, int test_data_size, const char* test_data,
        bool& node_pass, std::set<int>& neighbors_pass) = 0;

    /**
     * Computes the set intersection of the two vectors \p a and \p b,
     * and replaces \p a with the contents of that intersection.
     * Assumes vectors do not contain duplicate values.
     */
    static void intersect(std::vector<int>& a, const std::vector<int>& b);

  private:
  
    /**
     * Send and receive request messages.  A request message will send
     * the test_data.  \p n_recvs is the number of request messages that
     * will be received from other processors.
     */
    void comm_requests(int n_recvs);
    
    /**
     * Receive a single request message.
     */
    void recv_request();
  
    /**
     * Send and receive response messages.  A response message will send
     * whether the source processor should be included in the dest processor's
     * extended neighbors, along with the neighbors of the source processor
     * that should be checked.  \p n_recvs is the number of response messages
     * that will be received from other processors.
     */
    void comm_responses(int n_recvs);
    
    /**
     * Receive a single response message.
     */
    void recv_response();
    
    /**
     * Send and receive messages from neighbors.  These neighbor messages
     * contain the processor IDs that the source processor referred the
     * destination processor to.  This is needed  to determine how many
     * requests will be received for the next comm_requests call.
     */
    int comm_neighbors();
    
    /**
     * Receive a single neighbor message.
     */
    int recv_neighbor();

    /**
     * Performs an all-reduce operation to determine if all processors
     * are finished forming their extended neighbors.
     */
    bool all_processors_done();

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
    MessageTag tag_request;

    /**
     * Communication tag for sending extended connectivity results
     * to another processor.
     */
    MessageTag tag_response;

    /**
     * Communication tag for communicating next request counts to
     * immediate neighbors of this processor.
     */
    MessageTag tag_neighbor;

    /**
     * The test_data originating from this processor (argument of
     * resolve(...)), stored as a vector for sending to other processors.
     */
    std::vector<char> test_data;

    /**
     * The outgoing extended neighbors (argument of resolve(...)).
     */
    std::vector<int>* out_neighbors;

    /**
     * The incoming extended neighbors (argument of resolve(...)).
     */
    std::vector<int>* in_neighbors;
    
    /**
     * Processor IDs that requests were sent to, including
     * contents of request_layer.
     */
    std::set<int> request_set;
    
    /**
     * Processor IDs that requests should be sent to next.
     */
    std::vector<int> request_layer;
    
    /**
     * Processor IDs that responses were sent to or are about to be sent to.
     */
    std::set<int> response_set;
    
    /**
     * Processor IDs that responses should be sent to next.
     */
    std::vector<int> response_layer;
    
    /**
     * Buffers for response messages.
     */
    std::vector<std::vector<int> > response_msgs;
    
    /**
     * Buffers for neighbor messages.
     */
    std::vector<std::vector<int> > neighbor_msgs;
};

} // namespace Parallel
} // namespace libMesh

#endif // LIBMESH_NEIGHBORS_EXTENDER_H

