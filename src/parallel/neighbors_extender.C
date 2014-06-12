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


// Local Includes -----------------------------------
#include "libmesh/neighbors_extender.h"

namespace libMesh {
namespace Parallel {

NeighborsExtender::NeighborsExtender(const Communicator& comm)
    : ParallelObject(comm),
      tag_request(comm.get_unique_tag(12753)),
      tag_response(comm.get_unique_tag(12754)),
      tag_neighbor(comm.get_unique_tag(12755)),
      out_neighbors(NULL), in_neighbors(NULL)
{}

void NeighborsExtender::set_neighbors(const std::vector<int>& neighbors) {
  this->neighbors = neighbors;
  neighbor_msgs.resize(neighbors.size());
}

const std::vector<int>& NeighborsExtender::get_neighbors() {
  return neighbors;
}

void NeighborsExtender::resolve(int test_data_size, const char* test_data,
    std::vector<int>& out_neighbors, std::vector<int>& in_neighbors)
{
  //initialize vectors
  this->out_neighbors = &out_neighbors;
  this->in_neighbors = &in_neighbors;
  this->test_data.clear();
  this->test_data.reserve(test_data_size);
  for(int i = 0; i < test_data_size; i++) {
    this->test_data.push_back(test_data[i]);
  }
  
  //start by making request to this processor
  request_set.insert(comm().rank());
  request_layer.push_back(comm().rank());
  response_set.insert(comm().rank());
  
  //loop for expanding extended neighbors
  bool is_first_iteration = true;
  do {
    int next_response_layer_size;
    if(is_first_iteration) next_response_layer_size = 1;
    else next_response_layer_size = comm_neighbors();
    is_first_iteration = false;
    int num_requests_made = request_layer.size();
    comm_requests(next_response_layer_size);
    comm_responses(num_requests_made);
  } while(!all_processors_done());
  
  //cleanup
  request_set.clear();
  response_set.clear();
  this->out_neighbors = NULL;
  this->in_neighbors = NULL;
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

void NeighborsExtender::comm_requests(int n_recvs) {
  std::vector<Request> comm_reqs(request_layer.size());
  for(int i = 0; i < (int)request_layer.size(); i++) {
    comm().send(request_layer[i], test_data, comm_reqs[i], tag_request);
  }
  for(int i = 0; i < (int)neighbor_msgs.size(); i++) neighbor_msgs[i].clear();
  for(; n_recvs > 0; n_recvs--) recv_request();
  for(int i = 0; i < (int)request_layer.size(); i++) comm_reqs[i].wait();
  request_layer.clear();
}

void NeighborsExtender::recv_request() {
  std::vector<char> buffer;
  int source = comm().receive(any_source, buffer, tag_request).source();

  int layer_i = response_layer.size();
  response_layer.push_back(source);
  if(response_msgs.size() < response_layer.size()) {
    response_msgs.resize(response_layer.size());
  }
  std::vector<int>& responseMsg = response_msgs[layer_i];
  responseMsg.clear();
  bool node_pass = false;
  std::set<int> neighbors_pass;
  test(source, buffer.size(), &buffer[0], node_pass, neighbors_pass);
  if(node_pass) {
    responseMsg.push_back(1);
    in_neighbors->push_back(source);
    for(int i = 0; i < (int)neighbors.size(); i++) {
      if(neighbors[i] != source && neighbors_pass.count(neighbors[i]) > 0) {
        responseMsg.push_back(neighbors[i]);
        neighbor_msgs[i].push_back(source);
      }
    }
  }
  else {
    responseMsg.push_back(0);
  }
}

void NeighborsExtender::comm_responses(int n_recvs) {
  std::vector<Request> comm_reqs(response_layer.size());
  for(int i = 0; i < (int)response_layer.size(); i++) {
    comm().send(response_layer[i], response_msgs[i], comm_reqs[i],
        tag_response);
  }
  for(; n_recvs > 0; n_recvs--) recv_response();
  for(int i = 0; i < (int)response_layer.size(); i++) comm_reqs[i].wait();
  response_layer.clear();
}

void NeighborsExtender::recv_response() {
  std::vector<int> buffer;
  int source = comm().receive(any_source, buffer, tag_response).source();

  if(buffer[0]) out_neighbors->push_back(source);
  for(int i = 1; i < (int)buffer.size(); i++) {
    bool success = request_set.insert(buffer[i]).second;
    if(success) request_layer.push_back(buffer[i]);
  }
}

int NeighborsExtender::comm_neighbors() {
  std::vector<Request> comm_reqs(neighbors.size());
  for(int i = 0; i < (int)neighbors.size(); i++) {
    comm().send(neighbors[i], neighbor_msgs[i], comm_reqs[i], tag_neighbor);
  }
  int result = 0;
  for(int i = 0; i < (int)neighbors.size(); i++) result += recv_neighbor();
  for(int i = 0; i < (int)neighbors.size(); i++) comm_reqs[i].wait();
  return result;
}

int NeighborsExtender::recv_neighbor() {
  std::vector<int> buffer;
  comm().receive(any_source, buffer, tag_neighbor);

  int result = 0;
  for(int i = 0; i < (int)buffer.size(); i++) {
    if(response_set.insert(buffer[i]).second) result++;
  }
  return result;
}

bool NeighborsExtender::all_processors_done() {
  bool done = (request_layer.empty() ? 1 : 0);
  comm().max(done);
  return done;
}

NeighborsExtender::NeighborsExtender(const NeighborsExtender& other)
    : ParallelObject(*(Communicator*)NULL)
{
  (void)other;
  libmesh_error();
}

NeighborsExtender& NeighborsExtender::operator=(
    const NeighborsExtender& other)
{
  (void)other;
  libmesh_error();
}

} // namespace Parallel
} // namespace libMesh

