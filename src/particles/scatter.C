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
#include "libmesh/libmesh_logging.h"
#include "libmesh/parallel.h"
#ifdef LIBMESH_HAVE_MPI
#include "mpi.h"
#endif
#include "libmesh/scatter.h"

// C++ Includes   -----------------------------------
#include <algorithm>

namespace libMesh {

using Parallel::Request;
using Parallel::MessageTag;

void Scatter::add_ochannel(int c)
{
  _setup = false;
  if (_ochannel_sources.find(c) == _ochannel_sources.end()) {
    _ochannel_sources[c] = std::vector<int>();
  }
}

void Scatter::add_ochannel_source(int ochannel, int source)
{
  std::map<int,std::vector<int> >::iterator ochannel_sources_it = _ochannel_sources.find(ochannel);
  if (ochannel_sources_it == _ochannel_sources.end()) {
    // TODO: obtain ochannel_it as part of insertion
    _ochannel_sources[ochannel] = std::vector<int>();
    ochannel_sources_it = _ochannel_sources.find(ochannel);
  }
  ochannel_sources_it->second.push_back(source);
}

void Scatter::add_ochannel_sources(int ochannel, const std::vector<int>& sources)
{
  std::map<int,std::vector<int> >::iterator ochannel_sources_it = _ochannel_sources.find(ochannel);
  if (ochannel_sources_it == _ochannel_sources.end()) {
    // TODO: obtain ochannel_sources_it as part of insertion
    _ochannel_sources[ochannel] = std::vector<int>();
    ochannel_sources_it = _ochannel_sources.find(ochannel);
  }
  std::copy(sources.begin(),sources.end(),std::back_inserter(ochannel_sources_it->second));
}

void Scatter::add_ichannel(int ichannel)
{
  _setup = false;
  if (_ichannel_sources.find(ichannel) == _ichannel_sources.end()) {
    _ichannel_sources[ichannel] = std::vector<int>();
  }
}

void Scatter::add_ichannel_source(int ichannel, int source)
{
  std::map<int,std::vector<int> >::iterator ichannel_sources_it = _ichannel_sources.find(ichannel);
  if (ichannel_sources_it == _ichannel_sources.end()) {
    // TODO: obtain ichannel_sources_it as part of insertion
    _ichannel_sources[ichannel] = std::vector<int>();
    ichannel_sources_it = _ichannel_sources.find(ichannel);
  }
  ichannel_sources_it->second.push_back(source);
}

void Scatter::add_ichannel_sources(int ichannel, const std::vector<int>& sources)
{
  std::map<int,std::vector<int> >::iterator ichannel_sources_it = _ichannel_sources.find(ichannel);
  if (ichannel_sources_it == _ichannel_sources.end()) {
    // TODO: obtain ichannel_sources_it as part of insertion
    _ichannel_sources[ichannel] = std::vector<int>();
    ichannel_sources_it = _ichannel_sources.find(ichannel);
  }
  std::copy(sources.begin(),sources.end(),std::back_inserter(ichannel_sources_it->second));
}

const std::vector<int> ScatterDistributed::NO_CHANNELS = std::vector<int>();

const std::vector<int>& ScatterDistributed::__rank_get_channels(int r, const std::map<int,std::vector<int> >& rank_channels)
{
  std::map<int,std::vector<int> >::const_iterator rank_channels_it = rank_channels.find(r);
  if (rank_channels_it == rank_channels.end()) {
    return ScatterDistributed::NO_CHANNELS;
  }
  return rank_channels_it->second;
}

void ScatterDistributedMPI::rank_add_ochannel(int r, int c)
{
  // TODO: check that r is within the comm
  if (_rank_ochannels.find(r) == _rank_ochannels.end()) {
    _rank_ochannels[r] = std::vector<int>();
  }
  _rank_ochannels[r].push_back(c);
}

void ScatterDistributedMPI::rank_add_ochannels(int r, const std::vector<int>& channels)
{
  // TODO: check that r is within the comm
  std::map<int,std::vector<int> >::iterator rit = _rank_ochannels.find(r);
  if (rit == _rank_ochannels.end()) {
    _rank_ochannels[r] = std::vector<int>();
    // TODO: obtain rit as part of insertion
    rit = _rank_ochannels.find(r);
  }
  std::copy(channels.begin(),channels.end(),std::back_inserter(rit->second));
}

void ScatterDistributedMPI::rank_add_ichannel(int r, int c)
{
  // TODO: check r is within the comm
  if (_rank_ichannels.find(r) == _rank_ichannels.end()) {
    _rank_ichannels[r] = std::vector<int>();
  } else {
    _rank_ichannels[r].push_back(c);
  }
}

void ScatterDistributedMPI::rank_add_ichannels(int r, const std::vector<int>& channels)
{
  // TODO: check that r is within the comm
  std::map<int,std::vector<int> >::iterator rit = _rank_ichannels.find(r);
  if (rit == _rank_ichannels.end()) {
    _rank_ichannels[r] = std::vector<int>();
    // TODO: obtain rit as part of insertion
    rit = _rank_ichannels.find(r);
  }
  std::copy(channels.begin(),channels.end(),std::back_inserter(rit->second));
}

void ScatterDistributedMPI::setup() {
  START_LOG("setup","ScatterDistributedMPI");
  _setup = true;
  STOP_LOG("setup","ScatterDistributedMPI");
}

void ScatterDistributedMPI::scatter(Scatter::Packer& packer, Scatter::Unpacker& unpacker) const
{
  libmesh_assert_msg(_setup, "Cannot scatter: ScatterDistributedMPI not set up.");
  START_LOG("scatter", "ScatterDistributedMPI");

  // TODO: we can further specialize the scatter on whether or not we expect incoming packets on all channels,
  //       and whether or not we expect incoming packets from all sources.
  //      A. Currently we expect packets on most channels, so the rendezvous data tells the receiver how many
  //         (and which) channels to receive from a given peer rank. However, currently the rendezvous data
  //         don't contain the per-rank information on how many sources feed into a given channel from a given
  //         rank. Thus, for each channel we need to prepend its buffer data segment by the number of sources
  //         fed into it.
  //      B. As mentioned above, there are no data on the possible sources feeding into a given channel from
  //         a given rank, we have no choice but to prepend the number of active sources to the channel's data
  //         segment, and to prepend the id of each active source to each active source's data subsegment.
  //         Thus, the receiving rank doesn't know a priori which sources are arriving from a given sending
  //         rank on a given channel. The only way to discover the arriving sources, is by removing their
  //         number from the head of the channel's data segment and their ids from the heads of the sources'
  //         subsegments.
  //             There are, however, rendezvous data on all of the sources fed into each channel (without a
  //         breakdown) of sources by peer rank.  Therefore we could at least check that only expected sources
  //         arrive from a given rank on a given channel.
  // ---------------------------------------------------------------------------------------------------------
  //      C. Assumptions  used in A waste bandwidth if most channels are empty: in that case we'll be sending
  //         a lot of zeros indicating no sources feeding into the inactive channels. Instead, we could start
  //         the buffer with the number of active channels and start each channel data segment in the buffer
  //         with the buffer's id, prepending to the size (i.e., the number of active sources in the channel).
  //         This should only be done if the additional data sent are expected to be offset by the savings of
  //         only sending the data for the active channels.
  //      D. We could further specialize the scatter and allow the rendezvous to exchange the per-rank
  //         channel-source composition data.  This would allow us to avoid having to send the source numbers
  //         and ids.  This would be analogous to A for channel data -- only beneficial when most sources
  //         are sending.
  //  Note that we regard the rendezvous data -- the per-rank incoming channels and, possibly in the future,
  //  the per-rank incoming channel sources, as static -- exchanged up front during the rendezvous -- and
  //  an opportunity for optimizing the communication thanks to this a priori information.
  //
  std::vector<Request> reqs(_rank_ochannels.size());
  // iterate over a rank's ochannels and pack all of the ochannels into a single OutBuffer
  int rank_ocount = 0;
  for(std::map<int,std::vector<int> >::const_iterator rank_ochannels_it = _rank_ochannels.begin(); rank_ochannels_it != _rank_ochannels.end(); ++rank_ochannels_it,++rank_ocount) {
    int rank_o = rank_ochannels_it->first;
    Scatter::OutBuffer r_obuffer;
    if (_prepack) {
      // Iterate over rank's ochannels and prepack the sources going into each ochannel
      const std::vector<int>& r_ochannels = rank_ochannels_it->second;
      for (std::vector<int>::const_iterator r_ochannels_it = r_ochannels.begin(); r_ochannels_it != r_ochannels.end(); ++r_ochannels_it) {
	int r_ochannel = *r_ochannels_it;
	// Iterate over the r_ochannel's sources and prepack the r_obuffer segment
	std::map<int,std::vector<int> >::const_iterator r_ochannel_sources_it = _ochannel_sources.find(r_ochannel);
	const std::vector<int>& r_oc_sources = r_ochannel_sources_it->second;
	for (std::vector<int>::const_iterator r_oc_sources_it = r_oc_sources.begin(); r_oc_sources_it != r_oc_sources.end(); ++r_oc_sources_it) {
	  // prepack
	  int r_oc_source = *r_oc_sources_it;
	  packer.prepack(r_oc_source,r_ochannel);
	}
      }
    }
    // Iterate over rank's ochannels and pack the sources going into each ochannel
    const std::vector<int>& r_ochannels = rank_ochannels_it->second;
    for (std::vector<int>::const_iterator r_ochannels_it = r_ochannels.begin(); r_ochannels_it != r_ochannels.end(); ++r_ochannels_it) {
      int r_ochannel = *r_ochannels_it;
      std::map<int,std::vector<int> >::const_iterator r_ochannel_sources_it = _ochannel_sources.find(r_ochannel);
      const std::vector<int>& r_oc_sources = r_ochannel_sources_it->second;
      int r_oc_sources_size = r_oc_sources.size();
      r_obuffer.write(r_oc_sources_size);
      // Iterate over the r_ochannel's sources and pack the r_obuffer segment, one subsegment at a time
      for (std::vector<int>::const_iterator r_oc_sources_it = r_oc_sources.begin(); r_oc_sources_it != r_oc_sources.end(); ++r_oc_sources_it) {
	int r_oc_source = *r_oc_sources_it;
	// write the source id
	r_obuffer.write(r_oc_source);
	// pack the packets
	packer.pack(r_oc_source,r_ochannel,r_obuffer);
      }
    }
    // send
    comm().send(rank_o, r_obuffer.str(), reqs[rank_ocount], _tag);
  }
  // receive buffers from various ranks sending to our ichannels
  for(unsigned int rank_icount = 0; rank_icount < _rank_ichannels.size(); ++rank_icount) {
    std::string r_ibuffer_str;
    int rank_i = comm().receive(Parallel::any_source, r_ibuffer_str, _tag).source(); // source rank
    Scatter::InBuffer r_ibuffer(r_ibuffer_str);
    // Iterate over rank_i's declared ichannels
    std::map<int,std::vector<int> >::const_iterator rank_ichannels_it = _rank_ichannels.find(rank_i);
    const std::vector<int>& r_ichannels = rank_ichannels_it->second;
    for (std::vector<int>::const_iterator r_ichannels_it = r_ichannels.begin(); r_ichannels_it != r_ichannels.end(); ++r_ichannels_it) {
      int r_ichannel = *r_ichannels_it;
      // read the number of r_ichannel's incoming sources
      int r_ic_sources_size;
      r_ibuffer.read(r_ic_sources_size);
      // Iterate over r_ichannel's incoming sources and unpack the data from the sent r_ichannel's sources.
      for (int r_ic_sources_count = 0; r_ic_sources_count < r_ic_sources_size; ++r_ic_sources_count) {
	int r_ic_source;
	r_ibuffer.read(r_ic_source);
	unpacker.unpack(r_ic_source,r_ichannel,r_ibuffer);
      }
    }
  }
  // wait for all response sends to finish
  // TODO: Can we do MPI_Waitall?
  for(unsigned int i = 0; i < reqs.size(); i++) {
    reqs[i].wait();
  }
  STOP_LOG("scatter", "ScatterDistributedMPI");
}

} // end namespace libMesh
