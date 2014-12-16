#ifndef LIBMESH_SCATTER
#define LIBMESH_SCATTER


// TODO
// 1. Implement ScatterAllgather.
// This one doesn't require a Rendezvous object to construct.
// What does it need?  How do we construct it?  What's the "direct" API?
// Note that it could be ANY Rendezvous that produces this scatter object,
// as long as something can filter out the received indices.
// RendezvousAllToAll can result in a ScatterAllgather.
// There could be an alternative implementation ScatterCrystal.

#include "libmesh/parallel_object.h"
#include <vector>

#include "libmesh/libmesh_common.h"

namespace libMesh {

  class Scatter
  {
  public:
    // TODO: reimplement OutBuffer/InBuffer to avoid copying strings.
    // Replace the buffer API with a stream API?
    class OutBuffer
    {
    public:
      template <typename T>
      void write(const T& t) {_ss.write((char*)&t,sizeof(t));}
      std::string& str() {_str = _ss.str(); return _str;}
    protected:
      std::ostringstream _ss;
      std::string _str;
    };
    class InBuffer
    {
    public:
      InBuffer(std::string& str) : _ss(str){};
      template <typename T>
      void read(T& t) {_ss.read((char*)&t,sizeof(t));}
    protected:
      std::istringstream _ss;
    };
    class Packer
    {
    public:
      virtual void prepack(int ochannel, int oc_source)                 = 0;
      virtual void pack(int ochannel,int oc_source,OutBuffer& buffer)   = 0;
    };
    class Unpacker
    {
    public:
      virtual void unpack(int ichannel,int ic_source,InBuffer& buffer)     = 0;
      // There is no postunpack, at least not until per-rank ichannel_sources are not known:
      // to enable the postunpack we would have to record the arrived r_ichannel sources,
      // something that we are trying to avoid.
    };

    Scatter() : _setup(false), _prepack(false) {};

    bool get_prepacking() {return _prepack;};

    void set_prepacking(bool prepack) {_prepack = prepack;};

    typedef std::map<int,std::vector<int> >::iterator       channel_sources_iterator;
    typedef std::map<int,std::vector<int> >::const_iterator const_channel_sources_iterator;

    void add_ochannel(int ochannel);

    void add_ochannel_sources(int ochannel,const std::vector<int>& sources);

    void add_ochannel_source(int ochannel,int source);

    const std::vector<int>& get_ochannel_sources(int ochannel) {return _ochannel_sources[ochannel];};

    const_channel_sources_iterator ochannel_sources_begin() {return _ochannel_sources.begin();};

    const_channel_sources_iterator ochannel_sources_end()   {return _ochannel_sources.end();};

    bool has_ochannel(int ochannel) {return _ochannel_sources.find(ochannel) != _ochannel_sources.end();};
    //

    void add_ichannel(int ochannel);

    void add_ichannel_sources(int ichannel,const std::vector<int>& sources);

    void add_ichannel_source(int ichannel,int source);

    const std::vector<int>& get_ichannel_sources(int ichannel) {return _ichannel_sources[ichannel];};

    const_channel_sources_iterator ichannel_sources_begin() {return _ichannel_sources.begin();};

    const_channel_sources_iterator ichannel_sources_end()   {return _ichannel_sources.end();};

    bool has_ichannel(int ichannel) {return _ichannel_sources.find(ichannel) != _ichannel_sources.end();};

    //
    virtual void setup(){_setup = true;};

    bool is_setup() const {return _setup;};

    virtual void scatter(Packer& /*packer*/, Unpacker& /*unpacker*/) const {};

  protected:
    bool _setup;

    bool _prepack;

    // outchannel -> (outgoing-sources)
    std::map<int,std::vector<int> >  _ochannel_sources;

    // inchannel -> (incoming-sources)
    std::map<int,std::vector<int> >  _ichannel_sources;
  };// class Scatter

  class ScatterDistributed : public Scatter
  {
  public:
    ScatterDistributed() : Scatter() {};
    virtual ~ScatterDistributed(){};
    // NOTE: adding a channel to a rank does not imply adding a channel_source array.
    // This is a low-level interface that does not guarantee this consistency,
    // which is supposed to be ensured by a higher-level rendezvous layer that
    // constructs the scatter.

    virtual void rank_add_ichannel(int rank,int inchannel) = 0;

    virtual void rank_add_ichannels(int rank,const std::vector<int>& inchannels) = 0;

    static const std::vector<int> NO_CHANNELS;

    // Returns an empty array, if no such rank.
    const std::vector<int>& rank_get_ichannels(int rank) {return __rank_get_channels(rank,_rank_ichannels);};

    // TODO: oranks_begin(), oranks_end()

    virtual void rank_add_ochannel(int rank,int outchanel) = 0;

    virtual void rank_add_ochannels(int rank,const std::vector<int>& outchanels) = 0;

    // Return an empty array, if no such rank.
    const std::vector<int>& rank_get_ochannels(int rank) {return __rank_get_channels(rank,_rank_ochannels);}

    // TODO: iranks_begin(), iranks_end()

    virtual void setup() = 0;

    virtual void scatter(Scatter::Packer& packer, Scatter::Unpacker& unpacker) const = 0;

  protected:
    std::map<int,std::vector<int> > _rank_ochannels; // rank -> (outchannels received by rank)
    std::map<int,std::vector<int> > _rank_ichannels; // rank -> (inchannels  originated by rank)
    const std::vector<int>& __rank_get_channels(int rank, const std::map<int,std::vector<int> >& channels);

    // Note:
    //   outchannels are the sender ranks' ids of the channels they established to receiver ranks.
    //   inchannels are the receiver ids of the channels from the senders.
    // outchannels correspond 1-1 to the inchannels by their position in the exchanged buffer.
    //   For example, suppose p sends to q and consider the set {c} of channels between them.
    // Channel c has local names cp and cq, respectively. the correspondence between cp and cq
    // is established by the ordering of _rank_ochannels[q] on p and the ordering of _rank_ichannels[p] on q,
    // which are required to have the same size. That this correspondence is correct must be ensured
    // by whatever Rendezvous process creates the scatter (e.g., manual "rendezvous").
    //   Alternatively, the outchannel names can be packed into the outbuffer, unpacked upon reciept
    // and matched to the corresponding inchannels by a mapping on the receiver that must be set up
    // by a Rendezvous. However, since bufferes are packed/unpacked sequentially, simply matching
    // up the ordering of channels in the buffer will suffice and results in less memory and communication
    // at scatter time (at the expense of more memory and communication at the Rendezvous time).
    //

  };// ScatterDistributed


  class ScatterDistributedMPI : public ScatterDistributed, ParallelObject
  {
  public:
    ScatterDistributedMPI(const Parallel::Communicator & comm) :
    ScatterDistributed(), ParallelObject(comm), _tag(comm.get_unique_tag(15382)) {};

    virtual void rank_add_ichannel(int rank,int inchannel);
    virtual void rank_add_ochannel(int rank,int ochanel);
    virtual void rank_add_ichannels(int rank,const std::vector<int>& ichannels);
    virtual void rank_add_ochannels(int rank,const std::vector<int>& ochannels);

    void setup();
    void scatter(Scatter::Packer& packer, Scatter::Unpacker& unpacker) const;
  protected:
    /**
     * Communication tag used for sending indices to another processor.
     */
    const Parallel::MessageTag _tag;
  };// class ScatterDistributedMPI
}

#endif //LIBMESH_SCATTER
