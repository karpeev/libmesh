
#include <string>
#include <vector>

#include "libmesh/parallel_object.h"
#include "libmesh/parallel_printer.h"
// Should be incorporated into ParallelObject?

using namespace libMesh;

void ParallelPrinter::gather_print(const std::ostringstream& sout, std::ostream& os) const {
  libmesh_parallel_only(_comm);
  std::string text_str = sout.str();
  std::vector<char> text(text_str.begin(), text_str.end());
  text.push_back('\0');
  _comm.gather(0, text);
  if (!_comm.rank()) {
    int ci = 0;
    while(ci < (int)text.size()) {
      os << &text[ci] << std::endl;
      while(text[ci++] != '\0');
    }
  }
}
