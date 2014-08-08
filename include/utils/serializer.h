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


#ifndef LIBMESH_SERIALIZER_H
#define LIBMESH_SERIALIZER_H

// C++ Includes   -----------------------------------
#include <istream>
#include <ostream>

namespace libMesh {

/**
 * This is the \p Serializer class.  It is used to read/write an object
 * from/to a stream so that the object can be sent between processors.
 * This is a pure virtual class.
 *
 * \author  Matthew D. Michelotti
 */

template <class T>
class Serializer {

public:
  /**
   * Reads an object from \p stream and sets it in \p buffer.
   * NOTE: implementations of this method should probably use the
   * istream::read method instead of the >> operator.
   */
  virtual void read(std::istream& stream, T& buffer) = 0;

  /**
   * Writes the object \p buffer to \p stream.
   * NOTE: implementations of this method should probably use the
   * ostream::write method instead of the << operator.
   */
  virtual void write(std::ostream& stream, const T& buffer) = 0;

};

} // namespace libMesh

#endif // LIBMESH_SERIALIZER_H
