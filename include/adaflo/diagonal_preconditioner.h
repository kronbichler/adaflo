// --------------------------------------------------------------------------
//
// Copyright (C) 2011 - 2016 by the adaflo authors
//
// This file is part of the adaflo library.
//
// The adaflo library is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.  The full text of the
// license can be found in the file LICENSE at the top level of the adaflo
// distribution.
//
// --------------------------------------------------------------------------

#ifndef __adaflo_diagonal_preconditioner_h
#define __adaflo_diagonal_preconditioner_h

#include <fstream>
#include <iostream>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>

using namespace dealii;


template <typename Number>
class DiagonalPreconditioner
{
public:
  DiagonalPreconditioner () {}

  DiagonalPreconditioner (const LinearAlgebra::distributed::Vector<Number> &diagonal_vector_in);
  DiagonalPreconditioner (const LinearAlgebra::distributed::BlockVector<Number> &diagonal_vector_in);
  void reinit (const LinearAlgebra::distributed::Vector<Number> &diagonal_vector_in);
  void reinit (const LinearAlgebra::distributed::BlockVector<Number> &diagonal_vector_in);
  void vmult (LinearAlgebra::distributed::Vector<Number> &dst,
              const LinearAlgebra::distributed::Vector<Number> &src) const;
  void vmult (LinearAlgebra::distributed::BlockVector<Number> &dst,
              const LinearAlgebra::distributed::BlockVector<Number> &src) const;
  const LinearAlgebra::distributed::Vector<Number> &get_inverse_vector() const
  {
    return inverse_diagonal_vector;
  }
  const LinearAlgebra::distributed::Vector<Number> &get_vector() const
  {
    return diagonal_vector;
  }

private:
  LinearAlgebra::distributed::Vector<Number>      diagonal_vector;
  LinearAlgebra::distributed::Vector<Number>      inverse_diagonal_vector;
  LinearAlgebra::distributed::BlockVector<Number> inverse_diagonal_block_vector;
};


#endif
