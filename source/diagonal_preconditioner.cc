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

#include <adaflo/diagonal_preconditioner.h>


using namespace dealii;



template <typename Number>
DiagonalPreconditioner<Number>::DiagonalPreconditioner(
  const LinearAlgebra::distributed::Vector<Number> &diagonal_vector_in)
{
  reinit(diagonal_vector_in);
}



template <typename Number>
DiagonalPreconditioner<Number>::DiagonalPreconditioner(
  const LinearAlgebra::distributed::BlockVector<Number> &diagonal_vector_in)
{
  reinit(diagonal_vector_in);
}



template <typename Number>
void
DiagonalPreconditioner<Number>::reinit(
  const LinearAlgebra::distributed::Vector<Number> &diagonal_vector_in)
{
  inverse_diagonal_block_vector.reinit(0);
  diagonal_vector = diagonal_vector_in;
  inverse_diagonal_vector.reinit(diagonal_vector, true);
  const double linfty_norm = diagonal_vector.linfty_norm();
  for (unsigned int i = 0; i < diagonal_vector.locally_owned_size(); ++i)
    if (std::abs(diagonal_vector.local_element(i)) > 1e-10 * linfty_norm)
      inverse_diagonal_vector.local_element(i) = 1. / diagonal_vector.local_element(i);
    else
      inverse_diagonal_vector.local_element(i) = 1.;
  diagonal_vector.zero_out_ghost_values();
  inverse_diagonal_vector.zero_out_ghost_values();
}



template <typename Number>
void
DiagonalPreconditioner<Number>::reinit(
  const LinearAlgebra::distributed::BlockVector<Number> &diagonal_vector_in)
{
  diagonal_vector.reinit(0);
  inverse_diagonal_vector.reinit(0);
  inverse_diagonal_block_vector = diagonal_vector_in;
  const double linfty_norm      = inverse_diagonal_block_vector.linfty_norm();
  for (unsigned int bl = 0; bl < inverse_diagonal_block_vector.n_blocks(); ++bl)
    for (unsigned int i = 0;
         i < inverse_diagonal_block_vector.block(bl).locally_owned_size();
         ++i)
      if (std::abs(inverse_diagonal_block_vector.block(bl).local_element(i)) >
          1e-10 * linfty_norm)
        inverse_diagonal_block_vector.block(bl).local_element(i) =
          1. / inverse_diagonal_block_vector.block(bl).local_element(i);
      else
        inverse_diagonal_block_vector.block(bl).local_element(i) = 1.;
  inverse_diagonal_block_vector.zero_out_ghost_values();
}



template <typename Number>
void
DiagonalPreconditioner<Number>::vmult(
  LinearAlgebra::distributed::Vector<Number> &      dst,
  const LinearAlgebra::distributed::Vector<Number> &src) const
{
  AssertDimension(diagonal_vector.size(), src.size());
  AssertDimension(inverse_diagonal_block_vector.size(), 0);
  for (unsigned int i = 0; i < inverse_diagonal_vector.locally_owned_size(); ++i)
    {
      dst.local_element(i) =
        src.local_element(i) * inverse_diagonal_vector.local_element(i);
    }
}



template <typename Number>
void
DiagonalPreconditioner<Number>::vmult(
  LinearAlgebra::distributed::BlockVector<Number> &      dst,
  const LinearAlgebra::distributed::BlockVector<Number> &src) const
{
  if (inverse_diagonal_block_vector.n_blocks() > 0)
    {
      AssertDimension(inverse_diagonal_block_vector.size(), src.size());
      AssertDimension(inverse_diagonal_vector.size(), 0);
      dst = src;
      dst.scale(inverse_diagonal_block_vector);
    }
  else
    {
      for (unsigned int block = 0; block < src.n_blocks(); ++block)
        AssertDimension(inverse_diagonal_vector.size(), dst.block(block).size());
      const unsigned int n_blocks = src.n_blocks();
      for (unsigned int i = 0; i < inverse_diagonal_vector.locally_owned_size(); ++i)
        for (unsigned int block = 0; block < n_blocks; ++block)
          {
            dst.block(block).local_element(i) = src.block(block).local_element(i) *
                                                inverse_diagonal_vector.local_element(i);
          }
    }
}



template class DiagonalPreconditioner<double>;
template class DiagonalPreconditioner<float>;
