// --------------------------------------------------------------------------
//
// Copyright (C) 2015 - 2016 by the adaflo authors
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

#ifndef __adaflo_block_matrix_extension_h
#define __adaflo_block_matrix_extension_h


#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>


namespace adaflo
{
  using namespace dealii;

  class BlockMatrixExtension : public TrilinosWrappers::SparseMatrix
  {
  public:
    using TrilinosWrappers::SparseMatrix::vmult;

    void
    vmult(LinearAlgebra::distributed::BlockVector<double>       &dst,
          const LinearAlgebra::distributed::BlockVector<double> &src) const
    {
      const Epetra_CrsMatrix &matrix = this->trilinos_matrix();
      AssertDimension(size_type(dst.block(0).end() - dst.block(0).begin()),
                      static_cast<size_type>(matrix.RangeMap().NumMyPoints()));
      AssertDimension(size_type(src.block(0).end() - src.block(0).begin()),
                      static_cast<size_type>(matrix.DomainMap().NumMyPoints()));
      AssertDimension(dst.n_blocks(), src.n_blocks());
      Assert(dst.n_blocks() < 16, ExcNotImplemented());
      double *dst_ptrs[16], *src_ptrs[16];
      for (unsigned int i = 0; i < dst.n_blocks(); ++i)
        {
          dst_ptrs[i] = dst.block(i).begin();
          src_ptrs[i] = const_cast<double *>(src.block(i).begin());
        }

      Epetra_MultiVector tril_dst(View, matrix.RangeMap(), dst_ptrs, dst.n_blocks());
      Epetra_MultiVector tril_src(View, matrix.DomainMap(), src_ptrs, src.n_blocks());

      const int ierr = matrix.Multiply(false, tril_src, tril_dst);
      Assert(ierr == 0, ExcTrilinosError(ierr));
      (void)ierr; // removes -Wunused-variable in optimized mode
    }
  };


  class BlockILUExtension : public TrilinosWrappers::PreconditionILU
  {
  public:
    using TrilinosWrappers::PreconditionBase::vmult;

    void
    vmult(LinearAlgebra::distributed::BlockVector<double>       &dst,
          const LinearAlgebra::distributed::BlockVector<double> &src) const
    {
      Assert(preconditioner.get() != 0, ExcNotInitialized());
      const Epetra_Operator &prec = *this->preconditioner;
      AssertDimension(size_type(dst.block(0).end() - dst.block(0).begin()),
                      static_cast<size_type>(prec.OperatorRangeMap().NumMyPoints()));
      AssertDimension(size_type(src.block(0).end() - src.block(0).begin()),
                      static_cast<size_type>(prec.OperatorDomainMap().NumMyPoints()));
      AssertDimension(dst.n_blocks(), src.n_blocks());
      AssertThrow(dst.n_blocks() < 16, ExcNotImplemented());
      double *dst_ptrs[16], *src_ptrs[16];
      for (unsigned int i = 0; i < dst.n_blocks(); ++i)
        {
          dst_ptrs[i] = dst.block(i).begin();
          src_ptrs[i] = const_cast<double *>(src.block(i).begin());
        }

      Epetra_MultiVector tril_dst(View,
                                  prec.OperatorRangeMap(),
                                  dst_ptrs,
                                  dst.n_blocks());
      Epetra_MultiVector tril_src(View,
                                  prec.OperatorDomainMap(),
                                  src_ptrs,
                                  src.n_blocks());

      const int ierr = prec.ApplyInverse(tril_src, tril_dst);
      Assert(ierr == 0, ExcTrilinosError(ierr));
      (void)ierr; // removes -Wunused-variable in optimized mode
    }
  };



  template <int dim>
  class ComponentILUExtension : public TrilinosWrappers::PreconditionILU
  {
  public:
    void
    initialize(const TrilinosWrappers::SparseMatrix                    &matrix,
               const TrilinosWrappers::PreconditionILU::AdditionalData &data,
               const std::vector<unsigned int> &index_by_component)
    {
      this->dealii::TrilinosWrappers::PreconditionILU::initialize(matrix, data);
      this->index_by_component = index_by_component;
      src_cpy =
        std::make_unique<Epetra_MultiVector>(matrix.trilinos_matrix().DomainMap(), dim);
      dst_cpy =
        std::make_unique<Epetra_MultiVector>(matrix.trilinos_matrix().RangeMap(), dim);
    }

    // Application to a vector src, stored in dst (do not call the method vmult
    // in order to avoid overloading a virtual function in deal.II)
    void
    multiply(LinearAlgebra::distributed::Vector<double>       &dst,
             const LinearAlgebra::distributed::Vector<double> &src) const
    {
      Assert(preconditioner.get() != 0, ExcNotInitialized());
      const Epetra_Operator &prec       = *this->preconditioner;
      const unsigned int     local_size = index_by_component.size() / dim;
      AssertDimension((int)local_size, src_cpy->MyLength());
      AssertDimension(size_type(dst.end() - dst.begin()),
                      static_cast<size_type>(prec.OperatorRangeMap().NumMyPoints()) *
                        dim);
      AssertDimension(size_type(src.end() - src.begin()),
                      static_cast<size_type>(prec.OperatorDomainMap().NumMyPoints()) *
                        dim);

      for (unsigned int i = 0; i < local_size; ++i)
        for (unsigned int d = 0; d < dim; ++d)
          (*src_cpy)[d][i] = src.local_element(index_by_component[i * dim + d]);

      const int ierr = prec.ApplyInverse(*src_cpy, *dst_cpy);
      Assert(ierr == 0, ExcTrilinosError(ierr));
      (void)ierr;

      for (unsigned int i = 0; i < local_size; ++i)
        for (unsigned int d = 0; d < dim; ++d)
          dst.local_element(index_by_component[i * dim + d]) = (*dst_cpy)[d][i];
    }

  private:
    std::vector<unsigned int>                   index_by_component;
    mutable std::unique_ptr<Epetra_MultiVector> src_cpy, dst_cpy;
  };
} // namespace adaflo

#endif
