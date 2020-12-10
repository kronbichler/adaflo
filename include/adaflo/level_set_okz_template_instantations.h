// --------------------------------------------------------------------------
//
// Copyright (C) 2020 by the adaflo authors
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


#ifndef __adaflo_level_set_okz_template_instantiations_h
#define __adaflo_level_set_okz_template_instantiations_h

#define EXPAND_OPERATIONS(OPERATION)                                                      \
  const unsigned int degree_u  = this->navier_stokes.get_dof_handler_u().get_fe().degree; \
  const unsigned int ls_degree = this->parameters.concentration_subdivisions;             \
                                                                                          \
  AssertThrow(degree_u >= 2 && degree_u <= 5, ExcNotImplemented());                       \
  AssertThrow(ls_degree >= 1 && ls_degree <= 4, ExcNotImplemented());                     \
  if (ls_degree == 1)                                                                     \
    {                                                                                     \
      if (degree_u == 2)                                                                  \
        OPERATION(1, 2);                                                                  \
      else if (degree_u == 3)                                                             \
        OPERATION(1, 3);                                                                  \
      else if (degree_u == 4)                                                             \
        OPERATION(1, 4);                                                                  \
      else if (degree_u == 5)                                                             \
        OPERATION(1, 5);                                                                  \
    }                                                                                     \
  else if (ls_degree == 2)                                                                \
    {                                                                                     \
      if (degree_u == 2)                                                                  \
        OPERATION(2, 2);                                                                  \
      else if (degree_u == 3)                                                             \
        OPERATION(2, 3);                                                                  \
      else if (degree_u == 4)                                                             \
        OPERATION(2, 4);                                                                  \
      else if (degree_u == 5)                                                             \
        OPERATION(2, 5);                                                                  \
    }                                                                                     \
  else if (ls_degree == 3)                                                                \
    {                                                                                     \
      if (degree_u == 2)                                                                  \
        OPERATION(3, 2);                                                                  \
      else if (degree_u == 3)                                                             \
        OPERATION(3, 3);                                                                  \
      else if (degree_u == 4)                                                             \
        OPERATION(3, 4);                                                                  \
      else if (degree_u == 5)                                                             \
        OPERATION(3, 5);                                                                  \
    }                                                                                     \
  else if (ls_degree == 4)                                                                \
    {                                                                                     \
      if (degree_u == 2)                                                                  \
        OPERATION(4, 2);                                                                  \
      else if (degree_u == 3)                                                             \
        OPERATION(4, 3);                                                                  \
      else if (degree_u == 4)                                                             \
        OPERATION(4, 4);                                                                  \
      else if (degree_u == 5)                                                             \
        OPERATION(4, 5);                                                                  \
    }

#endif
