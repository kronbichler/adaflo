// --------------------------------------------------------------------------
//
// Copyright (C) 2009 - 2016 by the adaflo authors
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

#ifndef __dealii__timestep_control_h
#define __dealii__timestep_control_h

#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/vector_memory.h>

#include <cstdio>


using namespace dealii;

struct FlowParameters;

class TimeStepping : public Subscriptor
{
public:
  enum Scheme
  {
    implicit_euler,
    explicit_euler,
    crank_nicolson,
    bdf_2
  };

  TimeStepping(const FlowParameters &parameters);

  double
  start() const;
  double
  final() const;
  double
  tolerance() const;
  double
  step_size() const;
  double
  max_step_size() const;
  double
  old_step_size() const;
  double
  now() const;
  double
  previous() const;
  double
  tau1() const;
  double
  tau2() const;
  double
  weight() const;
  double
  max_weight_uniform() const;
  double
  weight_old() const;
  double
  weight_old_old() const;
  bool
  weight_has_changed() const;
  unsigned int
  step_no() const;

  // Returns whether the current time is close within Delta t to a given
  // tick_size, similar to the integer test "step_no % step_frequency ==
  // 0". If the time stepping is at the end of the global time range, this
  // method always returns true.
  bool
  at_tick(const double tick_size) const;

  // Extrapolates two values from to the previous time levels to the current
  // time level
  template <typename Number>
  Number
  extrapolate(const Number &old_value, const Number &old_old_value) const;

  // advances one step step and return the next time. If at final time, may
  // adjust the time step a bit to hit it exactly
  double
  next();

  void
  set_start_time(double);
  void
  set_final_time(double);
  void
  set_time_step(double);
  void
  set_tolerance(double);
  std::string
  name() const;
  Scheme
  scheme() const;

  void
  restart();
  bool
  at_end() const;

private:
  double       start_val;
  double       final_val;
  double       tolerance_val;
  Scheme       scheme_val;
  double       start_step_val;
  double       max_step_val;
  double       min_step_val;
  double       current_step_val;
  double       last_step_val;
  double       step_val;
  double       weight_val;
  double       weight_old_val;
  double       weight_old_old_val;
  double       factor_extrapol_old;
  double       factor_extrapol_old_old;
  unsigned int step_no_val;
  bool         at_end_val;
  bool         weight_changed;

  double now_val;
  double prev_val;
  double tau1_val;
  double tau2_val;
};


inline double
TimeStepping::start() const
{
  return start_val;
}


inline double
TimeStepping::final() const
{
  return final_val;
}



inline double
TimeStepping::step_size() const
{
  return current_step_val;
}



inline double
TimeStepping::max_step_size() const
{
  return max_step_val;
}


inline double
TimeStepping::old_step_size() const
{
  return last_step_val;
}

inline double
TimeStepping::tolerance() const
{
  return tolerance_val;
}


inline double
TimeStepping::now() const
{
  return now_val;
}

inline double
TimeStepping::previous() const
{
  return prev_val;
}

inline double
TimeStepping::tau1() const
{
  return tau1_val;
}

inline double
TimeStepping::tau2() const
{
  return tau2_val;
}

inline unsigned int
TimeStepping::step_no() const
{
  return step_no_val;
}

inline double
TimeStepping::weight() const
{
  return weight_val;
}

inline double
TimeStepping::max_weight_uniform() const
{
  if (scheme_val == bdf_2)
    return 1.5 / current_step_val;
  else
    return 1. / current_step_val;
}

inline double
TimeStepping::weight_old() const
{
  return weight_old_val;
}

inline double
TimeStepping::weight_old_old() const
{
  return weight_old_old_val;
}

inline bool
TimeStepping::weight_has_changed() const
{
  return weight_changed;
}

template <typename Number>
inline Number
TimeStepping::extrapolate(const Number &old, const Number &old_old) const
{
  return old * factor_extrapol_old + old_old * factor_extrapol_old_old;
}

inline void
TimeStepping::set_start_time(double t)
{
  start_val = t;
}


inline void
TimeStepping::set_final_time(double t)
{
  final_val = t;
}


inline void
TimeStepping::set_tolerance(double t)
{
  tolerance_val = t;
}


inline bool
TimeStepping::at_end() const
{
  return at_end_val;
}


#endif
