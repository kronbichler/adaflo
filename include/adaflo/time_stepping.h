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


namespace adaflo
{
  using namespace dealii;


  struct FlowParameters;

  struct TimeSteppingParameters
  {
    /**
     * TODO
     */
    enum class Scheme
    {
      implicit_euler,
      explicit_euler,
      crank_nicolson,
      bdf_2
    };
    Scheme time_step_scheme;
    double start_time;
    double end_time;
    double time_step_size_start;
    double time_stepping_cfl;
    double time_stepping_coef2;
    double time_step_tolerance;
    double time_step_size_max;
    double time_step_size_min;
  };


  class TimeStepping : public Subscriptor
  {
  public:
    TimeStepping(const FlowParameters &parameters);

    TimeStepping(const TimeSteppingParameters &parameters);

    double
    start() const;
    double
    final() const;
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
    set_desired_time_step(double);
    void
    set_time_step(double);
    std::string
    name() const;
    TimeSteppingParameters::Scheme
    scheme() const;

    void
    restart();
    bool
    at_end() const;

  private:
    double start_val; // [m] start time; may be modified by set_start_time @todo:
    // modification is nowhere used, keep?
    double
      final_val; // [m] end time; may be modified by set_final_time @todo: modification
    // is nowhere used, keep?
    const TimeSteppingParameters::Scheme scheme_val; // [i] time integration scheme
    const double start_step_val;   // [i] initial value of the time increment
    const double max_step_val;     // [i] maximum value of the time increment
    const double min_step_val;     // [i] minimum value of the time increment
    double       current_step_val; // [m] current value of the time increment;
    //     initialized in the constructor by start_step_val
    //     can be modified in set_desired_time_step(desired_value)
    //     fulfilling the criteria
    //         - 0.5 * step_size_prev <= current_step_val <=
    //         2*step_size_prev
    //         - min_step_val <= current_step_val <= max_step_val
    double last_step_val; // [m] constructor and restart() sets this parameter to zero.
    //     next() sets this parameter equal to current_step_val
    //     (corresponding to the previous time increment)
    double step_val; // [m] current value of the time increment; m]
    //     - initialized in the constructor by start_step_val
    //     - changed in set_desired_time_step to be equal to current_step_val
    //     @todo - ambiguous with current_step_val (?)
    double weight_val; // [m] 1/time_increment
    //     - constructor: 1/start_step_val;
    //     - next():      1/current_step_val
    double weight_old_val; // [m] old time increment;
    //     - BDF2: this parameter is used for the time integration
    //     - else this parameter is used to determine weight_changed
    double weight_old_old_val;  // [m] old old time increment; only used in case of BDF2
    double factor_extrapol_old; // [m] extrapolation factor determined between the current
    // and the last
    //     value of the increment
    double factor_extrapol_old_old; // [m] extrapolation factor determined by the ratio
    // between the current and
    //     and the old value of the time increment
    unsigned int step_no_val; // [m] - constructor/restart(): initialize to zero
    //     - incremented by 1 in next()
    bool at_end_val;     // [m] determines if the end time is reached
    bool weight_changed; // [m] determines if the integration weight has changed; this
    // parameter is never reused
    //     this is used in navier_stokes.cc or phase_field.cc
    double now_val; // [m] current time; time to be reached after time integration (t_n)
    //     - constructor/restart(): initialize to start_val
    //     - calculated as return argument from next()
    double prev_val; // [m] old time; time at the begin of the integration step
    double
      tau1_val; // [i] integration weight for multiplication with the current function
    // value, i.e. f(t_n)
    //     - constructor: parameter depends on the scheme_val
    double tau2_val; // [i] integration weight for multiplication with the old function
    // value, i.e. f(t_n-1)
    //     - constructor: parameter depends on the scheme_val
  };


  /**
   * Getter functions
   */
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
    if (scheme_val == TimeSteppingParameters::Scheme::bdf_2)
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

  inline bool
  TimeStepping::at_end() const
  {
    return at_end_val;
  }
} // namespace adaflo

#endif
