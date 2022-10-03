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

#include <deal.II/base/parameter_handler.h>

#include <adaflo/parameters.h>
#include <adaflo/time_stepping.h>

TimeStepping::TimeStepping(const FlowParameters &parameters)
  : start_val(parameters.start_time)
  , final_val(parameters.end_time)
  , scheme_val(parameters.time_step_scheme)
  , start_step_val(parameters.time_step_size_start)
  , max_step_val(parameters.time_step_size_max)
  , min_step_val(parameters.time_step_size_min)
  , current_step_val(start_step_val)
  , last_step_val(0.)
  , step_val(start_step_val)
  , weight_val(1. / start_step_val)
  , weight_old_val(-1.)
  , weight_old_old_val(0.)
  , factor_extrapol_old(0.)
  , factor_extrapol_old_old(0.)
  , step_no_val(0)
  , at_end_val(false)
  , weight_changed(true)
{
  now_val  = start_val;
  prev_val = start_val;
  if (scheme_val == TimeSteppingParameters::Scheme::implicit_euler)
    {
      tau1_val = 1.;
      tau2_val = 0.;
    }
  else if (scheme_val == TimeSteppingParameters::Scheme::explicit_euler)
    {
      tau1_val = 0.;
      tau2_val = 1.;
    }
  else if (scheme_val == TimeSteppingParameters::Scheme::crank_nicolson)
    tau1_val = tau2_val = .5;
  else if (scheme_val == TimeSteppingParameters::Scheme::bdf_2)
    {
      tau1_val = 1.;
      tau2_val = 0.;
    }
}

TimeStepping::TimeStepping(const TimeSteppingParameters &parameters)
  : start_val(parameters.start_time)
  , final_val(parameters.end_time)
  , scheme_val(parameters.time_step_scheme)
  , start_step_val(parameters.time_step_size_start)
  , max_step_val(parameters.time_step_size_max)
  , min_step_val(parameters.time_step_size_min)
  , current_step_val(start_step_val)
  , last_step_val(0.)
  , step_val(start_step_val)
  , weight_val(1. / start_step_val)
  , weight_old_val(-1.)
  , weight_old_old_val(0.)
  , factor_extrapol_old(0.)
  , factor_extrapol_old_old(0.)
  , step_no_val(0)
  , at_end_val(false)
  , weight_changed(true)
{
  now_val  = start_val;
  prev_val = start_val;
  if (scheme_val == TimeSteppingParameters::Scheme::implicit_euler)
    {
      tau1_val = 1.;
      tau2_val = 0.;
    }
  else if (scheme_val == TimeSteppingParameters::Scheme::explicit_euler)
    {
      tau1_val = 0.;
      tau2_val = 1.;
    }
  else if (scheme_val == TimeSteppingParameters::Scheme::crank_nicolson)
    tau1_val = tau2_val = .5;
  else if (scheme_val == TimeSteppingParameters::Scheme::bdf_2)
    {
      tau1_val = 1.;
      tau2_val = 0.;
    }
}

void
TimeStepping::restart()
{
  step_no_val      = 0;
  now_val          = start_val;
  step_val         = start_step_val;
  current_step_val = step_val;
  last_step_val    = 0;

  if ((final_val - start_val) / start_step_val < 1e-14)
    at_end_val = true;
  else
    at_end_val = false;

  weight_changed = true;
}



double
TimeStepping::next()
{
  Assert(at_end_val == false, ExcMessage("Final time already reached, cannot proceed"));
  double s = current_step_val;

  // Do time step control, but not in
  // first step.
  if (now_val != start())
    {
      last_step_val = current_step_val;
      if (scheme_val == TimeSteppingParameters::Scheme::bdf_2 && step_no_val == 1)
        s = step_val;

      if (s > max_step_val)
        s = max_step_val;
    }

  // Try incrementing time by s
  double h         = now_val + s;
  current_step_val = s;

  // If we just missed the final time, increase
  // the step size a bit. This way, we avoid a
  // very small final step. If the step shot
  // over the final time, adjust it so we hit
  // the final time exactly.
  double s1 = .01 * s;
  if (!at_end_val && h > final_val - s1)
    {
      current_step_val = final_val - now_val;
      h                = final_val;
      at_end_val       = true;
    }

  {
    double new_weight;
    if (scheme_val == TimeSteppingParameters::Scheme::bdf_2 && now_val != start())
      {
        new_weight = ((2. * current_step_val + last_step_val) /
                      (current_step_val * (current_step_val + last_step_val)));
        weight_old_val =
          -((current_step_val + last_step_val) / (current_step_val * last_step_val));
        weight_old_old_val =
          current_step_val / (last_step_val * (current_step_val + last_step_val));
      }
    else
      {
        new_weight     = 1. / current_step_val;
        weight_old_val = -1. / current_step_val;
      }
    if (std::fabs(new_weight - weight_val) / new_weight > 1e-12)
      {
        weight_val     = new_weight;
        weight_changed = true;
      }
    else
      weight_changed = false;

    // compute weights for extrapolation. Do not extrapolate in second time
    // step because initial condition might not have been consistent
    if (step_no_val > 1)
      {
        factor_extrapol_old     = (current_step_val + last_step_val) / last_step_val;
        factor_extrapol_old_old = -current_step_val / last_step_val;
      }
    else
      {
        factor_extrapol_old     = 1.;
        factor_extrapol_old_old = 0.;
      }
  }

  prev_val = now_val;
  now_val  = h;
  step_no_val++;
  return now_val;
}



std::string
TimeStepping::name() const
{
  std::string result;
  if (scheme_val == TimeSteppingParameters::Scheme::implicit_euler)
    result = std::string("ImplEuler");
  else if (scheme_val == TimeSteppingParameters::Scheme::explicit_euler)
    result = std::string("ExplEuler");
  else if (scheme_val == TimeSteppingParameters::Scheme::crank_nicolson)
    result = std::string("CrankNicolson");
  else if (scheme_val == TimeSteppingParameters::Scheme::bdf_2)
    result = std::string("BDF-2");
  return result;
}


TimeSteppingParameters::Scheme
TimeStepping::scheme() const
{
  return scheme_val;
}



bool
TimeStepping::at_tick(const double tick) const
{
  const double time     = now();
  const int    position = int(time * 1.0000000001 / tick);
  const double slot     = position * tick;
  if (((time - slot) > (step_size() * 0.95)) && !at_end())
    return false;
  else
    return true;
}



void
TimeStepping::set_time_step(const double value)
{
  current_step_val = value;
  step_val         = current_step_val;
}



void
TimeStepping::set_desired_time_step(const double desired_value)
{
  // We take into account the first iteration regarding to the
  // previous used time step
  double step_size_prev = now() == 0 ? desired_value : step_size();

  // When setting a new time step size one needs to consider three things:
  //  - That it is not smaller than the minimum given
  //  - That it is not larger than the maximum step size given
  //  - That the change from the previous value is not too big, which should
  //    be fulfilled automatically in this case because we look at quantities
  //    that vary slowly.
  current_step_val =
    std::min(2 * step_size_prev, std::max(desired_value, 0.5 * step_size_prev));
  current_step_val = std::min(max_step_val, std::max(min_step_val, current_step_val));

  step_val = current_step_val;
}
