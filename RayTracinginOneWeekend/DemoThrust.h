#pragma once
#ifndef DEMO_THRUST_H
#define DEMO_THRUST_H

// Thin host wrappers around the Thrust parallel primitives used by MonteCarloDemo.cu.
//
// The Thrust/CUB headers are large. Keeping them in this one translation unit means
// the demo code (MonteCarloDemo.cu) never includes them and recompiles quickly; only
// plain functions that take raw device pointers cross the boundary.

// Sum of n doubles that live in device memory (thrust::reduce).
double DeviceSum(const double* dValues, int n);

// Finds the x at which the running sum of dValues (in increasing x order) first
// reaches halfSum. Sorts dXs/dValues in place (sort_by_key), turns dValues into a
// prefix sum (inclusive_scan) and binary-searches it (lower_bound).
double DeviceHalfwayPoint(double* dXs, double* dValues, int n, double halfSum);

#endif
