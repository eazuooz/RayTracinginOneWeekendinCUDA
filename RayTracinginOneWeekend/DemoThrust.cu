// Thrust wrappers for the Monte Carlo demos.
// ASCII-only comments on purpose: see the note at the top of DemoThrust.h.

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

#include "DemoThrust.h"

double DeviceSum(const double* dValues, int n)
{
	thrust::device_ptr<const double> values(dValues);
	return thrust::reduce(values, values + n, 0.0, thrust::plus<double>());
}

double DeviceHalfwayPoint(double* dXs, double* dValues, int n, double halfSum)
{
	thrust::device_ptr<double> xs(dXs);
	thrust::device_ptr<double> values(dValues);

	// 1) sort the samples by x, carrying p(x) along as the value
	thrust::sort_by_key(xs, xs + n, values);

	// 2) running sum of p(x) in increasing x order (prefix sum, in place)
	thrust::inclusive_scan(values, values + n, values);

	// 3) first position whose running sum reaches half of the total (binary search)
	int idx = int(thrust::lower_bound(values, values + n, halfSum) - values);
	if (idx >= n)
		idx = n - 1;

	return xs[idx];   // reading one element = one small device-to-host copy
}
