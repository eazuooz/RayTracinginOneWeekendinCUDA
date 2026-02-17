#pragma once
#ifndef INTERVAL_H
#define INTERVAL_H

#include "cuda_runtime.h"
#include <cfloat>

struct Interval
{
    double Min;
    double Max;

    __host__ __device__ Interval()
        : Min(+DBL_MAX)
        , Max(-DBL_MAX)
    {
    }

    __host__ __device__ Interval(double minimum, double maximum)
        : Min(minimum)
        , Max(maximum)
    {
    }

    __host__ __device__ double Size() const
    {
        return Max - Min;
    }

    __host__ __device__ bool Contains(double value) const
    {
        return Min <= value && value <= Max;
    }

    __host__ __device__ bool Surrounds(double value) const
    {
        return Min < value && value < Max;
    }

    __host__ __device__ double Clamp(double value) const
    {
        if (value < Min) return Min;
        if (value > Max) return Max;
        return value;
    }
};

#endif
