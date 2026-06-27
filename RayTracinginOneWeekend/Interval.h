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

    // 두 구간을 빈틈없이 감싸는 구간을 생성한다 (AABB 합집합용)
    __host__ __device__ Interval(const Interval& a, const Interval& b)
        : Min(a.Min <= b.Min ? a.Min : b.Min)
        , Max(a.Max >= b.Max ? a.Max : b.Max)
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

    // 구간을 delta만큼 양쪽으로 넓힌다.
    // 두께가 0인 AABB(축에 완전히 평행한 면)에서 생기는 grazing/NaN 문제를
    // 막기 위해 약간의 패딩을 줄 때 사용한다.
    __host__ __device__ Interval Expand(double delta) const
    {
        double padding = delta / 2.0;
        return Interval(Min - padding, Max + padding);
    }
};

// === The Next Week Chapter 8: Instances ===
// 구간을 displacement만큼 평행이동한다. Translate 인스턴스가 자식의 AABB를
// 옮길 때, 각 축 구간을 이 연산으로 민다. (Aabb operator+ 가 축별로 호출)
__host__ __device__ inline Interval operator+(const Interval& ival, double displacement)
{
    return Interval(ival.Min + displacement, ival.Max + displacement);
}

__host__ __device__ inline Interval operator+(double displacement, const Interval& ival)
{
    return ival + displacement;
}

#endif
