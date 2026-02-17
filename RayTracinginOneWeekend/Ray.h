#pragma once
#ifndef RAY_H
#define RAY_H

#include "Vec3.h"

class Ray
{
public:
    __device__ Ray() {}

    __device__ Ray(const Point3& origin, const Vector3& direction)
        : mOrig(origin), mDir(direction) {}

    __device__ const Point3& Origin() const { return mOrig; }
    __device__ const Vector3& Direction() const { return mDir; }

    __device__ Point3 At(double t) const
    {
        return mOrig + t * mDir;
    }

private:
    Point3 mOrig;
    Vector3 mDir;
};

#endif
