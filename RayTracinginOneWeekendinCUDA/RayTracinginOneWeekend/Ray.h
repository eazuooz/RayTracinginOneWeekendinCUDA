#pragma once
#ifndef RAY_H
#define RAY_H

#include "Vec3.h"

class Ray
{
public:
    Ray() {}

    Ray(const Point3& origin, const Vector3& direction) 
        : mOrig(origin), mDir(direction) {}

    const Point3& Origin() const { return mOrig; }
    const Vector3& Direction() const { return mDir; }

    Point3 At(double t) const
    {
        return mOrig + t * mDir;
    }

private:
    Point3 mOrig;
    Vector3 mDir;
};

#endif