#pragma once
#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.h"
#include "Interval.h"

struct HitRecord
{
    Point3 P;
    Vector3 Normal;
    double T;
};

class Hittable
{
public:
    __device__ virtual bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& hitRecord) const = 0;
};

#endif
