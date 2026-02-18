#pragma once
#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.h"

class Material;

struct HitRecord
{
    Point3 P;
    Vector3 Normal;
    double T;
    Material* MaterialPtr;
};

class Hittable
{
public:
    __device__ virtual bool Hit(
        const Ray& ray,
        double tMin,
        double tMax,
        HitRecord& hitRecord) const = 0;
};

#endif
