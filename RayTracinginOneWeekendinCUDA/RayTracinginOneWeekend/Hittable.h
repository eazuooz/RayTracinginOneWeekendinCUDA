#pragma once
#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.h"

class HitRecord
{
public:
    Point3 P;
    Vec3 Normal;
    double T;
};

class Hittable
{
public:
    virtual ~Hittable() = default;

    virtual bool Hit(const Ray& r, double rayTMin, double rayTMax, HitRecord& rec) const = 0;
};

#endif