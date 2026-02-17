#pragma once
#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.h"

class Sphere : public Hittable
{
public:
    __device__ Sphere() {}
    __device__ Sphere(const Point3& center, double radius)
        : mCenter(center), mRadius(radius) {}

    __device__ bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& hitRecord) const override
    {
        Vector3 oc = ray.Origin() - mCenter;
        double a = Dot(ray.Direction(), ray.Direction());
        double b = Dot(oc, ray.Direction());
        double c = Dot(oc, oc) - mRadius * mRadius;
        double discriminant = b * b - a * c;

        if (discriminant > 0.0)
        {
            // 가까운 교차점 먼저 확인
            double temp = (-b - sqrt(discriminant)) / a;
            if (temp < tMax && temp > tMin)
            {
                hitRecord.T = temp;
                hitRecord.P = ray.At(hitRecord.T);
                hitRecord.Normal = (hitRecord.P - mCenter) / mRadius;
                return true;
            }

            // 먼 교차점 확인
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < tMax && temp > tMin)
            {
                hitRecord.T = temp;
                hitRecord.P = ray.At(hitRecord.T);
                hitRecord.Normal = (hitRecord.P - mCenter) / mRadius;
                return true;
            }
        }

        return false;
    }

private:
    Point3 mCenter;
    double mRadius;
};

#endif
