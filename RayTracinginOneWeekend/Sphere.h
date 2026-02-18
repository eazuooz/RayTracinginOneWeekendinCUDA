#pragma once
#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.h"

class Sphere : public Hittable
{
public:
    __device__ Sphere() {}

    __device__ Sphere(const Point3& center, double radius, Material* material)
        : mCenter(center)
        , mRadius(radius)
        , mMaterial(material)
    {
    }

    __device__ bool Hit(
        const Ray& ray,
        double tMin,
        double tMax,
        HitRecord& hitRecord) const override
    {
        Vector3 oc = ray.Origin() - mCenter;
        double a = Dot(ray.Direction(), ray.Direction());
        double b = Dot(oc, ray.Direction());
        double c = Dot(oc, oc) - mRadius * mRadius;
        double discriminant = b * b - a * c;

        if (discriminant > 0.0)
        {
            double temp = (-b - sqrt(discriminant)) / a;
            if (temp < tMax && temp > tMin)
            {
                hitRecord.T = temp;
                hitRecord.P = ray.At(hitRecord.T);
                Vector3 outwardNormal = (hitRecord.P - mCenter) / mRadius;
                hitRecord.SetFaceNormal(ray, outwardNormal);
                hitRecord.MaterialPtr = mMaterial;
                return true;
            }

            temp = (-b + sqrt(discriminant)) / a;
            if (temp < tMax && temp > tMin)
            {
                hitRecord.T = temp;
                hitRecord.P = ray.At(hitRecord.T);
                Vector3 outwardNormal = (hitRecord.P - mCenter) / mRadius;
                hitRecord.SetFaceNormal(ray, outwardNormal);
                hitRecord.MaterialPtr = mMaterial;
                return true;
            }
        }

        return false;
    }

private:
    Point3 mCenter;
    double mRadius;
    Material* mMaterial;
};

#endif
