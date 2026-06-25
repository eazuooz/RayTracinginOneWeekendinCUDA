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
        // 반지름 벡터로 중심 ± r 두 극점을 잡아 경계 상자를 만든다
        Vector3 rvec(radius, radius, radius);
        mBBox = Aabb(center - rvec, center + rvec);
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

    __device__ Aabb BoundingBox() const override { return mBBox; }

private:
    Point3 mCenter;
    double mRadius;
    Material* mMaterial;
    Aabb mBBox;
};

#endif
