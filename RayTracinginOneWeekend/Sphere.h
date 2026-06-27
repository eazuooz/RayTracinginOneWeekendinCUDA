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
        HitRecord& hitRecord,
        curandState* randState) const override
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
                GetSphereUV(outwardNormal, hitRecord.U, hitRecord.V);
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
                GetSphereUV(outwardNormal, hitRecord.U, hitRecord.V);
                hitRecord.MaterialPtr = mMaterial;
                return true;
            }
        }

        return false;
    }

    __device__ Aabb BoundingBox() const override { return mBBox; }

    // 원점 중심 단위 구 위의 점 p에 대한 (u,v) 텍스처 좌표를 구한다.
    //  u: Y축을 도는 각(경도). X=-1에서 0, 한 바퀴 돌아 1.
    //  v: Y=-1(바닥)에서 0, Y=+1(꼭대기)에서 1 (위도).
    // 구면 좌표 (theta, phi)를 통해 계산한다:
    //   theta = acos(-y)            (바닥 극에서 위로 잰 각)
    //   phi   = atan2(-z, x) + Pi   (Y축 둘레의 각, 0~2Pi 연속이 되도록 +Pi)
    //   u = phi / (2*Pi),  v = theta / Pi
    __device__ static void GetSphereUV(const Point3& p, double& u, double& v)
    {
        const double pi = 3.1415926535897932385;
        double theta = acos(-p.Y());
        double phi = atan2(-p.Z(), p.X()) + pi;
        u = phi / (2.0 * pi);
        v = theta / pi;
    }

private:
    Point3 mCenter;
    double mRadius;
    Material* mMaterial;
    Aabb mBBox;
};

#endif
