#pragma once
#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.h"
#include "Onb.h"
#include "Sampling.h"

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

    // === The Rest of Your Life Chapter 12: 구를 향한 샘플링 ===
    //
    // 바깥의 한 점에서 구를 보면 구는 원뿔 모양의 입체각을 차지한다. 그 원뿔 안에서
    // 균일하게 뽑으므로 밀도는 1 / (원뿔의 입체각)이다.
    //   입체각 = 2 pi (1 - cos(theta_max)),   sin(theta_max) = R / 거리
    // (구 표면에서 아무 점이나 고르면 뒤쪽 면을 고를 수 있어 못 쓴다. 보이는 쪽만
    //  균일하게 덮는 것이 이 원뿔 샘플링이다.)
    __device__ double PdfValue(
        const Point3& origin, const Vector3& direction, curandState* randState) const override
    {
        // 정지한 구에만 유효하다(MovingSphere는 샘플링 대상이 아니다).
        HitRecord rec;
        if (!this->Hit(Ray(origin, direction), 0.001, DBL_MAX, rec, randState))
            return 0.0;

        double distanceSquared = (mCenter - origin).LengthSquared();
        if (distanceSquared <= mRadius * mRadius)
            return 0.0;   // 구 안에서는 이 공식이 성립하지 않는다

        double cosThetaMax = sqrt(1.0 - mRadius * mRadius / distanceSquared);
        double solidAngle = 2.0 * kPi * (1.0 - cosThetaMax);

        return 1.0 / solidAngle;
    }

    __device__ Vector3 Random(const Point3& origin, curandState* randState) const override
    {
        Vector3 direction = mCenter - origin;
        double distanceSquared = direction.LengthSquared();

        // z축이 구 중심 방향인 좌표계에서 뽑아 ONB로 돌린다.
        Onb uvw(direction);
        return uvw.Transform(RandomToSphere(mRadius, distanceSquared, randState));
    }

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
