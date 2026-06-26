#pragma once
#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "Hittable.h"

// === The Next Week Chapter 2: 모션 블러 (Motion Blur) ===
//
// 셔터 개방 구간(time0 ~ time1) 동안 center0에서 center1으로 선형 이동하는 구체.
// Hit() 검사 시 레이의 발사 시각(ray.Time())에 해당하는 위치를 보간하여 사용한다.
//
// 카메라 셔터가 열린 동안 구체가 움직이면,
// 각 레이마다 다른 위치에서 교차 검사가 이루어져 모션 블러 효과가 나타난다.
class MovingSphere : public Hittable
{
public:
    __device__ MovingSphere() {}

    __device__ MovingSphere(
        Point3 center0, Point3 center1,
        double time0, double time1,
        double radius, Material* material)
        : mCenter0(center0)
        , mCenter1(center1)
        , mTime0(time0)
        , mTime1(time1)
        , mRadius(radius)
        , mMaterial(material)
    {
        // 운동 구간 전체를 감싸려면 time0 위치의 상자와 time1 위치의 상자를
        // 모두 포함하는 상자가 필요하다.
        Vector3 rvec(radius, radius, radius);
        Aabb box0(center0 - rvec, center0 + rvec);
        Aabb box1(center1 - rvec, center1 + rvec);
        mBBox = Aabb(box0, box1);
    }

    // 레이 발사 시각에 따라 보간된 구체 중심 위치를 반환
    __device__ Point3 Center(double time) const
    {
        return mCenter0 + ((time - mTime0) / (mTime1 - mTime0)) * (mCenter1 - mCenter0);
    }

    __device__ bool Hit(
        const Ray& ray,
        double tMin,
        double tMax,
        HitRecord& hitRecord) const override
    {
        // 레이 발사 시각에 따라 구체 중심 보간 (멤버함수 호출 대신 직접 계산)
        double frac = (ray.Time() - mTime0) / (mTime1 - mTime0);
        Point3 currentCenter = mCenter0 + frac * (mCenter1 - mCenter0);

        Vector3 oc = ray.Origin() - currentCenter;
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
                Vector3 outwardNormal = (hitRecord.P - currentCenter) / mRadius;
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
                Vector3 outwardNormal = (hitRecord.P - currentCenter) / mRadius;
                hitRecord.SetFaceNormal(ray, outwardNormal);
                GetSphereUV(outwardNormal, hitRecord.U, hitRecord.V);
                hitRecord.MaterialPtr = mMaterial;
                return true;
            }
        }

        return false;
    }

    __device__ Aabb BoundingBox() const override { return mBBox; }

    // 정지 구와 동일한 (u,v) 매핑. 움직이는 구도 표면 좌표 자체는 단위 구
    // 기준으로 동일하게 잡는다(중심 이동과 무관하게 법선 방향으로 결정).
    __device__ static void GetSphereUV(const Point3& p, double& u, double& v)
    {
        const double pi = 3.1415926535897932385;
        double theta = acos(-p.Y());
        double phi = atan2(-p.Z(), p.X()) + pi;
        u = phi / (2.0 * pi);
        v = theta / pi;
    }

private:
    Point3 mCenter0;   // 시각 time0에서의 중심
    Point3 mCenter1;   // 시각 time1에서의 중심
    double mTime0;     // 이동 시작 시각
    double mTime1;     // 이동 종료 시각
    double mRadius;
    Material* mMaterial;
    Aabb mBBox;        // 운동 구간 전체를 감싸는 경계 상자
};

#endif
