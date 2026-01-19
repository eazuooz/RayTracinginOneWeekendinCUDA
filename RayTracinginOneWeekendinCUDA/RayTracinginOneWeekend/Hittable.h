#pragma once
#ifndef HITTABLE_H
#define HITTABLE_H

#include "RtWeekend.h"

class HitRecord
{
public:
    void SetFaceNormal(const Ray& r, const Vec3& outwardNormal)
    {
        // 히트 레코드 법선 벡터를 설정합니다.
        // 참고: 매개변수 `outwardNormal`은 단위 길이를 가진다고 가정합니다.

        bFrontFace = Dot(r.Direction(), outwardNormal) < 0;
        Normal = bFrontFace ? outwardNormal : -outwardNormal;
    }

    Point3 P;
    Vec3 Normal;
    double T;
    bool bFrontFace;
};

class Hittable
{
public:
    virtual ~Hittable() = default;

    virtual bool Hit(const Ray& ray, const Interval& rayT, HitRecord& hitRecord) const = 0;
};

#endif