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
    bool bFrontFace;
    Material* MaterialPtr;

    // 레이 방향과 외부 법선으로 앞면/뒷면 판정
    // 법선은 항상 레이 반대 방향(표면 바깥쪽)을 가리키도록 설정
    __device__ void SetFaceNormal(const Ray& ray, const Vector3& outwardNormal)
    {
        bFrontFace = Dot(ray.Direction(), outwardNormal) < 0.0;
        Normal = bFrontFace ? outwardNormal : -outwardNormal;
    }
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
