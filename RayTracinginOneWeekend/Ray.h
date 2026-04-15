#pragma once
#ifndef RAY_H
#define RAY_H

#include "Vec3.h"

// === The Next Week: 모션 블러 ===
// Ray에 time 필드를 추가한다.
// 카메라가 셔터 개방 시간 동안 랜덤한 time을 각 레이에 부여하고,
// MovingSphere는 ray.Time()을 읽어 그 순간의 위치에서 교차 검사를 수행한다.
class Ray
{
public:
    __device__ Ray() {}

    __device__ Ray(const Point3& origin, const Vector3& direction, double time = 0.0)
        : mOrig(origin), mDir(direction), mTime(time) {}

    __device__ Ray(const Ray& other)
        : mOrig(other.mOrig), mDir(other.mDir), mTime(other.mTime) {}

    __device__ const Point3& Origin() const { return mOrig; }
    __device__ const Vector3& Direction() const { return mDir; }
    __device__ double Time() const { return mTime; }

    __device__ Point3 At(double t) const
    {
        return mOrig + t * mDir;
    }

private:
    Point3 mOrig;
    Vector3 mDir;
    double mTime;  // 레이가 발사된 시각 (셔터 개방 구간 내 랜덤 값)
};

#endif
