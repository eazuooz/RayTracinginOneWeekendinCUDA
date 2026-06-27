#pragma once
#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include <cfloat>
#include <curand_kernel.h>

#include "Hittable.h"
#include "Material.h"
#include "Texture.h"

// === The Next Week Chapter 9: Volumes (볼륨 / 참여 매질) ===
//
// 연기·안개·박무 같은 볼륨을 "확률적 표면"으로 흉내 낸다. 레이가 매질을
// 통과하는 동안 어느 지점에서든 산란할 수 있는데, 매질이 짙을수록 그 확률이
// 커진다. 미분방정식을 풀면, 균일 난수 하나로부터 "산란이 일어나는 거리"가 나온다:
//   hit_distance = (-1/density) * log(random)         (random ∈ (0,1])
// 이 거리가 경계 안쪽 이동거리보다 멀면 레이는 매질을 그냥 통과한다(미스).
//
// 경계(boundary)는 또 다른 Hittable로 받는다(상자/구 등). 구현은 경계가
// 볼록(convex)하다고 가정한다 — 상자·구는 되지만 토러스·공동 포함 형태는 안 된다.
//
// CUDA 적용 메모:
//  - random_double() → curand_uniform(randState). 그래서 Hittable::Hit가
//    randState를 받도록 시그니처를 확장했다(Hittable.h 주석 참고). curand_uniform은
//    (0,1] 을 반환하므로 log(0)=-inf 문제가 없다.
//  - shared_ptr → raw 포인터. ConstantMedium이 경계와 위상함수(Isotropic)를
//    소유하고, 소멸자에서 연쇄 해제한다(Hittable의 가상 소멸자 덕분).
class ConstantMedium : public Hittable
{
public:
    __device__ ConstantMedium(Hittable* boundary, double density, Texture* tex)
        : mBoundary(boundary)
        , mNegInvDensity(-1.0 / density)
        , mPhaseFunction(new Isotropic(tex))
    {
    }

    __device__ ConstantMedium(Hittable* boundary, double density, const Color& albedo)
        : mBoundary(boundary)
        , mNegInvDensity(-1.0 / density)
        , mPhaseFunction(new Isotropic(albedo))
    {
    }

    __device__ ~ConstantMedium() override
    {
        delete mBoundary;        // 경계 체인(Translate→RotateY→…)까지 연쇄 해제
        delete mPhaseFunction;
    }

    __device__ bool Hit(
        const Ray& ray, double tMin, double tMax, HitRecord& rec,
        curandState* randState) const override
    {
        HitRecord rec1, rec2;

        // 경계와의 두 교점(레이가 매질에 들어가는 곳 rec1, 나가는 곳 rec2)을 구한다.
        // 레이 시작점이 매질 내부일 수도 있어, 전 구간(universe)에서 검사한다.
        if (!mBoundary->Hit(ray, -DBL_MAX, DBL_MAX, rec1, randState))
            return false;

        if (!mBoundary->Hit(ray, rec1.T + 0.0001, DBL_MAX, rec2, randState))
            return false;

        // 레이 구간 [tMin, tMax]로 잘라낸다.
        if (rec1.T < tMin) rec1.T = tMin;
        if (rec2.T > tMax) rec2.T = tMax;

        if (rec1.T >= rec2.T)
            return false;

        if (rec1.T < 0.0)
            rec1.T = 0.0;

        // 매질 안에서 실제 이동 거리와, 난수로 정한 산란 거리를 비교한다.
        double rayLength = ray.Direction().Length();
        double distanceInsideBoundary = (rec2.T - rec1.T) * rayLength;
        double hitDistance = mNegInvDensity * log(curand_uniform(randState));

        // 산란 거리가 매질을 벗어나면 레이는 그냥 통과(미스).
        if (hitDistance > distanceInsideBoundary)
            return false;

        // 매질 내부에서 산란 — 그 지점을 히트로 기록한다.
        rec.T = rec1.T + hitDistance / rayLength;
        rec.P = ray.At(rec.T);

        rec.Normal = Vector3(1, 0, 0);   // 임의(등방성 산란이라 법선은 의미 없음)
        rec.bFrontFace = true;           // 임의
        rec.MaterialPtr = mPhaseFunction;

        return true;
    }

    __device__ Aabb BoundingBox() const override { return mBoundary->BoundingBox(); }

private:
    Hittable* mBoundary;
    double mNegInvDensity;
    Material* mPhaseFunction;
};

#endif
