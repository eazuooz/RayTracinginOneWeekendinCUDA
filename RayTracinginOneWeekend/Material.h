#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

#include "Ray.h"
#include "Vec3.h"
#include <curand_kernel.h>

struct HitRecord;

// cuRAND를 이용한 단위 구 내부의 랜덤 점 생성
__device__ inline Vector3 RandomInUnitSphere(curandState* randState)
{
    Vector3 p;
    do
    {
        p = 2.0 * Vector3(curand_uniform(randState),
                           curand_uniform(randState),
                           curand_uniform(randState)) - Vector3(1.0, 1.0, 1.0);
    } while (p.LengthSquared() >= 1.0);
    return p;
}

// 재질 기본 클래스
class Material
{
public:
    __device__ virtual bool Scatter(
        const Ray& rayIn,
        const HitRecord& rec,
        Color& attenuation,
        Ray& scattered,
        curandState* randState) const = 0;
};

// 난반사 재질 (Lambertian)
class Lambertian : public Material
{
public:
    __device__ Lambertian(const Color& albedo)
        : mAlbedo(albedo)
    {
    }

    __device__ bool Scatter(
        const Ray& rayIn,
        const HitRecord& rec,
        Color& attenuation,
        Ray& scattered,
        curandState* randState) const override
    {
        Vector3 target = rec.P + rec.Normal + RandomInUnitSphere(randState);
        scattered = Ray(rec.P, target - rec.P);
        attenuation = mAlbedo;
        return true;
    }

private:
    Color mAlbedo;
};

// 금속 재질 (Metal)
class Metal : public Material
{
public:
    __device__ Metal(const Color& albedo, double fuzz)
        : mAlbedo(albedo)
        , mFuzz(fuzz < 1.0 ? fuzz : 1.0)
    {
    }

    __device__ bool Scatter(
        const Ray& rayIn,
        const HitRecord& rec,
        Color& attenuation,
        Ray& scattered,
        curandState* randState) const override
    {
        Vector3 reflected = Reflect(UnitVector(rayIn.Direction()), rec.Normal);
        scattered = Ray(rec.P, reflected + mFuzz * RandomInUnitSphere(randState));
        attenuation = mAlbedo;
        return (Dot(scattered.Direction(), rec.Normal) > 0.0);
    }

private:
    Color mAlbedo;
    double mFuzz;
};

#endif
