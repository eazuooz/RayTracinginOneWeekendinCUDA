#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

#include "Ray.h"
#include "Vec3.h"
#include "Texture.h"
#include "Hittable.h"
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
//
// 텍스처 매핑 적용: 이제 albedo를 단일 색이 아니라 Texture*로 들고 있다.
// 단색 생성자(Color)를 주면 내부적으로 SolidColor 텍스처로 감싸므로,
// 기존 호출부는 그대로 두어도 동작한다("모든 색은 텍스처"라는 설계).
// 산란 시 히트 지점의 (u,v,p)로 텍스처 색을 조회해 감쇠색으로 쓴다.
class Lambertian : public Material
{
public:
    // 단색 → SolidColor 텍스처로 감싼다. (device new — 단발성 렌더라 누수 허용,
    // 기존 Material 들과 동일하게 program 종료 시 회수된다.)
    __device__ Lambertian(const Color& albedo)
        : mTexture(new SolidColor(albedo))
    {
    }

    // 임의 텍스처(체커/이미지 등)를 직접 받는 생성자.
    __device__ Lambertian(Texture* texture)
        : mTexture(texture)
    {
    }

    __device__ bool Scatter(
        const Ray& rayIn,
        const HitRecord& rec,
        Color& attenuation,
        Ray& scattered,
        curandState* randState) const override
    {
        Vector3 scatterDirection = rec.Normal + RandomInUnitSphere(randState);

        // 산란 방향이 법선과 거의 반대여서 영벡터에 가까워지는 경우 방지
        if (scatterDirection.NearZero())
            scatterDirection = rec.Normal;

        // 산란 레이는 입력 레이의 time을 그대로 물려받는다
        scattered = Ray(rec.P, scatterDirection, rayIn.Time());
        // 히트 지점의 텍스처 좌표로 색을 조회한다(단색이면 항상 같은 값).
        attenuation = mTexture->Value(rec.U, rec.V, rec.P);
        return true;
    }

private:
    Texture* mTexture;
};

#endif
