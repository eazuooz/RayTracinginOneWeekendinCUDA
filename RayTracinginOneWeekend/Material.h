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
    // === The Next Week Chapter 7: 발광(Emissive) ===
    // 물체가 장면에 빛을 방출하면 이 함수가 그 색을 알려준다(반사 없음).
    // 비발광 재질은 이 기본 구현(검정)을 그대로 물려받아 아무 빛도 내지 않는다.
    __device__ virtual Color Emitted(double u, double v, const Point3& p) const
    {
        return Color(0.0, 0.0, 0.0);
    }

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

// === The Next Week Chapter 7: 확산 광원 (Diffuse Light) ===
//
// 빛을 방출하는 재질. 배경처럼 레이에 "색"만 알려주고 반사는 하지 않는다
// (Scatter는 항상 false). 방출색은 텍스처로 조회하므로, 단색뿐 아니라
// 이미지/노이즈 텍스처도 광원으로 쓸 수 있다.
//
// 광원은 보통 (1,1,1)보다 밝은 색(예: (4,4,4)/(15,15,15))을 주어야 주변을
// 비출 만큼 충분히 밝다.
class DiffuseLight : public Material
{
public:
    __device__ DiffuseLight(Texture* texture)
        : mTexture(texture)
    {
    }

    // 단색 → SolidColor로 감싼다.
    __device__ DiffuseLight(const Color& emit)
        : mTexture(new SolidColor(emit))
    {
    }

    __device__ Color Emitted(double u, double v, const Point3& p) const override
    {
        return mTexture->Value(u, v, p);
    }

    // 빛은 산란하지 않는다.
    __device__ bool Scatter(
        const Ray& rayIn,
        const HitRecord& rec,
        Color& attenuation,
        Ray& scattered,
        curandState* randState) const override
    {
        return false;
    }

private:
    Texture* mTexture;
};

#endif
