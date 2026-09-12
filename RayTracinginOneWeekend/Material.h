#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

#include "Ray.h"
#include "Vec3.h"
#include "Onb.h"
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

// 디바이스 코드에서 쓰는 π (RtWeekend.h의 Pi는 호스트 전용 헤더에 있다)
constexpr double kPi = 3.1415926535897932385;

// === The Rest of Your Life Chapter 6: 단위 구 "위"의 무작위 방향 ===
// 단위 공 안의 점을 뽑아 정규화한다. 원점에 아주 가까운 점은 정규화할 때 0으로
// 나누게 되므로 다시 뽑는다.
__device__ inline Vector3 RandomUnitVector(curandState* randState)
{
    while (true)
    {
        Vector3 p = 2.0 * Vector3(curand_uniform(randState),
                                   curand_uniform(randState),
                                   curand_uniform(randState)) - Vector3(1.0, 1.0, 1.0);
        double lengthSquared = p.LengthSquared();
        if (1e-160 < lengthSquared && lengthSquared < 1.0)
            return p / sqrt(lengthSquared);
    }
}

// === The Rest of Your Life Chapter 7: 역변환법으로 코사인 분포 방향 만들기 ===
// 거절법(RandomInUnitSphere / RandomUnitVector)과 달리 루프가 없다. 난수 두 개로
// 바로 계산하므로 워프 안 모든 레인이 같은 시간에 끝난다(GPU에 유리).
//   phi = 2 pi r1,  cos(theta) = sqrt(1 - r2)  ← cos(theta)/pi 분포의 CDF를 뒤집은 것
// 여기서 나오는 방향은 "z축이 법선"인 좌표계 기준이다. 8장에서 정규직교 기저(ONB)로
// 실제 법선 방향에 맞춰 돌린다.
__device__ inline Vector3 RandomCosineDirection(curandState* randState)
{
    double r1 = curand_uniform_double(randState);
    double r2 = curand_uniform_double(randState);

    double phi = 2.0 * kPi * r1;
    double z = sqrt(1.0 - r2);     // cos(theta)
    double r = sqrt(r2);           // sin(theta)

    return Vector3(cos(phi) * r, sin(phi) * r, z);
}

// 재질 기본 클래스
class Material
{
public:
    // === The Next Week Chapter 7: 발광(Emissive) ===
    // 물체가 장면에 빛을 방출하면 이 함수가 그 색을 알려준다(반사 없음).
    // 비발광 재질은 이 기본 구현(검정)을 그대로 물려받아 아무 빛도 내지 않는다.
    // === The Rest of Your Life Chapter 9: 입사 레이/히트 정보를 함께 받는다 ===
    // 광원이 "어느 면으로 빛을 내는지"를 판단할 수 있도록 rayIn과 rec를 넘긴다.
    __device__ virtual Color Emitted(
        const Ray& rayIn, const HitRecord& rec, double u, double v, const Point3& p) const
    {
        return Color(0.0, 0.0, 0.0);
    }

    // === The Rest of Your Life Chapter 8: pdf 출력 인자 추가 ===
    // 재질이 "어떤 밀도로 이 방향을 뽑았는지"(pdf)를 함께 알려준다. RayColor는 더 이상
    // pdfValue를 스스로 가정하지 않고 이 값으로 나눈다. 방향이 사실상 하나로 정해지는
    // 재질(Metal/Dielectric)은 pdf = 0을 돌려주고, 그 경우 RayColor가 감쇠만 곱한다.
    __device__ virtual bool Scatter(
        const Ray& rayIn,
        const HitRecord& rec,
        Color& attenuation,
        Ray& scattered,
        double& pdf,
        curandState* randState) const = 0;

    // === The Rest of Your Life Chapter 6: 산란 PDF (pScatter) ===
    // 이 재질이 scattered 방향으로 빛을 보낼 확률 밀도(입체각 기준).
    // 기본값 0 = "정의하지 않음". Metal/Dielectric처럼 방향이 사실상 하나로 정해지는
    // (PDF가 델타 함수인) 재질은 f/p로 나눌 수 없으므로, RayColor가 이 경우 감쇠만
    // 곱한다(원서 12장의 skip_pdf와 같은 처리).
    __device__ virtual double ScatteringPdf(
        const Ray& rayIn, const HitRecord& rec, const Ray& scattered) const
    {
        return 0.0;
    }
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
        double& pdf,
        curandState* randState) const override
    {
        // === 3권 8장: ONB + 코사인 분포 역변환 샘플링 ===
        // 7장의 RandomCosineDirection은 "z축이 법선"인 좌표계의 방향이다. 이 표면의
        // 법선에 맞춘 정규직교 기저로 옮기면 곧바로 cos/pi 분포의 산란 방향이 된다.
        // (6장까지 쓰던 "법선 + 구 위의 점"과 같은 분포지만, 거절법 루프가 없어
        //  워프 발산이 없고 pdf 값을 그 자리에서 정확히 알 수 있다.)
        Onb uvw(rec.Normal);
        Vector3 scatterDirection = UnitVector(uvw.Transform(RandomCosineDirection(randState)));

        // 산란 레이는 입력 레이의 time을 그대로 물려받는다
        scattered = Ray(rec.P, scatterDirection, rayIn.Time());
        // 히트 지점의 텍스처 좌표로 색을 조회한다(단색이면 항상 같은 값).
        attenuation = mTexture->Value(rec.U, rec.V, rec.P);
        // 이 방향을 뽑은 밀도 = cos(theta)/pi (방향과 법선 사이 각)
        pdf = Dot(uvw.W(), scatterDirection) / kPi;
        return true;
    }

    // === 3권 6장: Lambertian의 산란 PDF ===
    // pScatter = cos(theta) / pi  (theta는 법선과 산란 방향 사이 각, 수평선 아래는 0).
    // 반구에서 적분하면 1이 되도록 1/pi로 정규화한 값이다(3권 5장).
    __device__ double ScatteringPdf(
        const Ray& rayIn, const HitRecord& rec, const Ray& scattered) const override
    {
        double cosTheta = Dot(rec.Normal, UnitVector(scattered.Direction()));
        return cosTheta < 0.0 ? 0.0 : cosTheta / kPi;
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

    // === 3권 9장: 한쪽 면만 빛을 낸다 ===
    // 코넬 박스 천장 광원은 아래(방 안쪽)로만 빛을 내야 한다. 뒷면(천장과 광원 사이
    // 좁은 틈에서 보이는 쪽)까지 빛나면 그 틈에서 반짝이는 노이즈가 생긴다.
    __device__ Color Emitted(
        const Ray& rayIn, const HitRecord& rec, double u, double v, const Point3& p) const override
    {
        if (!rec.bFrontFace)
            return Color(0.0, 0.0, 0.0);

        return mTexture->Value(u, v, p);
    }

    // 빛은 산란하지 않는다.
    __device__ bool Scatter(
        const Ray& rayIn,
        const HitRecord& rec,
        Color& attenuation,
        Ray& scattered,
        double& pdf,
        curandState* randState) const override
    {
        return false;
    }

private:
    Texture* mTexture;
};

// === The Next Week Chapter 9: 등방성 산란 (Isotropic) ===
//
// ConstantMedium(연기/안개)이 산란할 때 쓰는 위상 함수(phase function).
// 입사 방향과 무관하게 "균일한 무작위 방향"으로 산란시킨다(등방성). 감쇠색은
// 텍스처로 조회하므로 단색(연기=검정, 안개=흰색)뿐 아니라 임의 텍스처도 가능.
class Isotropic : public Material
{
public:
    __device__ Isotropic(const Color& albedo)
        : mTexture(new SolidColor(albedo))
    {
    }

    __device__ Isotropic(Texture* texture)
        : mTexture(texture)
    {
    }

    __device__ bool Scatter(
        const Ray& rayIn,
        const HitRecord& rec,
        Color& attenuation,
        Ray& scattered,
        double& pdf,
        curandState* randState) const override
    {
        // 균일 무작위 방향(단위 구 위의 한 점). 입사 레이의 time은 보존한다.
        scattered = Ray(rec.P, RandomUnitVector(randState), rayIn.Time());
        attenuation = mTexture->Value(rec.U, rec.V, rec.P);
        // === 3권 8장 === 구 전체에 균일 → 밀도는 1/(4 pi)
        pdf = 1.0 / (4.0 * kPi);
        return true;
    }

    // 등방성 산란의 산란 PDF도 구 전체 균일(1/4 pi)이다.
    __device__ double ScatteringPdf(
        const Ray& rayIn, const HitRecord& rec, const Ray& scattered) const override
    {
        return 1.0 / (4.0 * kPi);
    }

private:
    Texture* mTexture;
};

#endif
