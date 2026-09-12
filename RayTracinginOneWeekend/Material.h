#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

#include "Ray.h"
#include "Vec3.h"
#include "Onb.h"
#include "Sampling.h"
#include "Pdf.h"
#include "Texture.h"
#include "Hittable.h"
#include <curand_kernel.h>

struct HitRecord;

// === The Rest of Your Life Chapter 12: 산란 결과 묶음 (ScatterRecord) ===
//
// 11장에서 정리한 문제 중 하나가 "PDF를 RayColor가 직접 고른다"였다. 이제 재질이
// 자기 PDF를 돌려준다. 늘어나는 출력값(감쇠, PDF, 정반사 여부, 정반사 레이)을
// 인자 목록에 계속 붙이는 대신 하나의 구조체로 묶는다.
//
//   bSkipPdf == true  : 거울/유리처럼 방향이 하나로 정해지는 재질. PDF 없이
//                       SkipPdfRay를 그대로 따라간다(f/p 가중 없음).
//   bSkipPdf == false : 확률적 산란. PdfPtr이 가리키는 PDF로 방향을 뽑는다.
//
// ─── CUDA 적용 메모 ───────────────────────────────────────────────
// 원서는 srec.pdf_ptr에 make_shared로 만든 PDF를 담는다. GPU에서는 바운스마다
// 힙 할당을 할 수 없으므로, 쓸 수 있는 PDF들을 **값으로 품고 있다가** PdfPtr로
// 그중 하나를 가리킨다. ScatterRecord 자체가 RayColor의 스택에 있으므로 추가
// 할당이 전혀 없다. (ScatterRecord를 복사하면 PdfPtr이 원본을 가리키게 되므로
// 복사하지 말고 참조로만 넘긴다.)
struct ScatterRecord
{
    Color Attenuation;
    const Pdf* PdfPtr;     // bSkipPdf == false일 때 유효 (아래 저장소 중 하나를 가리킨다)
    bool bSkipPdf;
    Ray SkipPdfRay;        // bSkipPdf == true일 때 따라갈 레이

    // PDF 저장소(디바이스 new 회피용)
    CosinePdf CosinePdfStorage;
    SpherePdf SpherePdfStorage;

    __device__ ScatterRecord()
        : PdfPtr(nullptr)
        , bSkipPdf(false)
    {
    }
};

// 재질 기본 클래스
class Material
{
public:
    // === The Next Week Chapter 7: 발광(Emissive) ===
    // 물체가 장면에 빛을 방출하면 이 함수가 그 색을 알려준다(반사 없음).
    // === The Rest of Your Life Chapter 9: 입사 레이/히트 정보를 함께 받는다 ===
    // 광원이 "어느 면으로 빛을 내는지"를 판단할 수 있도록 rayIn과 rec를 넘긴다.
    __device__ virtual Color Emitted(
        const Ray& rayIn, const HitRecord& rec, double u, double v, const Point3& p) const
    {
        return Color(0.0, 0.0, 0.0);
    }

    // === The Rest of Your Life Chapter 12: ScatterRecord로 정리 ===
    // 산란하지 않는 재질(광원)은 기본 구현대로 false를 돌려주면 된다.
    __device__ virtual bool Scatter(
        const Ray& rayIn, const HitRecord& rec, ScatterRecord& srec, curandState* randState) const
    {
        return false;
    }

    // === The Rest of Your Life Chapter 6: 산란 PDF (pScatter) ===
    // 이 재질이 scattered 방향으로 빛을 보낼 확률 밀도(입체각 기준).
    // 정반사 재질은 정의하지 않는다(bSkipPdf로 걸러지므로 호출되지 않는다).
    __device__ virtual double ScatteringPdf(
        const Ray& rayIn, const HitRecord& rec, const Ray& scattered) const
    {
        return 0.0;
    }
};

// 난반사 재질 (Lambertian)
//
// 텍스처 매핑 적용: albedo를 단일 색이 아니라 Texture*로 들고 있다. 단색 생성자(Color)를
// 주면 내부적으로 SolidColor 텍스처로 감싸므로 기존 호출부는 그대로 동작한다.
class Lambertian : public Material
{
public:
    __device__ Lambertian(const Color& albedo)
        : mTexture(new SolidColor(albedo))
    {
    }

    __device__ Lambertian(Texture* texture)
        : mTexture(texture)
    {
    }

    // === 3권 12장 ===
    // 방향을 여기서 뽑지 않는다. "코사인 분포로 뽑아 달라"고 PDF만 넘긴다.
    // 실제 생성과 밀도 계산은 RayColor가 (광원 PDF와 섞어서) 한다.
    __device__ bool Scatter(
        const Ray& rayIn, const HitRecord& rec, ScatterRecord& srec, curandState* randState) const override
    {
        srec.Attenuation = mTexture->Value(rec.U, rec.V, rec.P);
        srec.CosinePdfStorage = CosinePdf(rec.Normal);
        srec.PdfPtr = &srec.CosinePdfStorage;
        srec.bSkipPdf = false;
        return true;
    }

    // === 3권 6장: Lambertian의 산란 PDF ===
    // pScatter = cos(theta) / pi  (theta는 법선과 산란 방향 사이 각, 수평선 아래는 0).
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
// (Scatter는 기본 구현이 false를 돌려준다).
class DiffuseLight : public Material
{
public:
    __device__ DiffuseLight(Texture* texture)
        : mTexture(texture)
    {
    }

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

private:
    Texture* mTexture;
};

// === The Next Week Chapter 9: 등방성 산란 (Isotropic) ===
//
// ConstantMedium(연기/안개)이 산란할 때 쓰는 위상 함수(phase function).
// 입사 방향과 무관하게 "균일한 무작위 방향"으로 산란시킨다(등방성).
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

    // === 3권 12장 === 구 전체 균일 분포(SpherePdf)를 돌려준다.
    __device__ bool Scatter(
        const Ray& rayIn, const HitRecord& rec, ScatterRecord& srec, curandState* randState) const override
    {
        srec.Attenuation = mTexture->Value(rec.U, rec.V, rec.P);
        srec.SpherePdfStorage = SpherePdf();
        srec.PdfPtr = &srec.SpherePdfStorage;
        srec.bSkipPdf = false;
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
