#pragma once
#ifndef PDF_H
#define PDF_H

#include "Vec3.h"
#include "Onb.h"
#include "Hittable.h"
#include "Sampling.h"   // kPi, RandomUnitVector, RandomCosineDirection

// === The Rest of Your Life Chapter 10: PDF 클래스 ===
//
// 지금까지 PDF는 "값을 구하는 식"과 "그 분포로 방향을 뽑는 코드"가 여기저기 흩어져
// 있었다. 둘은 언제나 짝이어야 하므로(뽑은 분포와 나눈 밀도가 다르면 편향된다, 6장)
// 하나의 클래스로 묶는다. PDF 하나가 할 일은 두 가지다.
//   1) Generate : 자기 분포대로 무작위 방향을 하나 만든다.
//   2) Value    : 주어진 방향에 대한 밀도를 돌려준다.
//
// ─── CUDA 적용 메모 ───────────────────────────────────────────────
// 원서는 ray_color 안에서 make_shared로 PDF를 만든다. GPU에서 바운스마다 힙 할당을
// 하면(디바이스 new) 매우 느리고 힙도 금방 바닥난다. 그래서 이 클래스들은 전부
// **스택에 만드는 작은 값 타입**으로 쓴다. 가상 함수는 그대로 쓸 수 있다(디바이스에서
// 생성한 객체의 vtable은 디바이스 코드에서 정상 동작한다).
//
//   CosinePdf surfacePdf(rec.Normal);          // 스택
//   HittablePdf lightPdf(*lights, rec.P);      // 스택
//   MixturePdf mixed(&lightPdf, &surfacePdf);  // 포인터만 들고 있음
class Pdf
{
public:
    __device__ virtual ~Pdf() {}

    // direction 방향의 밀도(입체각 기준)
    __device__ virtual double Value(const Vector3& direction) const = 0;

    // 이 분포를 따르는 무작위 방향
    __device__ virtual Vector3 Generate(curandState* randState) const = 0;
};

// 구 전체에 균일한 분포. 등방성 산란(연기/안개)에 대응한다.
class SpherePdf : public Pdf
{
public:
    __device__ SpherePdf() {}

    __device__ double Value(const Vector3& direction) const override
    {
        return 1.0 / (4.0 * kPi);
    }

    __device__ Vector3 Generate(curandState* randState) const override
    {
        return RandomUnitVector(randState);
    }
};

// 법선 기준 cos(theta)/pi 분포. Lambertian 표면에 대응한다(8장의 ONB + 역변환 샘플링).
class CosinePdf : public Pdf
{
public:
    // 12장의 ScatterRecord가 값으로 품고 있다가 나중에 채우므로 기본 생성자가 필요하다.
    __device__ CosinePdf() {}
    __device__ CosinePdf(const Vector3& w) : mUvw(w) {}

    __device__ double Value(const Vector3& direction) const override
    {
        double cosTheta = Dot(UnitVector(direction), mUvw.W());
        return (cosTheta <= 0.0) ? 0.0 : cosTheta / kPi;
    }

    __device__ Vector3 Generate(curandState* randState) const override
    {
        return mUvw.Transform(RandomCosineDirection(randState));
    }

private:
    Onb mUvw;
};

// 어떤 Hittable(보통 광원)을 향하는 분포. 실제 계산은 그 물체가 한다
// (Hittable::PdfValue / Hittable::Random).
class HittablePdf : public Pdf
{
public:
    __device__ HittablePdf(const Hittable* objects, const Point3& origin, curandState* randState)
        : mObjects(objects), mOrigin(origin), mRandState(randState)
    {
    }

    __device__ double Value(const Vector3& direction) const override
    {
        return mObjects->PdfValue(mOrigin, direction, mRandState);
    }

    __device__ Vector3 Generate(curandState* randState) const override
    {
        return mObjects->Random(mOrigin, randState);
    }

private:
    const Hittable* mObjects;
    Point3 mOrigin;
    // Hittable::PdfValue는 내부에서 Hit()을 부르는데, 우리 Hit 시그니처는 볼륨 때문에
    // curandState*를 요구한다(2권 9장). Value()는 난수를 쓰지 않지만 전달만 해 준다.
    curandState* mRandState;
};

// 두 PDF를 반반 섞은 분포.
//   Generate : 반 확률로 한쪽을 골라 뽑는다.
//   Value    : "어느 쪽에서 나왔는지"는 알 수 없고 알 필요도 없다. 같은 방향을 양쪽이
//              만들 수 있으므로, 두 밀도의 가중 평균이 곧 혼합 분포의 밀도다.
class MixturePdf : public Pdf
{
public:
    __device__ MixturePdf(const Pdf* p0, const Pdf* p1)
    {
        mPdf[0] = p0;
        mPdf[1] = p1;
    }

    __device__ double Value(const Vector3& direction) const override
    {
        return 0.5 * mPdf[0]->Value(direction) + 0.5 * mPdf[1]->Value(direction);
    }

    __device__ Vector3 Generate(curandState* randState) const override
    {
        return (curand_uniform(randState) < 0.5f)
            ? mPdf[0]->Generate(randState)
            : mPdf[1]->Generate(randState);
    }

private:
    const Pdf* mPdf[2];
};

#endif
