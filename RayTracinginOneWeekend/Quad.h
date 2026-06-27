#pragma once
#ifndef QUAD_H
#define QUAD_H

#include "Hittable.h"

// === The Next Week Chapter 6: Quadrilaterals (사각형) ===
//
// 구(sphere)에 이어 두 번째 프리미티브. 이름은 quad지만 엄밀히는 평행사변형이다.
// 세 가지 기하 요소로 정의한다:
//   Q : 시작 모서리(평면의 원점 역할)
//   u : 첫 번째 변 벡터. Q+u 가 인접 모서리.
//   v : 두 번째 변 벡터. Q+v 가 다른 인접 모서리. (대각 모서리는 Q+u+v)
//
// 레이-사각형 교차는 3단계로 푼다:
//   1) 사각형을 포함하는 평면을 찾는다(법선 n, 상수 D).
//   2) 레이와 그 평면의 교점 t 를 구한다.
//   3) 교점이 사각형 내부((alpha,beta) ∈ [0,1]^2)인지 판정한다.
//
// CUDA 적용 메모: 원서의 shared_ptr<material> 대신 raw Material*. 모든 메서드는
// __device__. 평면/좌표계 상수(normal, D, w)는 생성자에서 캐시한다.
class Quad : public Hittable
{
public:
    __device__ Quad(const Point3& q, const Vector3& u, const Vector3& v, Material* material)
        : mQ(q)
        , mU(u)
        , mV(v)
        , mMaterial(material)
    {
        // 평면 값 캐시: 법선 n = u×v, 단위 법선 normal, D = normal·Q,
        // 그리고 평면 좌표 계산용 상수 w = n / (n·n).
        Vector3 n = Cross(u, v);
        mNormal = UnitVector(n);
        mD = Dot(mNormal, mQ);
        mW = n / Dot(n, n);

        SetBoundingBox();
    }

    // 네 꼭짓점을 감싸는 AABB. 두 대각선 박스를 합쳐 만든다.
    // (평평한 도형이라 한 축 두께가 0이 될 수 있는데, Aabb가 PadToMinimums로 보정)
    __device__ void SetBoundingBox()
    {
        Aabb diag1 = Aabb(mQ, mQ + mU + mV);
        Aabb diag2 = Aabb(mQ + mU, mQ + mV);
        mBBox = Aabb(diag1, diag2);
    }

    __device__ Aabb BoundingBox() const override { return mBBox; }

    __device__ bool Hit(
        const Ray& ray, double tMin, double tMax, HitRecord& hitRecord,
        curandState* randState) const override
    {
        double denom = Dot(mNormal, ray.Direction());

        // 레이가 평면과 평행하면 교차 없음.
        if (fabs(denom) < 1e-8)
            return false;

        // 평면 교점 파라미터 t 가 레이 구간 밖이면 미스.
        double t = (mD - Dot(mNormal, ray.Origin())) / denom;
        if (t < tMin || t > tMax)
            return false;

        // 교점의 평면 좌표(alpha, beta)로 사각형 내부인지 판정.
        Point3 intersection = ray.At(t);
        Vector3 planarHitptVector = intersection - mQ;
        double alpha = Dot(mW, Cross(planarHitptVector, mV));
        double beta = Dot(mW, Cross(mU, planarHitptVector));

        if (!IsInterior(alpha, beta, hitRecord))
            return false;

        // 사각형에 맞음 → 나머지 히트 레코드 채우기.
        hitRecord.T = t;
        hitRecord.P = intersection;
        hitRecord.MaterialPtr = mMaterial;
        hitRecord.SetFaceNormal(ray, mNormal);

        return true;
    }

    // 평면 좌표 (a,b)가 사각형 내부인지. 내부면 텍스처 좌표(U,V)도 채운다.
    // (이 판정만 바꾸면 원판/삼각형 등 다른 평면 도형으로 확장 가능)
    //   원판:    sqrt(a*a + b*b) < r
    //   삼각형:  a > 0 && b > 0 && a + b < 1
    __device__ bool IsInterior(double a, double b, HitRecord& hitRecord) const
    {
        Interval unitInterval(0.0, 1.0);

        if (!unitInterval.Contains(a) || !unitInterval.Contains(b))
            return false;

        hitRecord.U = a;
        hitRecord.V = b;
        return true;
    }

private:
    Point3 mQ;
    Vector3 mU;
    Vector3 mV;
    Vector3 mW;        // 평면 좌표 계산용 상수 = n / (n·n)
    Material* mMaterial;
    Aabb mBBox;
    Vector3 mNormal;   // 단위 법선
    double mD;         // 평면 상수: normal·Q
};

#endif
