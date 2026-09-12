#pragma once
#ifndef ONB_H
#define ONB_H

#include "Vec3.h"

// === The Rest of Your Life Chapter 8: 정규직교 기저 (Orthonormal Basis) ===
//
// 7장에서 만든 방향들은 "z축이 법선"인 좌표계 기준이다. 실제 표면의 법선은 제각각이므로,
// 그 좌표계를 법선에 맞춰 돌려 줘야 한다. 서로 수직인 단위 벡터 세 개(정규직교 기저)를
// 만들어 두면 회전은 곱셈 세 번으로 끝난다.
//
//   w = 단위 법선
//   a = w와 나란하지 않은 아무 축   (|w.x| > 0.9 이면 y축, 아니면 x축을 고른다)
//   v = unit(cross(w, a))           (w와 a 모두에 수직)
//   u = cross(w, v)                 (w, v가 단위이고 수직이라 u도 단위 벡터)
//
// 그러면 z축 기준 방향 (x, y, z)는 x*u + y*v + z*w 로 옮겨진다.
//
// CUDA 적용 메모: 원서의 onb 클래스와 같은 구조지만 모든 멤버가 __device__ 이고,
// 힙을 쓰지 않는 작은 값 타입이라 산란 함수 안에서 스택에 그대로 만들어 쓴다.
class Onb
{
public:
    __device__ Onb() {}

    __device__ Onb(const Vector3& n)
    {
        mAxis[2] = UnitVector(n);
        Vector3 a = (fabs(mAxis[2].X()) > 0.9) ? Vector3(0.0, 1.0, 0.0) : Vector3(1.0, 0.0, 0.0);
        mAxis[1] = UnitVector(Cross(mAxis[2], a));
        mAxis[0] = Cross(mAxis[2], mAxis[1]);
    }

    __device__ const Vector3& U() const { return mAxis[0]; }
    __device__ const Vector3& V() const { return mAxis[1]; }
    __device__ const Vector3& W() const { return mAxis[2]; }

    // 기저 좌표 → 월드 좌표
    __device__ Vector3 Transform(const Vector3& v) const
    {
        return v.X() * mAxis[0] + v.Y() * mAxis[1] + v.Z() * mAxis[2];
    }

private:
    Vector3 mAxis[3];
};

#endif
