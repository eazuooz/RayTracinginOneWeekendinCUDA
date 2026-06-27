#pragma once
#ifndef AABB_H
#define AABB_H

#include "Ray.h"
#include "Interval.h"

// === The Next Week Chapter 3: 축 정렬 경계 상자 (Axis-Aligned Bounding Box) ===
//
// 3차원 AABB는 x/y/z 세 축의 구간(슬랩, slab)이 겹치는 영역이다.
// 레이가 이 상자를 통과하는지 빠르게 판정하는 것이 BVH 가속의 핵심이다.
//
// 슬랩 방법: 각 축마다 레이가 두 평면(min/max)을 지나는 파라미터 t 구간을
// 구하고, 세 축의 t 구간이 모두 겹치면(교집합이 비어있지 않으면) 레이가
// 상자를 통과한 것이다.
class Aabb
{
public:
    Interval X;
    Interval Y;
    Interval Z;

    // 기본 AABB는 비어 있다 (Interval 기본 생성자가 빈 구간을 만든다)
    __host__ __device__ Aabb() {}

    __host__ __device__ Aabb(const Interval& x, const Interval& y, const Interval& z)
        : X(x), Y(y), Z(z)
    {
        PadToMinimums();
    }

    // 두 점 a, b를 상자의 양 극점으로 보고 AABB를 만든다
    // (좌표의 대소 순서를 신경 쓰지 않아도 되도록 min/max로 정렬)
    __host__ __device__ Aabb(const Point3& a, const Point3& b)
    {
        X = (a[0] <= b[0]) ? Interval(a[0], b[0]) : Interval(b[0], a[0]);
        Y = (a[1] <= b[1]) ? Interval(a[1], b[1]) : Interval(b[1], a[1]);
        Z = (a[2] <= b[2]) ? Interval(a[2], b[2]) : Interval(b[2], a[2]);
        PadToMinimums();
    }

    // 두 AABB를 모두 감싸는 AABB를 만든다 (BVH 노드의 경계 합산용)
    __host__ __device__ Aabb(const Aabb& box0, const Aabb& box1)
        : X(box0.X, box1.X)
        , Y(box0.Y, box1.Y)
        , Z(box0.Z, box1.Z)
    {
    }

    __host__ __device__ const Interval& AxisInterval(int n) const
    {
        if (n == 1) return Y;
        if (n == 2) return Z;
        return X;
    }

    // 슬랩 방법으로 레이-상자 교차 여부만 판정한다 (히트 지점/법선은 불필요).
    // Ray 접근자가 __device__ 전용이므로 이 함수도 __device__ 전용이다.
    //
    // CUDA 12.9 nvcc codegen issue workaround:
    //   "3회 루프 + 루프 내부 조기 return" 형태의 슬랩 검사는 nvcc 12.9 에서
    //   일부 레이에 대해 잘못 컴파일된다 (첫 축 이후 검사가 끊겨 false 를 반환
    //   → BVH 컬링이 false negative 를 내어 화면이 배경만 나오거나 어둡고
    //   노이즈가 낀다). 루프/조기 return 을 모두 제거하고 세 축을 풀어 쓴
    //   분기 없는(fmin/fmax) 형태로 작성하면 안정적으로 올바른 코드가 나온다.
    //   각 축에서 슬랩 t 구간 [lo,hi] 를 구해 누적 교집합 [tMin,tMax] 를 좁히고,
    //   마지막에 tMax > tMin 이면 세 구간이 겹친 것(=상자 통과)이다.
    __device__ bool Hit(const Ray& ray, Interval rayT) const
    {
        const Point3& o = ray.Origin();
        const Vector3& d = ray.Direction();

        double tMin = rayT.Min;
        double tMax = rayT.Max;

        // X 축 슬랩
        double invDx = 1.0 / d[0];
        double tx0 = (X.Min - o[0]) * invDx;
        double tx1 = (X.Max - o[0]) * invDx;
        tMin = fmax(tMin, fmin(tx0, tx1));
        tMax = fmin(tMax, fmax(tx0, tx1));

        // Y 축 슬랩
        double invDy = 1.0 / d[1];
        double ty0 = (Y.Min - o[1]) * invDy;
        double ty1 = (Y.Max - o[1]) * invDy;
        tMin = fmax(tMin, fmin(ty0, ty1));
        tMax = fmin(tMax, fmax(ty0, ty1));

        // Z 축 슬랩
        double invDz = 1.0 / d[2];
        double tz0 = (Z.Min - o[2]) * invDz;
        double tz1 = (Z.Max - o[2]) * invDz;
        tMin = fmax(tMin, fmin(tz0, tz1));
        tMax = fmin(tMax, fmax(tz0, tz1));

        return tMax > tMin;
    }

    // 가장 긴 축의 인덱스를 반환한다 (BVH 분할 시 분할 축 선택용)
    __host__ __device__ int LongestAxis() const
    {
        if (X.Size() > Y.Size())
            return X.Size() > Z.Size() ? 0 : 2;
        else
            return Y.Size() > Z.Size() ? 1 : 2;
    }

private:
    // === The Next Week Chapter 6: 두께 0 AABB 패딩 ===
    // 사각형(quad)처럼 평평한 도형이 XY/YZ/ZX 평면에 놓이면 한 축의 두께가 0이
    // 되어 레이-상자 교차에서 수치 문제가 생길 수 있다. 어느 변도 delta보다
    // 좁지 않도록 살짝 패딩한다. (교차 결과는 그대로, 경계만 약간 넓힌다.)
    __host__ __device__ void PadToMinimums()
    {
        double delta = 0.0001;
        if (X.Size() < delta) X = X.Expand(delta);
        if (Y.Size() < delta) Y = Y.Expand(delta);
        if (Z.Size() < delta) Z = Z.Expand(delta);
    }
};

// === The Next Week Chapter 8: Instances ===
// AABB를 offset만큼 평행이동한다. Translate가 자식의 경계 상자를 월드 공간으로
// 옮길 때 쓴다(상자를 옮겨 두지 않으면 BVH/슬랩 검사가 엉뚱한 곳을 보고 레이를
// 조기 기각해 버린다). 세 축 구간을 각각 Interval operator+ 로 민다.
__host__ __device__ inline Aabb operator+(const Aabb& bbox, const Vector3& offset)
{
    return Aabb(bbox.X + offset.X(), bbox.Y + offset.Y(), bbox.Z + offset.Z());
}

__host__ __device__ inline Aabb operator+(const Vector3& offset, const Aabb& bbox)
{
    return bbox + offset;
}

#endif
