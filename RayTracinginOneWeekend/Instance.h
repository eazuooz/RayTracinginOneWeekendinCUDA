#pragma once
#ifndef INSTANCE_H
#define INSTANCE_H

#include <cfloat>

#include "Hittable.h"
#include "HittableList.h"
#include "Quad.h"

// === The Next Week Chapter 8: Instances (인스턴스) ===
//
// 인스턴스(instance)는 장면에 배치된 기하 프리미티브의 "복사본"이다. 각 복사본은
// 독립적으로 이동/회전할 수 있다. 레이트레이싱에서는 물체를 실제로 옮기는 대신
// 레이를 반대로 옮긴다(좌표계 변환). 그래서 같은 Quad/Box 메시를 두고도
//   - Translate : 레이를 -offset 만큼 옮겨 교차 → 교점을 +offset 으로 되돌림
//   - RotateY   : 레이를 -θ 만큼 회전해 교차 → 교점/법선을 +θ 로 되돌림
// 으로 옮기거나 돌릴 수 있다.
//
// CUDA 적용 메모:
//   - 원서의 shared_ptr<hittable> → raw Hittable*. 인스턴스가 자식을 "소유"하며,
//     가상 소멸자(Hittable::~Hittable)를 통해 delete가 자식까지 연쇄된다.
//   - 원서 interval 기반 Hit 시그니처 대신 이 프로젝트의 (tMin, tMax) 시그니처.
//   - Pi/Infinity는 호스트 전용 RtWeekend.h에 있어 디바이스에서 못 쓴다.
//     라디안 변환은 상수를 직접 곱하고, 무한대 대용으로 DBL_MAX를 쓴다.

// ── 평행이동 인스턴스 ────────────────────────────────────────────────
class Translate : public Hittable
{
public:
    __device__ Translate(Hittable* object, const Vector3& offset)
        : mObject(object), mOffset(offset)
    {
        // 자식의 경계 상자를 offset만큼 옮겨 둔다. 안 옮기면 BVH 슬랩 검사가
        // 엉뚱한 곳을 보고 레이를 조기 기각한다.
        mBBox = object->BoundingBox() + offset;
    }

    __device__ ~Translate() override { delete mObject; }

    __device__ bool Hit(
        const Ray& ray, double tMin, double tMax, HitRecord& rec) const override
    {
        // 1) 레이를 -offset 만큼 옮긴다(월드 → 오브젝트 공간).
        Ray offsetRay(ray.Origin() - mOffset, ray.Direction(), ray.Time());

        // 2) 옮긴 레이로 교차 검사.
        if (!mObject->Hit(offsetRay, tMin, tMax, rec))
            return false;

        // 3) 교점을 +offset 으로 되돌린다(오브젝트 → 월드 공간).
        rec.P += mOffset;

        return true;
    }

    __device__ Aabb BoundingBox() const override { return mBBox; }

private:
    Hittable* mObject;
    Vector3 mOffset;
    Aabb mBBox;
};

// ── Y축 회전 인스턴스 ────────────────────────────────────────────────
// 회전은 이동보다 까다로워, "좌표계 변환"으로 다루는 편이 안전하다.
//   x' =  cosθ·x + sinθ·z
//   z' = -sinθ·x + cosθ·z   (월드 → 오브젝트는 -θ 회전)
// 회전은 교점뿐 아니라 표면 법선도 같이 돌려야 반사/굴절 방향이 맞는다.
class RotateY : public Hittable
{
public:
    __device__ RotateY(Hittable* object, double angle)
        : mObject(object)
    {
        // 라디안 변환(호스트 DegreesToRadians 대신 디바이스에서 직접 계산).
        double radians = angle * 3.1415926535897932385 / 180.0;
        mSinTheta = sin(radians);
        mCosTheta = cos(radians);
        mBBox = object->BoundingBox();

        // 자식 AABB의 8개 꼭짓점을 모두 회전시켜, 회전 후를 감싸는 새 AABB를 구한다.
        Point3 min(DBL_MAX, DBL_MAX, DBL_MAX);
        Point3 max(-DBL_MAX, -DBL_MAX, -DBL_MAX);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    double x = i * mBBox.X.Max + (1 - i) * mBBox.X.Min;
                    double y = j * mBBox.Y.Max + (1 - j) * mBBox.Y.Min;
                    double z = k * mBBox.Z.Max + (1 - k) * mBBox.Z.Min;

                    double newx = mCosTheta * x + mSinTheta * z;
                    double newz = -mSinTheta * x + mCosTheta * z;

                    Vector3 tester(newx, y, newz);

                    for (int c = 0; c < 3; c++)
                    {
                        min[c] = fmin(min[c], tester[c]);
                        max[c] = fmax(max[c], tester[c]);
                    }
                }
            }
        }

        mBBox = Aabb(min, max);
    }

    __device__ ~RotateY() override { delete mObject; }

    __device__ bool Hit(
        const Ray& ray, double tMin, double tMax, HitRecord& rec) const override
    {
        // 레이를 월드 → 오브젝트 공간으로 회전(-θ).
        Point3 origin(
            (mCosTheta * ray.Origin().X()) - (mSinTheta * ray.Origin().Z()),
            ray.Origin().Y(),
            (mSinTheta * ray.Origin().X()) + (mCosTheta * ray.Origin().Z()));

        Vector3 direction(
            (mCosTheta * ray.Direction().X()) - (mSinTheta * ray.Direction().Z()),
            ray.Direction().Y(),
            (mSinTheta * ray.Direction().X()) + (mCosTheta * ray.Direction().Z()));

        Ray rotatedRay(origin, direction, ray.Time());

        // 오브젝트 공간에서 교차 검사.
        if (!mObject->Hit(rotatedRay, tMin, tMax, rec))
            return false;

        // 교점을 오브젝트 → 월드 공간으로 되돌린다(+θ).
        rec.P = Point3(
            (mCosTheta * rec.P.X()) + (mSinTheta * rec.P.Z()),
            rec.P.Y(),
            (-mSinTheta * rec.P.X()) + (mCosTheta * rec.P.Z()));

        // 법선도 같은 식으로 회전(반사/굴절 방향이 맞도록).
        rec.Normal = Vector3(
            (mCosTheta * rec.Normal.X()) + (mSinTheta * rec.Normal.Z()),
            rec.Normal.Y(),
            (-mSinTheta * rec.Normal.X()) + (mCosTheta * rec.Normal.Z()));

        return true;
    }

    __device__ Aabb BoundingBox() const override { return mBBox; }

private:
    Hittable* mObject;
    double mSinTheta;
    double mCosTheta;
    Aabb mBBox;
};

// ── 상자 만들기 ──────────────────────────────────────────────────────
// 두 대각 꼭짓점 a, b로 정의되는 직육면체(6면)를 Quad 6개로 만든다.
// 반환값은 6개 Quad를 "소유"하는 HittableList* (bOwns=true). 이 포인터를
// RotateY/Translate로 감싼 뒤 list[]에 넣으면, FreeWorld가 최상위 래퍼를
// delete할 때 소멸자 연쇄로 내부 Quad들까지 한 번씩 해제된다.
__device__ inline Hittable* MakeBox(const Point3& a, const Point3& b, Material* mat)
{
    Point3 min(fmin(a.X(), b.X()), fmin(a.Y(), b.Y()), fmin(a.Z(), b.Z()));
    Point3 max(fmax(a.X(), b.X()), fmax(a.Y(), b.Y()), fmax(a.Z(), b.Z()));

    Vector3 dx(max.X() - min.X(), 0, 0);
    Vector3 dy(0, max.Y() - min.Y(), 0);
    Vector3 dz(0, 0, max.Z() - min.Z());

    Hittable** sides = new Hittable*[6];
    sides[0] = new Quad(Point3(min.X(), min.Y(), max.Z()),  dx,  dy, mat); // front
    sides[1] = new Quad(Point3(max.X(), min.Y(), max.Z()), -dz,  dy, mat); // right
    sides[2] = new Quad(Point3(max.X(), min.Y(), min.Z()), -dx,  dy, mat); // back
    sides[3] = new Quad(Point3(min.X(), min.Y(), min.Z()),  dz,  dy, mat); // left
    sides[4] = new Quad(Point3(min.X(), max.Y(), max.Z()),  dx, -dz, mat); // top
    sides[5] = new Quad(Point3(min.X(), min.Y(), min.Z()),  dx,  dz, mat); // bottom

    return new HittableList(sides, 6, true);
}

#endif
