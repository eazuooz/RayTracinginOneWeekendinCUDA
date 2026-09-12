#pragma once
#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "Hittable.h"

// GPU에서는 std::vector를 사용할 수 없으므로
// Hittable 포인터 배열(Hittable**)을 직접 관리한다
class HittableList : public Hittable
{
public:
    __device__ HittableList() {}

    // bOwns: 이 리스트가 list 배열과 그 안의 Hittable들을 소유하는지 여부.
    //   - false(기본): 외부가 메모리를 관리(기존 사용처와 호환).
    //   - true: 소멸자에서 자식들과 배열을 직접 해제한다.
    // === The Next Week Chapter 8: Instances ===
    // MakeBox()가 6개 Quad와 그 포인터 배열을 소유한 채로 이 리스트를 만들고,
    // Translate/RotateY로 감싸 list[]에 넣는다. FreeWorld가 최상위 Translate를
    // delete하면 소멸자 연쇄로 RotateY→HittableList(소유)→Quad 6개까지 해제된다.
    __device__ HittableList(Hittable** list, int count, bool bOwns = false)
        : mList(list), mCount(count), mbOwns(bOwns)
    {
        // 자식들의 경계 상자를 합쳐 리스트 전체의 경계 상자를 구한다
        for (int i = 0; i < count; i++)
            mBBox = Aabb(mBBox, list[i]->BoundingBox());
    }

    __device__ ~HittableList() override
    {
        if (mbOwns)
        {
            for (int i = 0; i < mCount; i++)
                delete mList[i];
            delete[] mList;
        }
    }

    __device__ bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& hitRecord,
        curandState* randState) const override
    {
        HitRecord tempRecord;
        bool bHitAnything = false;
        double closestSoFar = tMax;

        for (int i = 0; i < mCount; i++)
        {
            if (mList[i]->Hit(ray, tMin, closestSoFar, tempRecord, randState))
            {
                bHitAnything = true;
                closestSoFar = tempRecord.T;
                hitRecord = tempRecord;
            }
        }

        return bHitAnything;
    }

    __device__ Aabb BoundingBox() const override { return mBBox; }

    // === The Rest of Your Life Chapter 12: 여러 물체를 함께 샘플링 ===
    // 광원이 여럿일 때(예: 천장 광원 + 유리 구) 이 리스트 자체를 샘플링 대상으로 쓴다.
    //   PdfValue : 각 물체 밀도의 평균 (각각을 1/n 확률로 고르므로)
    //   Random   : 물체 하나를 무작위로 골라 그쪽으로 방향을 뽑는다
    __device__ double PdfValue(
        const Point3& origin, const Vector3& direction, curandState* randState) const override
    {
        if (mCount <= 0)
            return 0.0;

        double weight = 1.0 / double(mCount);
        double sum = 0.0;
        for (int i = 0; i < mCount; i++)
            sum += weight * mList[i]->PdfValue(origin, direction, randState);

        return sum;
    }

    __device__ Vector3 Random(const Point3& origin, curandState* randState) const override
    {
        if (mCount <= 0)
            return Vector3(1.0, 0.0, 0.0);

        // curand_uniform은 (0,1]이라 그대로 곱하면 mCount가 나올 수 있다 → 마지막 칸으로 클램프.
        int index = int(curand_uniform(randState) * float(mCount));
        if (index >= mCount)
            index = mCount - 1;

        return mList[index]->Random(origin, randState);
    }

private:
    Hittable** mList;
    int mCount = 0;
    Aabb mBBox;
    bool mbOwns = false;
};

#endif
