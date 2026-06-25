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
    __device__ HittableList(Hittable** list, int count)
        : mList(list), mCount(count)
    {
        // 자식들의 경계 상자를 합쳐 리스트 전체의 경계 상자를 구한다
        for (int i = 0; i < count; i++)
            mBBox = Aabb(mBBox, list[i]->BoundingBox());
    }

    __device__ bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& hitRecord) const override
    {
        HitRecord tempRecord;
        bool bHitAnything = false;
        double closestSoFar = tMax;

        for (int i = 0; i < mCount; i++)
        {
            if (mList[i]->Hit(ray, tMin, closestSoFar, tempRecord))
            {
                bHitAnything = true;
                closestSoFar = tempRecord.T;
                hitRecord = tempRecord;
            }
        }

        return bHitAnything;
    }

    __device__ Aabb BoundingBox() const override { return mBBox; }

private:
    Hittable** mList;
    int mCount;
    Aabb mBBox;
};

#endif
