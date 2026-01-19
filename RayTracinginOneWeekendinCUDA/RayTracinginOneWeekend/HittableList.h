#pragma once
#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "RtWeekend.h"
#include "Hittable.h"

#include <vector>



class HittableList : public Hittable
{
public:
    HittableList() = default;

    explicit HittableList(const std::shared_ptr<Hittable>& object)
    {
        Add(object);
    }

    void Clear()
    {
        mObjects.clear();
    }

    void Add(const std::shared_ptr<Hittable>& object)
    {
        mObjects.push_back(object);
    }

    bool Hit(
        const Ray& ray,
        const Interval& rayT,
        HitRecord& hitRecord
    ) const override
    {
        HitRecord temporaryHitRecord;
        bool bHitAnything = false;
        auto closestSoFar = rayT.Max;

        for (const auto& object : mObjects)
        {
            Interval currentRayT(rayT.Min, closestSoFar);

            if (object->Hit(ray, currentRayT, temporaryHitRecord))
            {
                bHitAnything = true;
                closestSoFar = temporaryHitRecord.T;
                hitRecord = temporaryHitRecord;
            }
        }

        return bHitAnything;
    }

private:
    std::vector<std::shared_ptr<Hittable>> mObjects;
};

#endif
