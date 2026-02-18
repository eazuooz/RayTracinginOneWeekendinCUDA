#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "Ray.h"

// GPU에서 사용하는 간소화된 카메라
// 뷰포트의 lowerLeftCorner, horizontal, vertical, origin으로 레이를 생성
class Camera
{
public:
    __device__ Camera()
    {
        mLowerLeftCorner = Vector3(-2.0, -1.0, -1.0);
        mHorizontal = Vector3(4.0, 0.0, 0.0);
        mVertical = Vector3(0.0, 2.0, 0.0);
        mOrigin = Vector3(0.0, 0.0, 0.0);
    }

    __device__ Ray GetRay(double u, double v) const
    {
        return Ray(mOrigin,
            mLowerLeftCorner + u * mHorizontal + v * mVertical - mOrigin);
    }

private:
    Vector3 mOrigin;
    Vector3 mLowerLeftCorner;
    Vector3 mHorizontal;
    Vector3 mVertical;
};

#endif
