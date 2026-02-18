#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "Ray.h"

// === Chapter 10: 자유 시점 카메라 (Positionable Camera) ===
//
// lookfrom: 카메라 위치
// lookat: 바라보는 지점
// vup: 월드 업 벡터 (카메라 기울기 기준)
// vfov: 수직 시야각 (degrees)
// aspect: 가로/세로 비율
//
// 카메라 좌표계를 구성하는 정규직교 기저 벡터:
//   w = UnitVector(lookfrom - lookat)  → 카메라 뒤쪽 방향
//   u = UnitVector(Cross(vup, w))      → 카메라 오른쪽 방향
//   v = Cross(w, u)                    → 카메라 위쪽 방향
class Camera
{
public:
	// 기본 생성자: 이전 챕터와 동일한 고정 카메라
	__device__ Camera()
	{
		mLowerLeftCorner = Vector3(-2.0, -1.0, -1.0);
		mHorizontal = Vector3(4.0, 0.0, 0.0);
		mVertical = Vector3(0.0, 2.0, 0.0);
		mOrigin = Vector3(0.0, 0.0, 0.0);
	}

	// 자유 시점 생성자
	__device__ Camera(
		Point3 lookfrom,
		Point3 lookat,
		Vector3 vup,
		double vfov,
		double aspect)
	{
		// vfov를 라디안으로 변환하여 뷰포트 높이 계산
		double theta = vfov * 3.14159265358979323846 / 180.0;
		double halfHeight = tan(theta / 2.0);
		double halfWidth = aspect * halfHeight;

		// 카메라 좌표계 기저 벡터 (정규직교)
		mW = UnitVector(lookfrom - lookat);
		mU = UnitVector(Cross(vup, mW));
		mV = Cross(mW, mU);

		mOrigin = lookfrom;
		mLowerLeftCorner = mOrigin - halfWidth * mU - halfHeight * mV - mW;
		mHorizontal = 2.0 * halfWidth * mU;
		mVertical = 2.0 * halfHeight * mV;
	}

	__device__ Ray GetRay(double s, double t) const
	{
		return Ray(mOrigin,
			mLowerLeftCorner + s * mHorizontal + t * mVertical - mOrigin);
	}

private:
	Vector3 mOrigin;
	Vector3 mLowerLeftCorner;
	Vector3 mHorizontal;
	Vector3 mVertical;
	Vector3 mU, mV, mW;
};

#endif
