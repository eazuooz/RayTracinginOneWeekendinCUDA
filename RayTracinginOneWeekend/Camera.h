#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "Ray.h"
#include <curand_kernel.h>

// 렌즈 위의 랜덤 점 생성 (단위 원반 내부)
// 피사계 심도에서 레이 원점을 랜덤하게 흩뜨리는 데 사용
__device__ inline Vector3 RandomInUnitDisk(curandState* randState)
{
	Vector3 p;
	do
	{
		p = 2.0 * Vector3(curand_uniform(randState), curand_uniform(randState), 0.0)
			- Vector3(1.0, 1.0, 0.0);
	} while (Dot(p, p) >= 1.0);
	return p;
}

// === Chapter 11: 피사계 심도 (Defocus Blur / Depth of Field) ===
//
// 실제 카메라의 렌즈를 시뮬레이션한다.
// aperture(조리개): 클수록 배경 흐림이 강해짐
// focusDist(초점 거리): 이 거리의 물체만 선명하게 보임
//
// 레이 원점을 렌즈 원반 위의 랜덤 점으로 오프셋하여
// 초점 평면 위의 물체만 선명하고 나머지는 흐릿하게 만든다.
class Camera
{
public:
	// 피사계 심도 지원 생성자
	// aperture: 조리개 크기 (0이면 핀홀 카메라 = 모든 것이 선명)
	// focusDist: 초점 거리 (이 거리의 물체가 선명)
	__device__ Camera(
		Point3 lookfrom,
		Point3 lookat,
		Vector3 vup,
		double vfov,
		double aspect,
		double aperture,
		double focusDist)
	{
		mLensRadius = aperture / 2.0;

		// vfov를 라디안으로 변환하여 뷰포트 높이 계산
		double theta = vfov * 3.14159265358979323846 / 180.0;
		double halfHeight = tan(theta / 2.0);
		double halfWidth = aspect * halfHeight;

		// 카메라 좌표계 기저 벡터 (정규직교)
		mW = UnitVector(lookfrom - lookat);
		mU = UnitVector(Cross(vup, mW));
		mV = Cross(mW, mU);

		mOrigin = lookfrom;
		// 뷰포트를 focusDist만큼 스케일하여 초점 평면에 배치
		mLowerLeftCorner = mOrigin
			- halfWidth * focusDist * mU
			- halfHeight * focusDist * mV
			- focusDist * mW;
		mHorizontal = 2.0 * halfWidth * focusDist * mU;
		mVertical = 2.0 * halfHeight * focusDist * mV;
	}

	// 피사계 심도가 적용된 레이 생성
	// 렌즈 위의 랜덤 점에서 초점 평면의 목표 지점으로 레이를 쏜다
	__device__ Ray GetRay(double s, double t, curandState* randState) const
	{
		Vector3 rd = mLensRadius * RandomInUnitDisk(randState);
		Vector3 offset = mU * rd.X() + mV * rd.Y();
		return Ray(
			mOrigin + offset,
			mLowerLeftCorner + s * mHorizontal + t * mVertical - mOrigin - offset);
	}

private:
	Vector3 mOrigin;
	Vector3 mLowerLeftCorner;
	Vector3 mHorizontal;
	Vector3 mVertical;
	Vector3 mU, mV, mW;
	double mLensRadius;
};

#endif
