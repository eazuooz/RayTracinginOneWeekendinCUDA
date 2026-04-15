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
// === The Next Week: 모션 블러 (Motion Blur) ===
//
// 실제 카메라의 셔터 개방 시간을 시뮬레이션한다.
// time0~time1 구간 동안 셔터가 열려 있으며,
// 각 레이는 그 구간 내 랜덤한 시각을 가진다.
// MovingSphere는 레이의 time을 읽어 그 순간의 위치를 계산한다.
//
// aperture(조리개): 클수록 피사계 심도 흐림이 강해짐
// focusDist(초점 거리): 이 거리의 물체만 선명하게 보임
class Camera
{
public:
	// 피사계 심도 + 모션 블러 지원 생성자
	// time0, time1: 셔터 개방 시작/종료 시각 (기본값 0.0)
	__device__ Camera(
		Point3 lookfrom,
		Point3 lookat,
		Vector3 vup,
		double vfov,
		double aspect,
		double aperture,
		double focusDist,
		double time0 = 0.0,
		double time1 = 0.0)
	{
		mTime0 = time0;
		mTime1 = time1;
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

	// 피사계 심도 + 모션 블러가 적용된 레이 생성
	// 렌즈 위의 랜덤 점에서 초점 평면의 목표 지점으로 레이를 쏘고,
	// [time0, time1] 구간에서 랜덤한 발사 시각을 레이에 부여한다.
	__device__ Ray GetRay(double s, double t, curandState* randState) const
	{
		Vector3 rd = mLensRadius * RandomInUnitDisk(randState);
		Vector3 offset = mU * rd.X() + mV * rd.Y();
		double time = mTime0 + curand_uniform(randState) * (mTime1 - mTime0);
		return Ray(
			mOrigin + offset,
			mLowerLeftCorner + s * mHorizontal + t * mVertical - mOrigin - offset,
			time);
	}

private:
	Vector3 mOrigin;
	Vector3 mLowerLeftCorner;
	Vector3 mHorizontal;
	Vector3 mVertical;
	Vector3 mU, mV, mW;
	double mLensRadius;
	double mTime0;  // 셔터 개방 시각
	double mTime1;  // 셔터 닫힘 시각
};

#endif
