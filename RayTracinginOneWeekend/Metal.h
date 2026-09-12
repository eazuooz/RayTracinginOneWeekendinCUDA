#pragma once
#ifndef METAL_H
#define METAL_H

#include "Material.h"

// 금속 재질 (Metal)
// 입사 레이를 법선 기준으로 반사하며, fuzz로 흐릿한 정도를 조절
class Metal : public Material
{
public:
	__device__ Metal(const Color& albedo, double fuzz)
		: mAlbedo(albedo)
		, mFuzz(fuzz < 1.0 ? fuzz : 1.0)
	{
	}

	__device__ bool Scatter(
		const Ray& rayIn,
		const HitRecord& rec,
		Color& attenuation,
		Ray& scattered,
		double& pdf,
		curandState* randState) const override
	{
		Vector3 reflected = Reflect(UnitVector(rayIn.Direction()), rec.Normal);
		// 산란 레이는 입력 레이의 time을 그대로 물려받는다
		scattered = Ray(rec.P, reflected + mFuzz * RandomInUnitSphere(randState), rayIn.Time());
		attenuation = mAlbedo;
		// === 3권 8장 ===
		// 거울 반사는 방향이 (거의) 하나로 정해지는 델타 분포라 밀도로 나눌 수 없다.
		// pdf = 0 은 "PDF 없음"을 뜻하고, RayColor가 f/p 가중 없이 감쇠만 곱한다.
		pdf = 0.0;
		return (Dot(scattered.Direction(), rec.Normal) > 0.0);
	}

private:
	Color mAlbedo;
	double mFuzz;
};

#endif
