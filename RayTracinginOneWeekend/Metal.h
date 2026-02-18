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
		curandState* randState) const override
	{
		Vector3 reflected = Reflect(UnitVector(rayIn.Direction()), rec.Normal);
		scattered = Ray(rec.P, reflected + mFuzz * RandomInUnitSphere(randState));
		attenuation = mAlbedo;
		return (Dot(scattered.Direction(), rec.Normal) > 0.0);
	}

private:
	Color mAlbedo;
	double mFuzz;
};

#endif
