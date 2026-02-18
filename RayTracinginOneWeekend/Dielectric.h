#pragma once
#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "Material.h"

// 유전체 재질 (Dielectric) - 유리, 물 등 투명한 재질
// 스넬의 법칙으로 굴절하고, 전반사(total internal reflection) 시 반사
// Schlick 근사로 반사 확률을 계산
class Dielectric : public Material
{
public:
	__device__ Dielectric(double refractionIndex)
		: mRefractionIndex(refractionIndex)
	{
	}

	__device__ bool Scatter(
		const Ray& rayIn,
		const HitRecord& rec,
		Color& attenuation,
		Ray& scattered,
		curandState* randState) const override
	{
		// 유전체는 빛을 흡수하지 않음
		attenuation = Color(1.0, 1.0, 1.0);

		// bFrontFace로 굴절률 비율 결정
		// 앞면: 공기(1.0) → 유리(mRefractionIndex)
		// 뒷면: 유리(mRefractionIndex) → 공기(1.0)
		double refractionRatio = rec.bFrontFace ? (1.0 / mRefractionIndex) : mRefractionIndex;

		Vector3 unitDirection = UnitVector(rayIn.Direction());

		// 전반사 판정: sinTheta > 1.0이면 굴절 불가
		double cosTheta = fmin(Dot(-unitDirection, rec.Normal), 1.0);
		double sinTheta = sqrt(1.0 - cosTheta * cosTheta);
		bool cannotRefract = refractionRatio * sinTheta > 1.0;

		Vector3 direction;
		if (cannotRefract || Reflectance(cosTheta, refractionRatio) > (double)curand_uniform(randState))
		{
			// 전반사 또는 Schlick 확률에 의한 반사
			direction = Reflect(unitDirection, rec.Normal);
		}
		else
		{
			// 스넬의 법칙에 의한 굴절
			direction = Refract(unitDirection, rec.Normal, refractionRatio);
		}

		scattered = Ray(rec.P, direction);
		return true;
	}

private:
	// 굴절률 (IOR). 유리 ≈ 1.5, 물 ≈ 1.33
	double mRefractionIndex;

	// Schlick 근사: 입사각에 따른 반사 확률 계산
	// 비스듬한 각도에서 반사가 강해지는 프레넬 효과를 근사
	__device__ static double Reflectance(double cosine, double refractionIndex)
	{
		double r0 = (1.0 - refractionIndex) / (1.0 + refractionIndex);
		r0 = r0 * r0;
		return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
	}
};

#endif
