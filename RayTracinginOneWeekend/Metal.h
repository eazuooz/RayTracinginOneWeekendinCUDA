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

	// === 3권 12장: ScatterRecord ===
	// 거울 반사는 방향이 (거의) 하나로 정해지는 델타 분포라 밀도로 나눌 수 없다.
	// bSkipPdf = true 로 "PDF를 쓰지 말고 이 레이를 그대로 따라가라"고 알려준다.
	__device__ bool Scatter(
		const Ray& rayIn,
		const HitRecord& rec,
		ScatterRecord& srec,
		curandState* randState) const override
	{
		Vector3 reflected = Reflect(UnitVector(rayIn.Direction()), rec.Normal);
		// 산란 레이는 입력 레이의 time을 그대로 물려받는다
		Ray reflectedRay(rec.P, reflected + mFuzz * RandomInUnitSphere(randState), rayIn.Time());

		srec.Attenuation = mAlbedo;
		srec.PdfPtr = nullptr;
		srec.bSkipPdf = true;
		srec.SkipPdfRay = reflectedRay;

		// 표면 아래로 반사되면(퍼지가 큰 경우) 흡수로 처리한다.
		return (Dot(reflectedRay.Direction(), rec.Normal) > 0.0);
	}

private:
	Color mAlbedo;
	double mFuzz;
};

#endif
