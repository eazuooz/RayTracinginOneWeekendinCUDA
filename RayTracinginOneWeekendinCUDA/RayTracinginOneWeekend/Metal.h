#pragma once
#include "Material.h"

class Metal : public Material
{
public:
    explicit Metal(const Color& albedo, double fuzz)
        : mAlbedo(albedo)
        , mFuzz(fuzz < 1 ? fuzz : 1)
    {
    }

    bool Scatter(const Ray& rayIn, const HitRecord& hitRecord, Color& attenuation, Ray& scattered) const override
    {
        Vec3 reflected = Reflect(rayIn.Direction(), hitRecord.Normal);
		reflected += UnitVector(reflected) + (mFuzz * RandomUnitVector());
        scattered = Ray(hitRecord.P, reflected);
        attenuation = mAlbedo;

        return (Dot(scattered.Direction(), hitRecord.Normal) > 0);
    }

private:
    Color mAlbedo;
    double mFuzz;
};

class Dielectric : public Material
{
public:
    explicit Dielectric(double refractionIndex)
        : mRefractionIndex(refractionIndex)
    {
    }

    bool Scatter(
        const Ray& rayIn,
        const HitRecord& hitRecord,
        Color& attenuation,
        Ray& scattered
    ) const override
    {
        attenuation = Color(1.0, 1.0, 1.0);

        const double etaInOverEtaOut =
            hitRecord.bFrontFace ? (1.0 / mRefractionIndex) : mRefractionIndex;

        const Vec3 unitDirection = UnitVector(rayIn.Direction());
        const Vec3 refracted = Refract(unitDirection, hitRecord.Normal, etaInOverEtaOut);

        scattered = Ray(hitRecord.P, refracted);

        return true;
    }

private:
    // Refraction index (IOR). For air/vacuum -> material, typical glass is ~1.5.
    double mRefractionIndex = 1.0;
};