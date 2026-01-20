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
    static double Reflectance(double cosine, double refractionIndex)
    {
        // Schlick의 반사율 근사 사용
        auto r0 = (1.0 - refractionIndex) / (1.0 + refractionIndex);
        r0 = r0 * r0;
        return r0 + (1.0 - r0) * std::pow((1.0 - cosine), 5);
    }

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

        const double refractionRatio =
            hitRecord.bFrontFace ? (1.0 / mRefractionIndex) : mRefractionIndex;

        const Vec3 unitDirection = UnitVector(rayIn.Direction());

        const double cosTheta =
            std::fmin(Dot(-unitDirection, hitRecord.Normal), 1.0);

        const double sinTheta =
            std::sqrt(1.0 - cosTheta * cosTheta);

        const bool cannotRefract =
            refractionRatio * sinTheta > 1.0;

        Vec3 direction;

        if (cannotRefract || Reflectance(cosTheta, refractionRatio) > RandomDouble())
        {
            direction = Reflect(unitDirection, hitRecord.Normal);
        }
        else
        {
            direction = Refract(unitDirection, hitRecord.Normal, refractionRatio);
        }

        scattered = Ray(hitRecord.P, direction);

        return true;
    }

private:
    // Refraction index (IOR). For air/vacuum -> material, typical glass is ~1.5.
    double mRefractionIndex = 1.0;
};