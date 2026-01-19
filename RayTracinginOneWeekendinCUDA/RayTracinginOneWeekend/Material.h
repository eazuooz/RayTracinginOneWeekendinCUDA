#pragma once

#include "Hittable.h"

class Material
{
public:
    virtual ~Material() = default;

    virtual bool Scatter(const Ray& rayIn, const HitRecord& hitRecord, Color& attenuation, Ray& scattered) const
    {
        return false;
    }
};

class Lambertian : public Material
{
public:
    explicit Lambertian(const Color& albedo)
        : mAlbedo(albedo)
    {
    }

    bool Scatter(const Ray& rayIn, const HitRecord& hitRecord, Color& attenuation,
        Ray& scattered) const override
    {
        Vec3 scatterDirection = hitRecord.Normal + RandomUnitVector();

        // Catch degenerate scatter direction
        if (scatterDirection.NearZero())
        {
            scatterDirection = hitRecord.Normal;
        }

        scattered = Ray(hitRecord.P, scatterDirection);
        attenuation = mAlbedo;

        return true;
    }

private:
    Color mAlbedo;
};