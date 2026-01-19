#pragma once
#include "Material.h"

class Metal : public Material
{
public:
    explicit Metal(const Color& albedo)
        : mAlbedo(albedo)
    {
    }

    bool Scatter(const Ray& rayIn, const HitRecord& hitRecord, Color& attenuation, Ray& scattered) const override
    {
        Vec3 reflected = Reflect(rayIn.Direction(), hitRecord.Normal);

        scattered = Ray(hitRecord.P, reflected);
        attenuation = mAlbedo;

        return true;
    }

private:
    Color mAlbedo;
};