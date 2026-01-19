#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "Hittable.h"

class Camera
{
public:
    double aspectRatio = 1.0;   // Ratio of image width over height
    int imageWidth = 100;       // Rendered image width in pixel count

    void Render(const Hittable& world)
    {
        Initialize();

        std::cout << "P3\n" << imageWidth << ' ' << mImageHeight << "\n255\n";

        for (int scanlineIndex = 0; scanlineIndex < mImageHeight; scanlineIndex++)
        {
            std::clog
                << "\rScanlines remaining: "
                << (mImageHeight - scanlineIndex)
                << ' '
                << std::flush;

            for (int pixelIndex = 0; pixelIndex < imageWidth; pixelIndex++)
            {
                auto pixelCenter =
                    mPixel00Location
                    + (pixelIndex * mPixelDeltaU)
                    + (scanlineIndex * mPixelDeltaV);

                auto rayDirection = pixelCenter - mCenter;
                Ray ray(mCenter, rayDirection);

                Color pixelColor = RayColor(ray, world);
                WriteColor(std::cout, pixelColor);
            }
        }

        std::clog << "\rDone.                 \n";
    }

private:
    void Initialize()
    {
        mImageHeight = static_cast<int>(imageWidth / aspectRatio);
        mImageHeight = (mImageHeight < 1) ? 1 : mImageHeight;

        mCenter = Point3(0.0, 0.0, 0.0);

        // Determine viewport dimensions
        auto focalLength = 1.0;
        auto viewportHeight = 2.0;
        auto viewportWidth = viewportHeight * (static_cast<double>(imageWidth) / mImageHeight);

        // Calculate the vectors across the horizontal and down the vertical viewport edges
        auto viewportU = Vec3(viewportWidth, 0.0, 0.0);
        auto viewportV = Vec3(0.0, -viewportHeight, 0.0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel
        mPixelDeltaU = viewportU / imageWidth;
        mPixelDeltaV = viewportV / mImageHeight;

        // Calculate the location of the upper left pixel
        auto viewportUpperLeft =
            mCenter
            - Vec3(0.0, 0.0, focalLength)
            - viewportU / 2.0
            - viewportV / 2.0;

        mPixel00Location = viewportUpperLeft + 0.5 * (mPixelDeltaU + mPixelDeltaV);
    }

    Color RayColor(const Ray& ray, const Hittable& world) const
    {
        HitRecord hitRecord;

        if (world.Hit(ray, Interval(0.0, Infinity), hitRecord))
        {
            return 0.5 * (hitRecord.Normal + Color(1.0, 1.0, 1.0));
        }

        Vec3 unitDirection = UnitVector(ray.Direction());
        auto a = 0.5 * (unitDirection.Y() + 1.0);

        return (1.0 - a) * Color(1.0, 1.0, 1.0)
            + a * Color(0.5, 0.7, 1.0);
    }

private:
    int mImageHeight = 0;        // Rendered image height
    Point3 mCenter;               // Camera center
    Point3 mPixel00Location;      // Location of pixel 0, 0
    Vec3 mPixelDeltaU;           // Offset to pixel to the right
    Vec3 mPixelDeltaV;           // Offset to pixel below
};
#endif