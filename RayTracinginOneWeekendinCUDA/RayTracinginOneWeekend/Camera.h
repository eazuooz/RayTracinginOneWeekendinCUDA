#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "Hittable.h"
#include "Material.h"

class Camera
{
public:
    double aspectRatio = 1.0;               // Ratio of image width over height
    int imageWidth = 100;                   // Rendered image width in pixel count
    int samplesPerPixel = 10;               // Count of random samples for each pixel
    int maxDepth = 10;                      // Maximum number of ray bounces into scene
    double vfov = 90;                       // 수직 시야각(시야)
    Point3 lookfrom = Point3(0, 0, 0);      // 카메라가 바라보는 위치
    Point3 lookat = Point3(0, 0, -1);       // 카메라가 바라보는 점
    Vec3 vup = Vec3(0, 1, 0);               // 카메라 상대 "위쪽" 방향

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
                Color pixelColor(0.0, 0.0, 0.0);

                for (int sampleIndex = 0; sampleIndex < samplesPerPixel; sampleIndex++)
                {
                    Ray ray = GetRay(pixelIndex, scanlineIndex);
                    pixelColor += RayColor(ray, maxDepth, world);
                }

                WriteColor(std::cout, mPixelSamplesScale * pixelColor);
            }
        }

        std::clog << "\rDone.                 \n";
    }

private:
    void Initialize()
    {
        mImageHeight = static_cast<int>(imageWidth / aspectRatio);
        mImageHeight = (mImageHeight < 1) ? 1 : mImageHeight;
        mPixelSamplesScale = 1.0 / static_cast<double>(samplesPerPixel);

        mCenter = lookfrom;

        // Determine viewport dimensions
        auto focalLength = (lookfrom - lookat).Length();
        auto theta = DegreesToRadians(vfov);
        auto h = std::tan(theta / 2);
        auto viewportHeight = 2 * h * focalLength;
        auto viewportWidth = viewportHeight * (static_cast<double>(imageWidth) / mImageHeight);

        // 카메라 좌표 프레임에 대한 u,v,w 단위 기저 벡터 계산
        w = UnitVector(lookfrom - lookat);
        u = UnitVector(Cross(vup, w));
        v = Cross(w, u);

        // 뷰포트의 수평 및 수직 가장자리를 가로지르는 벡터 계산
        Vec3 viewportU = viewportWidth * u;    // 뷰포트 수평 가장자리를 가로지르는 벡터
        Vec3 viewportV = viewportHeight * -v;  // 뷰포트 수직 가장자리를 따라 내려가는 벡터


        // Calculate the horizontal and vertical delta vectors from pixel to pixel
        mPixelDeltaU = viewportU / imageWidth;
        mPixelDeltaV = viewportV / mImageHeight;

        // Calculate the location of the upper left pixel
        auto viewportUpperLeft = 
            mCenter 
            - (focalLength * w) 
            - viewportU / 2.0 
            - viewportV / 2.0;

        mPixel00Location = viewportUpperLeft + 0.5 * (mPixelDeltaU + mPixelDeltaV);
    }

    Ray GetRay(int pixelIndex, int scanlineIndex) const
    {
        // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location pixelIndex, scanlineIndex.

        auto offset = SampleSquare();

        auto pixelSample =
            mPixel00Location
            + ((pixelIndex + offset.X()) * mPixelDeltaU)
            + ((scanlineIndex + offset.Y()) * mPixelDeltaV);

        auto rayOrigin = mCenter;
        auto rayDirection = pixelSample - rayOrigin;

        return Ray(rayOrigin, rayDirection);
    }

    Vec3 SampleSquare() const
    {
        // Returns the vector to a random point in the [-.5, -.5] - [+.5, +.5] unit square
        return Vec3(RandomDouble() - 0.5, RandomDouble() - 0.5, 0.0);
    }

    Color RayColor(const Ray& ray, int depth, const Hittable& world) const
    {
        // If we've exceeded the ray bounce limit, no more light is gathered
        if (depth <= 0)
        {
            return Color(0.0, 0.0, 0.0);
        }

        HitRecord hitRecord;

        if (world.Hit(ray, Interval(0.001, Infinity), hitRecord))
        {
            Ray scattered;
            Color attenuation;

            if (hitRecord.Material->Scatter(ray, hitRecord, attenuation, scattered))
            {
                return attenuation * RayColor(scattered, depth - 1, world);
            }

            return Color(0.0, 0.0, 0.0);
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
    Vec3 u, v, w;                  // 카메라 프레임 기저 벡터

    double mPixelSamplesScale = 1.0; // Color scale factor for a sum of pixel samples
};

#endif