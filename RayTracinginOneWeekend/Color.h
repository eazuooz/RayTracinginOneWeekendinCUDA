#ifndef COLOR_H
#define COLOR_H

#include "Vec3.h"
#include "Interval.h"

using Color = Vector3;

inline double LinearToGamma(double linearComponent)
{
    if (linearComponent > 0.0)
    {
        return std::sqrt(linearComponent);
    }

    return 0.0;
}


void WriteColor(std::ostream& out, const Color& pixelColor)
{
    auto r = pixelColor.X();
    auto g = pixelColor.Y();
    auto b = pixelColor.Z();

    // Apply a linear to gamma transform for gamma = 2.0
    r = LinearToGamma(r);
    g = LinearToGamma(g);
    b = LinearToGamma(b);

    // Translate the [0, 1] component values to the byte range [0, 255]
    static const Interval intensity(0.000, 0.999);

    int redByte = static_cast<int>(256.0 * intensity.Clamp(r));
    int greenByte = static_cast<int>(256.0 * intensity.Clamp(g));
    int blueByte = static_cast<int>(256.0 * intensity.Clamp(b));

    // Write out the pixel color components
    out << redByte << ' ' << greenByte << ' ' << blueByte << '\n';
}

#endif