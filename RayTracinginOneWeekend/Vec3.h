#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

#include "cuda_runtime.h"


struct Vector3
{
    double E[3];

    __host__ __device__ Vector3() : E{ 0,0,0 } {}
    __host__ __device__ Vector3(double e0, double e1, double e2) : E{ e0, e1, e2 } {}

    __host__ __device__ double X() const { return E[0]; }
    __host__ __device__ double Y() const { return E[1]; }
    __host__ __device__ double Z() const { return E[2]; }

    __host__ __device__ Vector3 operator-() const { return Vector3(-E[0], -E[1], -E[2]); }
    __host__ __device__ double operator[](int i) const { return E[i]; }
    __host__ __device__ double& operator[](int i) { return E[i]; }

    __host__ __device__ Vector3& operator+=(const Vector3& v)
    {
        E[0] += v.E[0];
        E[1] += v.E[1];
        E[2] += v.E[2];
        return *this;
    }

    __host__ __device__ Vector3& operator*=(double t)
    {
        E[0] *= t;
        E[1] *= t;
        E[2] *= t;
        return *this;
    }

    __host__ __device__ Vector3& operator/=(double t)
    {
        return *this *= 1 / t;
    }

    __host__ __device__ double Length() const
    {
        return sqrt(LengthSquared());
    }

    __host__ __device__ double LengthSquared() const
    {
        return E[0] * E[0] + E[1] * E[1] + E[2] * E[2];
    }

    __host__ __device__ bool NearZero() const
    {
        auto threshold = 1e-8;

        return (fabs(E[0]) < threshold)
            && (fabs(E[1]) < threshold)
            && (fabs(E[2]) < threshold);
    }
};
typedef Vector3 Vec3;

using Point3 = Vector3;
using Color = Vector3;


// 벡터 유틸리티 함수들

inline std::ostream& operator<<(std::ostream& out, const Vector3& v)
{
    return out << v.E[0] << ' ' << v.E[1] << ' ' << v.E[2];
}

__host__ __device__ inline Vector3 operator+(const Vector3& u, const Vector3& v)
{
    return Vector3(u.E[0] + v.E[0], u.E[1] + v.E[1], u.E[2] + v.E[2]);
}

__host__ __device__ inline Vector3 operator-(const Vector3& u, const Vector3& v)
{
    return Vector3(u.E[0] - v.E[0], u.E[1] - v.E[1], u.E[2] - v.E[2]);
}

__host__ __device__ inline Vector3 operator*(const Vector3& u, const Vector3& v)
{
    return Vector3(u.E[0] * v.E[0], u.E[1] * v.E[1], u.E[2] * v.E[2]);
}

__host__ __device__ inline Vector3 operator*(double t, const Vector3& v)
{
    return Vector3(t * v.E[0], t * v.E[1], t * v.E[2]);
}

__host__ __device__ inline Vector3 operator*(const Vector3& v, double t)
{
    return t * v;
}

__host__ __device__ inline Vector3 operator/(const Vector3& v, double t)
{
    return (1 / t) * v;
}

__host__ __device__ inline double Dot(const Vector3& u, const Vector3& v)
{
    return u.E[0] * v.E[0]
        + u.E[1] * v.E[1]
        + u.E[2] * v.E[2];
}

__host__ __device__ inline Vector3 Cross(const Vector3& u, const Vector3& v)
{
    return Vector3(u.E[1] * v.E[2] - u.E[2] * v.E[1],
        u.E[2] * v.E[0] - u.E[0] * v.E[2],
        u.E[0] * v.E[1] - u.E[1] * v.E[0]);
}

__host__ __device__ inline Vector3 UnitVector(const Vector3& v)
{
    return v / v.Length();
}

__host__ __device__ inline Vector3 Reflect(const Vector3& v, const Vector3& n)
{
    return v - 2.0 * Dot(v, n) * n;
}

__host__ __device__ inline Vector3 Refract(const Vector3& uv, const Vector3& n, double etaInOverEtaOut)
{
    const double cosTheta = fmin(Dot(-uv, n), 1.0);

    const Vector3 refractPerpendicular = etaInOverEtaOut * (uv + cosTheta * n);
    const Vector3 refractParallel =
        -sqrt(fabs(1.0 - refractPerpendicular.LengthSquared())) * n;

    return refractPerpendicular + refractParallel;
}

#endif
