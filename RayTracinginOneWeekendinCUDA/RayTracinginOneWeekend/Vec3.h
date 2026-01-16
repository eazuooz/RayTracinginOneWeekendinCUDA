#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>



class Vec3
{
public:
    double E[3];

    Vec3() : E{ 0,0,0 } {}
    Vec3(double e0, double e1, double e2) : E{ e0, e1, e2 } {}

    double X() const { return E[0]; }
    double Y() const { return E[1]; }
    double Z() const { return E[2]; }

    Vec3 operator-() const { return Vec3(-E[0], -E[1], -E[2]); }
    double operator[](int i) const { return E[i]; }
    double& operator[](int i) { return E[i]; }

    Vec3& operator+=(const Vec3& v)
    {
        E[0] += v.E[0];
        E[1] += v.E[1];
        E[2] += v.E[2];
        return *this;
    }

    Vec3& operator*=(double t)
    {
        E[0] *= t;
        E[1] *= t;
        E[2] *= t;
        return *this;
    }

    Vec3& operator/=(double t)
    {
        return *this *= 1 / t;
    }

    double Length() const
    {
        return std::sqrt(LengthSquared());
    }

    double LengthSquared() const
    {
        return E[0] * E[0] + E[1] * E[1] + E[2] * E[2];
    }
};

// Point3은 Vec3의 별칭입니다. 코드의 기하학적 명확성을 위해 유용합니다.
using Point3 = Vec3;


// 벡터 유틸리티 함수들

inline std::ostream& operator<<(std::ostream& out, const Vec3& v)
{
    return out << v.E[0] << ' ' << v.E[1] << ' ' << v.E[2];
}

inline Vec3 operator+(const Vec3& u, const Vec3& v)
{
    return Vec3(u.E[0] + v.E[0], u.E[1] + v.E[1], u.E[2] + v.E[2]);
}

inline Vec3 operator-(const Vec3& u, const Vec3& v)
{
    return Vec3(u.E[0] - v.E[0], u.E[1] - v.E[1], u.E[2] - v.E[2]);
}

inline Vec3 operator*(const Vec3& u, const Vec3& v)
{
    return Vec3(u.E[0] * v.E[0], u.E[1] * v.E[1], u.E[2] * v.E[2]);
}

inline Vec3 operator*(double t, const Vec3& v)
{
    return Vec3(t * v.E[0], t * v.E[1], t * v.E[2]);
}

inline Vec3 operator*(const Vec3& v, double t)
{
    return t * v;
}

inline Vec3 operator/(const Vec3& v, double t)
{
    return (1 / t) * v;
}

inline double Dot(const Vec3& u, const Vec3& v)
{
    return u.E[0] * v.E[0]
        + u.E[1] * v.E[1]
        + u.E[2] * v.E[2];
}

inline Vec3 Cross(const Vec3& u, const Vec3& v)
{
    return Vec3(u.E[1] * v.E[2] - u.E[2] * v.E[1],
        u.E[2] * v.E[0] - u.E[0] * v.E[2],
        u.E[0] * v.E[1] - u.E[1] * v.E[0]);
}

inline Vec3 UnitVector(const Vec3& v)
{
    return v / v.Length();
}

#endif