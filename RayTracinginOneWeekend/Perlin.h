#pragma once
#ifndef PERLIN_H
#define PERLIN_H

#include "Vec3.h"
#include <curand_kernel.h>

// === The Next Week Chapter 5: Perlin Noise (펄린 노이즈) ===
//
// 멋진 절차적(solid) 텍스처를 만들 때 흔히 쓰는 노이즈. 켄 펄린이 고안했다.
// 핵심 성질:
//   1) 재현 가능 — 3D 점을 넣으면 항상 같은 "랜덤스러운" 값을 돌려준다.
//   2) 인접한 점은 비슷한 값을 준다(백색 노이즈가 아니라 "흐릿한" 노이즈).
//   3) 단순하고 빠르다.
//
// CUDA 적용 메모:
//  - 원서의 perlin 생성자는 호스트 RNG(random_double/random_int)로 격자 데이터를
//    채운다. 우리는 월드를 CreateWorld 커널에서 만들므로, 생성자가 curandState를
//    받아 "디바이스에서" 랜덤 단위 벡터와 순열(permutation)을 만든다.
//  - Perlin 객체는 256개 벡터 + 3×256개 정수(약 9KB)를 멤버로 갖는다. 디바이스
//    new로 할당되며, 디폴트 디바이스 힙(8MB)에 충분히 들어간다.
class Perlin
{
public:
    // 격자 점마다 무작위 "단위 벡터"를 놓고(평균/최대를 격자에서 벗어나게 하는
    // 켄 펄린의 트릭), 세 축 순열 테이블을 만든다. 모두 curand로 디바이스에서.
    __device__ Perlin(curandState* randState)
    {
        for (int i = 0; i < POINT_COUNT; i++)
            mRandVec[i] = UnitVector(RandomVector(randState, -1.0, 1.0));

        GeneratePerm(mPermX, randState);
        GeneratePerm(mPermY, randState);
        GeneratePerm(mPermZ, randState);
    }

    // 점 p에서의 노이즈 값(에르미트 스무딩 + 격자 벡터 내적 보간). [-1,1] 범위.
    __device__ double Noise(const Point3& p) const
    {
        double u = p.X() - floor(p.X());
        double v = p.Y() - floor(p.Y());
        double w = p.Z() - floor(p.Z());

        int i = int(floor(p.X()));
        int j = int(floor(p.Y()));
        int k = int(floor(p.Z()));
        Vector3 c[2][2][2];

        // 둘러싼 8개 격자 점의 랜덤 벡터를 세 축 순열의 XOR 해싱으로 가져온다.
        for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++)
                    c[di][dj][dk] = mRandVec[
                        mPermX[(i + di) & 255] ^
                        mPermY[(j + dj) & 255] ^
                        mPermZ[(k + dk) & 255]
                    ];

        return PerlinInterp(c, u, v, w);
    }

    // 난류(turbulence): 주파수를 2배씩 올리며 노이즈를 가중 합산한 복합 노이즈.
    // 대리석 무늬 등에서 위상(phase)을 흔드는 데 쓴다.
    __device__ double Turb(const Point3& p, int depth) const
    {
        double accum = 0.0;
        Point3 tempP = p;
        double weight = 1.0;

        for (int i = 0; i < depth; i++)
        {
            accum += weight * Noise(tempP);
            weight *= 0.5;
            tempP *= 2.0;
        }

        return fabs(accum);
    }

private:
    static const int POINT_COUNT = 256;
    Vector3 mRandVec[POINT_COUNT];
    int mPermX[POINT_COUNT];
    int mPermY[POINT_COUNT];
    int mPermZ[POINT_COUNT];

    // [min,max]^3 안의 랜덤 벡터(완전 균일일 필요는 없다).
    __device__ static Vector3 RandomVector(curandState* s, double min, double max)
    {
        double range = max - min;
        return Vector3(min + range * curand_uniform(s),
                       min + range * curand_uniform(s),
                       min + range * curand_uniform(s));
    }

    __device__ static void GeneratePerm(int* p, curandState* s)
    {
        for (int i = 0; i < POINT_COUNT; i++)
            p[i] = i;
        Permute(p, POINT_COUNT, s);
    }

    // Fisher–Yates 셔플. 원서 random_int(0,i)를 curand로 대체.
    __device__ static void Permute(int* p, int n, curandState* s)
    {
        for (int i = n - 1; i > 0; i--)
        {
            // [0, i] 정수. curand_uniform은 (0,1]이라 1.0일 때만 i 초과 → 클램프.
            int target = int(curand_uniform(s) * (i + 1));
            if (target > i) target = i;

            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    // 격자 벡터 보간: 각 격자 벡터와 (점→격자) 가중 벡터의 내적을 삼선형 가중합.
    // u/v/w에 에르미트 3차(u*u*(3-2u))를 적용해 마하 밴드를 줄인다.
    __device__ static double PerlinInterp(const Vector3 c[2][2][2], double u, double v, double w)
    {
        double uu = u * u * (3.0 - 2.0 * u);
        double vv = v * v * (3.0 - 2.0 * v);
        double ww = w * w * (3.0 - 2.0 * w);
        double accum = 0.0;

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                {
                    Vector3 weightV(u - i, v - j, w - k);
                    accum += (i * uu + (1 - i) * (1 - uu))
                           * (j * vv + (1 - j) * (1 - vv))
                           * (k * ww + (1 - k) * (1 - ww))
                           * Dot(c[i][j][k], weightV);
                }

        return accum;
    }
};

#endif
