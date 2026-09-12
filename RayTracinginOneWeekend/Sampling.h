#pragma once
#ifndef SAMPLING_H
#define SAMPLING_H

#include "Vec3.h"
#include <curand_kernel.h>

// === The Rest of Your Life: 샘플링 도우미 모음 ===
//
// 원래 Material.h에 있던 무작위 방향 생성기들을 12장에서 이 파일로 옮겼다.
// Pdf.h(PDF 클래스들)와 Material.h(ScatterRecord)가 서로를 필요로 하게 되면서
// 순환 include가 생겼기 때문이다. 두 헤더 모두 이 파일만 include하면 된다.
//
//   Material.h  ->  Pdf.h  ->  Sampling.h
//                     ^            |
//                     +------------+  (순환 없음)

// 디바이스 코드에서 쓰는 π (RtWeekend.h의 Pi는 호스트 전용 헤더에 있다)
constexpr double kPi = 3.1415926535897932385;

// cuRAND를 이용한 단위 구 내부의 랜덤 점 생성 (거절법)
__device__ inline Vector3 RandomInUnitSphere(curandState* randState)
{
    Vector3 p;
    do
    {
        p = 2.0 * Vector3(curand_uniform(randState),
                           curand_uniform(randState),
                           curand_uniform(randState)) - Vector3(1.0, 1.0, 1.0);
    } while (p.LengthSquared() >= 1.0);
    return p;
}

// === The Rest of Your Life Chapter 6: 단위 구 "위"의 무작위 방향 ===
// 단위 공 안의 점을 뽑아 정규화한다. 원점에 아주 가까운 점은 정규화할 때 0으로
// 나누게 되므로 다시 뽑는다.
__device__ inline Vector3 RandomUnitVector(curandState* randState)
{
    while (true)
    {
        Vector3 p = 2.0 * Vector3(curand_uniform(randState),
                                   curand_uniform(randState),
                                   curand_uniform(randState)) - Vector3(1.0, 1.0, 1.0);
        double lengthSquared = p.LengthSquared();
        if (1e-160 < lengthSquared && lengthSquared < 1.0)
            return p / sqrt(lengthSquared);
    }
}

// === The Rest of Your Life Chapter 7: 역변환법으로 코사인 분포 방향 만들기 ===
// 거절법과 달리 루프가 없다. 난수 두 개로 바로 계산하므로 워프 안 모든 레인이
// 같은 시간에 끝난다(GPU에 유리).
//   phi = 2 pi r1,  cos(theta) = sqrt(1 - r2)  <- cos(theta)/pi 분포의 CDF를 뒤집은 것
// 여기서 나오는 방향은 "z축이 법선"인 좌표계 기준이다. 8장에서 정규직교 기저(ONB)로
// 실제 법선 방향에 맞춰 돌린다.
__device__ inline Vector3 RandomCosineDirection(curandState* randState)
{
    double r1 = curand_uniform_double(randState);
    double r2 = curand_uniform_double(randState);

    double phi = 2.0 * kPi * r1;
    double z = sqrt(1.0 - r2);     // cos(theta)
    double r = sqrt(r2);           // sin(theta)

    return Vector3(cos(phi) * r, sin(phi) * r, z);
}

// === The Rest of Your Life Chapter 12: 구를 향한 원뿔 안의 무작위 방향 ===
// 바깥의 한 점에서 구를 바라보면 구는 "원뿔" 모양의 입체각을 차지한다. 그 원뿔 안에서
// 균일하게 방향을 뽑는다(z축이 구 중심 방향인 좌표계 기준. ONB로 돌려 쓴다).
//   sin(theta_max) = R / 거리   ->   cos(theta_max) = sqrt(1 - R^2/거리^2)
//   cos(theta) = 1 + r2 * (cos(theta_max) - 1)
__device__ inline Vector3 RandomToSphere(double radius, double distanceSquared, curandState* randState)
{
    double r1 = curand_uniform_double(randState);
    double r2 = curand_uniform_double(randState);

    double cosThetaMax = sqrt(1.0 - radius * radius / distanceSquared);
    double z = 1.0 + r2 * (cosThetaMax - 1.0);

    double phi = 2.0 * kPi * r1;
    double sinTheta = sqrt(fmax(0.0, 1.0 - z * z));

    return Vector3(cos(phi) * sinTheta, sin(phi) * sinTheta, z);
}

#endif
