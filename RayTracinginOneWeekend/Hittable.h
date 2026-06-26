#pragma once
#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.h"
#include "AABB.h"

class Material;

struct HitRecord
{
    Point3 P;
    Vector3 Normal;
    double T;
    // 표면 텍스처 좌표 (u, v). 레이가 맞은 지점이 2D 텍스처/이미지 상에서
    // 어디에 대응되는지를 [0,1] x [0,1] 범위로 나타낸다. 각 Hittable의 Hit()이
    // 채워 넣고, Material(Lambertian)이 이 좌표로 텍스처 색을 조회한다.
    double U;
    double V;
    bool bFrontFace;
    Material* MaterialPtr;

    // 레이 방향과 외부 법선으로 앞면/뒷면 판정
    // 법선은 항상 레이 반대 방향(표면 바깥쪽)을 가리키도록 설정
    __device__ void SetFaceNormal(const Ray& ray, const Vector3& outwardNormal)
    {
        bFrontFace = Dot(ray.Direction(), outwardNormal) < 0.0;
        Normal = bFrontFace ? outwardNormal : -outwardNormal;
    }
};

class Hittable
{
public:
    __device__ virtual bool Hit(
        const Ray& ray,
        double tMin,
        double tMax,
        HitRecord& hitRecord) const = 0;

    // 이 객체를 감싸는 AABB를 반환한다.
    // BVH가 객체를 계층적으로 묶으려면 모든 Hittable이 자신의 경계 상자를
    // 알려줄 수 있어야 한다. 움직이는 객체는 운동 구간 전체(time0~time1)를
    // 포함하는 상자를 반환해야 한다.
    __device__ virtual Aabb BoundingBox() const = 0;

    // BVH 내부 노드 여부. BvhNode가 반복(iterative) 순회 중 자식이 내부 노드인지
    // 잎(primitive)인지 구분하는 데 쓴다. 기본값은 false(잎/일반 오브젝트).
    __device__ virtual bool IsBvhNode() const { return false; }
};

#endif
