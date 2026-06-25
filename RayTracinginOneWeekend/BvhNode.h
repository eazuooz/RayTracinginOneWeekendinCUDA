#pragma once
#ifndef BVH_NODE_H
#define BVH_NODE_H

#include "Hittable.h"
#include "AABB.h"

// === The Next Week Chapter 3: 경계 볼륨 계층 (Bounding Volume Hierarchy) ===
//
// BVH도 하나의 Hittable이다. "이 레이가 너에게 맞았니?" 라는 질의에 답하는
// 컨테이너이며, 자식(left/right)을 따라 내려가며 검사한다.
//
//   if (레이가 이 노드의 상자에 맞음)
//       hit_left  = 왼쪽 자식 검사
//       hit_right = 오른쪽 자식 검사
//       return hit_left || hit_right
//   else
//       return false   // 상자에 안 맞으면 자식들도 검사할 필요 없음
//
// (원서는 위처럼 재귀로 순회하지만, GPU 스택 부담 때문에 이 구현의 Hit는
//  아래에서 명시적 스택을 쓰는 반복(iterative) 순회로 작성했다. 자세한 이유는
//  Hit 함수 위 주석 참고.)
//
// ─── CUDA 적용 시 원서(CPU)와 달라진 점 ───────────────────────────────
// 원서는 호스트에서 std::vector / std::sort / make_shared(shared_ptr)로
// 트리를 만든다. 하지만 이 프로젝트는 월드를 CreateWorld 커널 안에서
// 디바이스 new 로 직접 생성한다. 디바이스에서는 STL을 쓸 수 없으므로:
//   1) std::vector  → Hittable** 포인터 배열(원본 list 재사용)
//   2) std::sort    → 아래 DeviceSort (단일 스레드 삽입 정렬)
//   3) make_shared  → 디바이스 new (BvhNode* 직접 할당)
//   4) shared_ptr가 해주던 자동 해제 → 노드 레지스트리(nodeStorage)로 대체
//
// ※ shared_ptr 부재로 인한 더블 프리 주의:
//   object_span == 1 인 경우 원서처럼 left = right = 같은 객체로 둔다.
//   즉 같은 잎(sphere) 포인터가 두 번 참조될 수 있다. 따라서 트리를 따라
//   재귀적으로 delete 하면 더블 프리가 난다. 그래서 이 구현은 트리로
//   해제하지 않는다. 대신:
//     - 잎(primitive)들은 기존처럼 list[] 를 통해 한 번씩 해제하고,
//     - 내부 BvhNode 들은 생성 시 nodeStorage[] 에 등록해 두었다가
//       FreeWorld에서 그 배열을 따라 한 번씩 해제한다.
//   이렇게 하면 누수도 더블 프리도 없다.
class BvhNode : public Hittable
{
public:
    __device__ BvhNode() {}

    // objects     : 잎 객체 포인터 배열. [start, end) 구간을 제자리 정렬한다.
    // nodeStorage : 생성되는 모든 BvhNode를 등록하는 레지스트리(해제용).
    // nodeCount   : nodeStorage에 등록된 노드 개수(공유 카운터).
    __device__ BvhNode(
        Hittable** objects, int start, int end,
        Hittable** nodeStorage, int* nodeCount)
    {
        // [최적화] 이 구간 객체들을 감싸는 경계 상자를 먼저 만든 뒤,
        // 가장 긴 축을 분할 축으로 고른다(랜덤 축보다 분할 효율이 좋다).
        mBBox = Aabb();
        for (int i = start; i < end; i++)
            mBBox = Aabb(mBBox, objects[i]->BoundingBox());

        int axis = mBBox.LongestAxis();
        int objectSpan = end - start;

        if (objectSpan == 1)
        {
            // 객체가 하나면 양쪽 자식에 같은 객체를 둔다(널 포인터 검사 회피).
            mLeft = mRight = objects[start];
        }
        else if (objectSpan == 2)
        {
            mLeft = objects[start];
            mRight = objects[start + 1];
        }
        else
        {
            // 분할 축 기준으로 구간을 정렬하고 절반씩 자식으로 나눈다.
            DeviceSort(objects, start, end, axis);

            int mid = start + objectSpan / 2;

            BvhNode* leftNode = new BvhNode(objects, start, mid, nodeStorage, nodeCount);
            BvhNode* rightNode = new BvhNode(objects, mid, end, nodeStorage, nodeCount);

            // 해제용 레지스트리에 등록 (각 노드는 부모가 정확히 한 번 등록)
            nodeStorage[(*nodeCount)++] = leftNode;
            nodeStorage[(*nodeCount)++] = rightNode;

            mLeft = leftNode;
            mRight = rightNode;
        }
    }

    // 반복(iterative) BVH 순회.
    //
    // ※ GPU 에서 재귀 + 가상함수로 BVH 를 순회하면 스레드당 스택 소비가 매우
    //   커진다. 트리 깊이(~log2 N)만큼 가상 호출 프레임이 쌓이는데, 렌더 커널은
    //   이미 RayColor/Scatter/Camera 인라인으로 프레임이 크다. 그 결과 일부
    //   스레드에서 스택이 넘쳐(컬링 경로에서) 결과가 비결정적으로 깨지고,
    //   스택을 키우면 이번엔 동시 점유(occupancy) 한계로 커널 실행이 실패한다.
    //   그래서 재귀 대신 작은 고정 크기 명시적 스택(stack[])으로 순회한다.
    //   내부 노드는 스택으로 펼치고, 잎(primitive)만 가상 Hit 로 검사한다.
    __device__ bool Hit(
        const Ray& ray,
        double tMin,
        double tMax,
        HitRecord& hitRecord) const override
    {
        // 트리 높이는 균형 분할로 ~log2(N) 이므로 32 면 충분(N < 2^32).
        const BvhNode* stack[32];
        int sp = 0;

        const BvhNode* node = this;
        bool hitAnything = false;
        double closest = tMax;

        while (true)
        {
            // 현재 내부 노드의 경계 상자에 맞을 때만 자식을 살펴본다.
            if (node->mBBox.Hit(ray, Interval(tMin, closest)))
            {
                const BvhNode* next = nullptr;
                Hittable* kids[2] = { node->mLeft, node->mRight };

                for (int c = 0; c < 2; c++)
                {
                    Hittable* kid = kids[c];
                    if (kid->IsBvhNode())
                    {
                        // 내부 노드: 하나는 바로 내려가고 나머지는 스택에 보관
                        const BvhNode* bn = static_cast<const BvhNode*>(kid);
                        if (next == nullptr) next = bn;
                        else if (sp < 32) stack[sp++] = bn;
                    }
                    else
                    {
                        // 잎(primitive): 직접 교차 검사. 더 가까우면 closest 갱신
                        if (kid->Hit(ray, tMin, closest, hitRecord))
                        {
                            hitAnything = true;
                            closest = hitRecord.T;
                        }
                    }
                }

                if (next != nullptr)
                {
                    node = next;
                    continue;
                }
            }

            // 더 내려갈 곳이 없으면 스택에서 다음 노드를 꺼낸다.
            if (sp == 0) break;
            node = stack[--sp];
        }

        return hitAnything;
    }

    __device__ Aabb BoundingBox() const override { return mBBox; }

    __device__ bool IsBvhNode() const override { return true; }

private:
    Hittable* mLeft;
    Hittable* mRight;
    Aabb mBBox;

    // 분할 축의 경계 상자 최솟값을 기준으로 a < b 인지 비교한다.
    __device__ static bool BoxCompare(const Hittable* a, const Hittable* b, int axisIndex)
    {
        Interval aAxis = a->BoundingBox().AxisInterval(axisIndex);
        Interval bAxis = b->BoundingBox().AxisInterval(axisIndex);
        return aAxis.Min < bAxis.Min;
    }

    // [start, end) 구간을 분할 축 기준으로 제자리 정렬한다.
    // 단일 스레드에서 동작하며, 구간이 작아 삽입 정렬로 충분하다
    // (디바이스에서는 std::sort를 쓸 수 없다).
    __device__ static void DeviceSort(Hittable** objects, int start, int end, int axis)
    {
        for (int i = start + 1; i < end; i++)
        {
            Hittable* key = objects[i];
            int j = i - 1;
            while (j >= start && BoxCompare(key, objects[j], axis))
            {
                objects[j + 1] = objects[j];
                j--;
            }
            objects[j + 1] = key;
        }
    }
};

#endif
