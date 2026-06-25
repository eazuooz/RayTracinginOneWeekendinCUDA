# Bounding Volume Hierarchies (경계 볼륨 계층 구조) — CUDA 적용판

> *Ray Tracing: The Next Week* 3장을 우리 **CUDA + 레이트레이싱 프로젝트**에 맞춰 다시 정리한 문서.
> 원서의 설명을 그대로 살리되, 코드는 우리 프레임워크(GPU 디바이스 코드)에 맞춰 바꿔 적었고, 실제로 부딪힌 GPU 특유의 문제와 해결까지 함께 담았다.

---

이 부분은 우리가 만들고 있는 레이 트레이서에서 단연 가장 어렵고, 작업량도 많은 파트다. 코드를 더 빠르게 돌리기 위해 이 장에 넣었고, 또 `Hittable`을 약간 리팩터링하게 되는데, 나중에 사각형(rectangle)과 박스(box)를 추가하면 다시 돌아와서 손볼 필요가 없도록 하기 위해서이기도 하다.

레이와 물체의 교차 판정(ray-object intersection)은 레이 트레이서에서 가장 큰 시간 병목이며, 실행 시간은 물체 개수에 비례해 **선형으로** 증가한다. 하지만 같은 장면(scene)에 대해 같은 종류의 탐색을 반복하는 일이므로, 이걸 이진 탐색의 느낌으로 **로그 시간(logarithmic)** 탐색으로 바꿀 수 있어야 한다. 동일한 장면에 수백만에서 수십억 개의 레이를 쏘기 때문에, 장면 안의 물체들을 정렬해 두면 각 레이의 교차 판정이 선형보다 훨씬 작은(sublinear) 탐색이 될 수 있다. 정렬/구조화의 가장 흔한 방식은 1) 공간을 분할하는 방법, 2) 물체를 분할하는 방법이다. 후자는 구현이 훨씬 쉽고, 대부분의 모델에서 실행 속도도 거의 비슷하게 잘 나온다.

> 💡 **이 장에서 CUDA 때문에 달라지는 큰 그림**
> 원서는 호스트(CPU)에서 STL로 트리를 만들지만, 우리는 월드를 **`CreateWorld` 커널 안에서 디바이스 `new`로 직접** 만든다. 그래서 `std::vector`/`std::sort`/`make_shared`/`shared_ptr`를 전부 못 쓴다. 대신 `Hittable**` 배열 + 디바이스 삽입정렬 + 노드 레지스트리를 쓴다. 또한 GPU 특성상 ① 슬랩 검사 코드 형태, ② 트리 순회 방식을 원서와 다르게 가야 했다(본문 콜아웃 참고).

---

## 핵심 아이디어

여러 프리미티브(primitive)들의 집합에 대해 경계 볼륨(bounding volume)을 만든다는 핵심 아이디어는, 그 물체들을 전부 완전히 감싸는(포함하는) 볼륨을 찾는 것이다. 예를 들어, 어떤 10개의 물체를 감싸는 구(sphere)를 계산했다고 하자. 어떤 레이가 그 경계 구를 빗나가면, 내부의 10개 물체도 확실히 전부 빗나간다. 반대로 레이가 경계 구를 맞추면, 내부의 10개 물체 중 하나를 맞출 수도 있다. 그래서 경계 볼륨 코드는 항상 이런 형태가 된다:

```text
if (레이가 경계 볼륨에 맞음)
    return 내부 물체들에 맞는지 여부
else
    return false
```

여기서 중요한 점은, 이 경계 볼륨을 이용해 장면 속 물체들을 여러 **하위 그룹(subgroup)** 으로 묶는다는 것이다. 화면(screen)이나 장면 공간(scene space)을 분할하는 게 아니다. 어떤 물체는 정확히 하나의 경계 볼륨에만 속하길 원하지만, 경계 볼륨끼리는 서로 겹칠 수 있다.

---

## 경계 볼륨의 계층 구조

탐색을 선형보다 빠르게 만들려면 경계 볼륨을 **계층적**으로 만들어야 한다. 예를 들어 물체들을 빨간 그룹과 파란 그룹 두 개로 나누고, 직사각형 경계 볼륨을 쓴다면 다음과 같은 구조가 된다:

![그림 1: 경계 볼륨 계층 구조](https://raytracing.github.io/images/fig-2.01-bvol-hierarchy.jpg)

파란/빨간 경계 박스는 보라색 박스 안에 포함되어 있지만, 서로 겹칠 수도 있고, 어떤 “순서”가 있는 것도 아니다. 그저 둘 다 보라색 내부에 들어 있을 뿐이다. 그래서 오른쪽 트리에서 왼쪽/오른쪽 자식은 “정렬된 순서” 같은 개념이 없다. 단지 내부에 있는 두 덩어리일 뿐이다. 코드는 다음과 같은 느낌이 된다:

```text
if (보라색에 맞음)
    hit0 = 파란색 안의 물체들에 맞는지
    hit1 = 빨간색 안의 물체들에 맞는지
    if (hit0 or hit1)
        return true + 더 가까운 히트 정보
return false
```

---

## 축 정렬 경계 상자(AABB: Axis-Aligned Bounding Boxes)

이걸 제대로 동작시키려면 “좋은 분할”을 만들어내는 방법이 필요하고(나쁜 분할 말고), 레이가 경계 볼륨과 교차하는지 빠르게 테스트하는 방법도 필요하다. 레이-경계 볼륨 교차는 매우 빨라야 하고, 경계 볼륨은 가능한 한 꽉 맞게(compact) 물체를 감싸는 편이 좋다. 실전에서는 대부분의 모델에서, (앞에서 언급한 구 형태 같은 것보다) **축 정렬 박스(axis-aligned box)** 가 더 잘 맞고 더 효율적이다. 다만 이 선택은 항상 염두에 두어야 한다. 다른 종류의 경계 모델이 더 유리한 상황도 있기 때문이다.

이제부터 우리는 축에 정렬된 직육면체(정확히 말하면 ‘직사각형 평행육면체’)를 **axis-aligned bounding box**, 줄여서 **AABB**라고 부른다. 코드에서는 “bounding box”의 약자인 **bbox**라는 이름도 흔히 보게 될 것이다. AABB와 레이의 교차를 검사하는 방법은 어떤 걸 써도 되며, 우리가 필요한 건 “맞았는지 여부”뿐이다. 물체를 그릴 때 필요한 교차점, 법선, 그 외 정보는 경계 볼륨 단계에선 필요 없다.

대부분은 **“슬랩(slab) 방법”** 을 사용한다. 이 방법은 관찰 하나에서 출발한다: n차원 AABB는 결국 **n개의 축 정렬 구간(interval)** 의 교집합이며, 이 구간들을 흔히 “slab(슬랩)”이라고 부른다. 구간이라는 건 예를 들어 `x`가 `3 ≤ x ≤ 5`인 점들의 집합이고, 더 간단히는 `x ∈ [3,5]`라고 쓴다. 2D에서 AABB(직사각형)는 두 구간의 겹침으로 정의된다:

![그림 2: 2D 축 정렬 경계 상자](https://raytracing.github.io/images/fig-2.02-2d-aabb.jpg)

어떤 레이가 한 구간(interval)을 맞는지 보려면, 먼저 레이가 경계면(plane)들과 어디서 만나는지 알아야 한다. 예를 들어 1D에서, 두 평면(사실상 점)과의 교차는 레이 파라미터 `t0`, `t1`을 준다. (레이가 평면과 평행하면 교차는 정의되지 않는다.)

![그림 3: 레이-슬랩 교차](https://raytracing.github.io/images/fig-2.03-ray-slab.jpg)

그러면 레이와 평면의 교차는 어떻게 구할까? 레이는 파라미터 `t`를 넣으면 위치를 반환하는 함수로 정의된다:

```text
P(t) = Q + t·d
```

이 식은 x/y/z 모든 좌표에 적용된다. 예를 들어 `x(t) = Qx + t·dx` 이다. 이 레이가 `x = x0` 평면을 맞는다면, 어떤 `t0`에서 다음을 만족한다:

```text
x0 = Qx + t0·dx   →   t0 = (x0 − Qx) / dx
```

같은 방식으로 `x1`에 대한 식도 얻는다:

```text
t1 = (x1 − Qx) / dx
```

이 1D 수식을 2D나 3D의 히트 테스트로 바꾸는 **핵심 관찰**은 이것이다: 레이가 여러 쌍의 경계면으로 둘러싸인 박스와 교차한다면, 각 축에서 얻어지는 모든 `t`-구간들이 서로 **겹쳐야** 한다. 예를 들어 2D에서는 다음처럼, 초록/파랑 `t` 구간이 겹칠 때에만 레이가 박스를 통과한다:

![그림 4: 레이-슬랩의 t-구간 겹침](https://raytracing.github.io/images/fig-2.04-ray-slab-interval.jpg)

위 그림의 윗부분에서는 `t` 구간이 서로 겹치지 않으므로 레이는 박스를 **맞지 않는다**. 아래 그림에서는 `t` 구간이 겹치므로 레이는 박스를 **맞는다**.

---

## AABB와 레이의 교차 판정

다음 의사 코드는 슬랩의 `t` 구간들이 겹치는지 확인해, 레이가 AABB를 맞는지 판단한다:

```text
interval_x ← compute_intersection_x(ray, x0, x1)
interval_y ← compute_intersection_y(ray, y0, y1)
return overlaps(interval_x, interval_y)
```

이게 놀라울 정도로 간단하고, 3D 버전도 그대로 확장된다는 점이 슬랩 방법이 사랑받는 이유다:

```text
interval_x ← compute_intersection_x(ray, x0, x1)
interval_y ← compute_intersection_y(ray, y0, y1)
interval_z ← compute_intersection_z(ray, z0, z1)
return overlaps(interval_x, interval_y, interval_z)
```

물론 처음 보기만큼 깔끔하지 않은 **주의사항**들이 있다. 1D에서의 `t0`, `t1` 식을 다시 보자:

```text
t0 = (x0 − Qx) / dx
t1 = (x1 − Qx) / dx
```

- **첫째**, 레이가 음의 x 방향으로 진행하면 위에서 계산한 `(t0, t1)` 구간이 뒤집힐 수 있다. 예를 들어 `(7, 3)` 같은 식이다.
- **둘째**, 분모 `dx`가 0이면 무한대 값이 나온다.
- **셋째**, 레이의 시작점이 슬랩 경계 위에 놓이면 분자와 분모가 모두 0이 되어 `NaN`이 생길 수 있다. 게다가 IEEE 부동소수점에서는 0이 ± 부호를 가질 수도 있다.

`dx = 0`인 경우의 좋은 소식은 `t0`와 `t1`이 같아진다는 점이다. 둘 다 `+∞` 또는 `-∞`가 된다(시작점이 `x0`~`x1` 사이에 있지 않다면). 그래서 `min/max`를 쓰면 대체로 올바른 답을 얻는다:

```text
t_small = min( (x0 − Qx)/dx , (x1 − Qx)/dx )
t_big   = max( (x0 − Qx)/dx , (x1 − Qx)/dx )
```

이렇게 해도 남는 골칫거리는 `dx = 0`이고 `x0 − Qx = 0` 또는 `x1 − Qx = 0`인 경우인데, 이때 `NaN`이 나온다. 이 경우는 임의로 “맞았다”/“안 맞았다”로 처리해도 된다(나중에 다룬다).

그리고 `overlaps()`는 두 `t` 구간의 겹침이 비어 있지 않은지로 판단한다:

```text
bool overlaps(t1, t2)
    t_min ← max(t1.min, t2.min)
    t_max ← min(t1.max, t2.max)
    return t_min < t_max
```

여기서 `NaN`이 끼어 있으면 비교가 false가 되므로, “스치듯이 닿는(grazing)” 경우까지 신경 쓴다면(레이 트레이서에서는 결국 모든 경우가 나오니 신경 쓰는 게 맞다) 경계 박스에 약간의 **패딩**을 주는 편이 좋다.

이를 위해 먼저 `Interval` 클래스에, 구간을 일정량만큼 늘려주는 `Expand` 함수를 추가한다.

> 📄 **파일: `Interval.h`** — 우리 `Interval`은 멤버가 `Min/Max`이고, 모든 메서드가 `__host__ __device__`다. 원서의 `static const interval empty, universe;` 전역 상수는 디바이스에서 번거롭고 불필요해서 만들지 않는다. 빈 구간은 **기본 생성자 `Interval()`**(= `Min=+DBL_MAX, Max=-DBL_MAX`)로 충분하다.

```cpp
__host__ __device__ double Clamp(double value) const
{
    if (value < Min) return Min;
    if (value > Max) return Max;
    return value;
}

// 구간을 delta만큼 양쪽으로 넓힌다.
// 두께가 0인 AABB(축에 완전히 평행한 면)에서 생기는 grazing/NaN 문제를 막는 패딩용.
__host__ __device__ Interval Expand(double delta) const
{
    double padding = delta / 2.0;
    return Interval(Min - padding, Max + padding);
}
```

***Listing 7:** [Interval.h] `Interval::Expand()` 메서드*

이제 새로운 AABB 클래스를 구현하는 데 필요한 것들이 모두 준비되었다.

> ⚠️ **CUDA 12.9 코드젠 이슈 (꼭 읽기)**
> 원서의 `aabb::hit`은 “`for (axis 0..2)` 루프 안에서 by-value `interval`을 갱신하며 조건이 깨지면 즉시 `return false`” 하는 형태다. 그런데 **nvcc 12.9에서 이 형태가 일부 레이에 대해 잘못 컴파일**된다. 첫 번째 축을 검사한 뒤 루프가 더 돌지 않고 `false`를 반환해 버려서, BVH 컬링이 “맞는데 안 맞았다”고 판단(false-negative)한다. 그 결과 화면이 **배경만 나오거나(스카이박스만), 또는 어둡고 노이즈가 잔뜩** 낀다. `-O`(최적화), `-G`(디버그) 둘 다에서 재현됐다.
> → 해결: **루프와 루프 내부 조기 `return`을 전부 없애고**, 세 축을 풀어 쓴 **분기 없는 `fmin`/`fmax`** 형태로 작성하면 안정적으로 올바른 코드가 나온다. 각 축에서 슬랩 t-구간 `[lo, hi]`를 구해 누적 교집합 `[tMin, tMax]`를 좁히고, 마지막에 `tMax > tMin`이면 세 구간이 겹친 것(=박스 통과)이다. 논리적으로 원서와 동일하다(각 축에서 한 번이라도 `tMax ≤ tMin`이 되면 이후 `fmin/fmax`로도 그대로 유지되므로, 끝에서 한 번만 검사해도 결과가 같다).

> 📄 **파일: `AABB.h` (신규)**

```cpp
#pragma once
#ifndef AABB_H
#define AABB_H

#include "Ray.h"
#include "Interval.h"

// 3차원 AABB = x/y/z 세 슬랩(구간)의 교집합.
// 레이가 이 상자를 통과하는지 빠르게 판정하는 것이 BVH 가속의 핵심이다.
class Aabb
{
public:
    Interval X;
    Interval Y;
    Interval Z;

    // 기본 AABB는 비어 있다(Interval 기본 생성자가 빈 구간을 만든다).
    __host__ __device__ Aabb() {}

    __host__ __device__ Aabb(const Interval& x, const Interval& y, const Interval& z)
        : X(x), Y(y), Z(z)
    {
    }

    // 두 점 a, b를 상자의 양 극점으로 보고 만든다(좌표 대소 순서 신경 안 써도 됨).
    __host__ __device__ Aabb(const Point3& a, const Point3& b)
    {
        X = (a[0] <= b[0]) ? Interval(a[0], b[0]) : Interval(b[0], a[0]);
        Y = (a[1] <= b[1]) ? Interval(a[1], b[1]) : Interval(b[1], a[1]);
        Z = (a[2] <= b[2]) ? Interval(a[2], b[2]) : Interval(b[2], a[2]);
    }

    // (Listing 13) 두 AABB를 모두 감싸는 AABB.
    __host__ __device__ Aabb(const Aabb& box0, const Aabb& box1)
        : X(box0.X, box1.X)
        , Y(box0.Y, box1.Y)
        , Z(box0.Z, box1.Z)
    {
    }

    __host__ __device__ const Interval& AxisInterval(int n) const
    {
        if (n == 1) return Y;
        if (n == 2) return Z;
        return X;
    }

    // 슬랩 방법으로 레이-상자 교차 여부만 판정한다(히트 지점/법선 불필요).
    // Ray 접근자가 __device__ 전용이라 이 함수도 __device__ 전용이다.
    // ※ 위 콜아웃 참고: 루프/조기 return 대신 분기 없는 fmin/fmax 형태.
    __device__ bool Hit(const Ray& ray, Interval rayT) const
    {
        const Point3& o = ray.Origin();
        const Vector3& d = ray.Direction();

        double tMin = rayT.Min;
        double tMax = rayT.Max;

        // X 축 슬랩
        double invDx = 1.0 / d[0];
        double tx0 = (X.Min - o[0]) * invDx;
        double tx1 = (X.Max - o[0]) * invDx;
        tMin = fmax(tMin, fmin(tx0, tx1));
        tMax = fmin(tMax, fmax(tx0, tx1));

        // Y 축 슬랩
        double invDy = 1.0 / d[1];
        double ty0 = (Y.Min - o[1]) * invDy;
        double ty1 = (Y.Max - o[1]) * invDy;
        tMin = fmax(tMin, fmin(ty0, ty1));
        tMax = fmin(tMax, fmax(ty0, ty1));

        // Z 축 슬랩
        double invDz = 1.0 / d[2];
        double tz0 = (Z.Min - o[2]) * invDz;
        double tz1 = (Z.Max - o[2]) * invDz;
        tMin = fmax(tMin, fmin(tz0, tz1));
        tMax = fmin(tMax, fmax(tz0, tz1));

        return tMax > tMin;
    }

    // (Listing 21) 가장 긴 축의 인덱스 — 분할 축 선택용.
    __host__ __device__ int LongestAxis() const
    {
        if (X.Size() > Y.Size())
            return X.Size() > Z.Size() ? 0 : 2;
        else
            return Y.Size() > Z.Size() ? 1 : 2;
    }
};

#endif
```

***Listing 8:** [AABB.h] 축 정렬 경계 상자 클래스 (+ Listing 13·21 미리 포함)*

---

## Hittable들을 위한 바운딩 박스 만들기

이제 모든 `Hittable`에 대해 바운딩 박스를 계산하는 함수를 추가해야 한다. 그런 다음 모든 프리미티브 위에 박스 계층 구조를 만들고, 구(sphere) 같은 개별 프리미티브들은 트리의 **잎(leaf)** 에 위치하게 된다.

`Interval`은 인자 없이 생성하면 기본으로 empty라는 걸 기억하자. `Aabb`는 3차원 각각의 `Interval`을 갖기 때문에, 기본 생성된 `Aabb`도 결국 empty가 된다. 따라서 어떤 오브젝트는 빈 바운딩 볼륨을 가질 수도 있다(예: 자식이 하나도 없는 `HittableList`). 다행히 우리가 만든 `Interval` 클래스 설계 덕분에, 이런 경우도 수학적으로 잘 굴러간다.

마지막으로 어떤 오브젝트는 애니메이션될 수도 있다는 점도 기억하자. 이런 오브젝트는 `time=0`부터 `time=1`까지의 전체 움직임 범위를 커버하는 경계를 반환해야 한다.

이를 위해 `Hittable` 인터페이스에 `BoundingBox()`를 추가한다. 추가로, 뒤에서 BVH를 **반복(iterative)** 으로 순회할 때 “이 자식이 내부 노드냐 잎이냐”를 구분하기 위한 `IsBvhNode()`도 함께 둔다(기본 false).

> 📄 **파일: `Hittable.h`**

```cpp
#include "Ray.h"
#include "AABB.h"

class Material;

// struct HitRecord { ... }  // 기존과 동일

class Hittable
{
public:
    __device__ virtual bool Hit(
        const Ray& ray,
        double tMin,
        double tMax,
        HitRecord& hitRecord) const = 0;

    // 이 객체를 감싸는 AABB. 움직이는 물체는 운동 전체 구간(time 0~1)을 감싼다.
    __device__ virtual Aabb BoundingBox() const = 0;

    // BVH 반복 순회에서 내부 노드/잎 구분용(기본 false = 잎/일반 오브젝트).
    __device__ virtual bool IsBvhNode() const { return false; }
};
```

***Listing 9:** [Hittable.h] 바운딩 박스를 포함한 Hittable 클래스*

정지한 구(stationary sphere)의 경우 `BoundingBox`는 간단하다. 반지름 벡터로 중심 ± r 두 극점을 잡아 박스를 만들고, 그것을 멤버에 저장해 두었다가 그대로 반환하면 된다.

> 📄 **파일: `Sphere.h`**

```cpp
__device__ Sphere(const Point3& center, double radius, Material* material)
    : mCenter(center)
    , mRadius(radius)
    , mMaterial(material)
{
    // 반지름 벡터로 중심 ± r 두 극점을 잡아 경계 상자를 만든다.
    Vector3 rvec(radius, radius, radius);
    mBBox = Aabb(center - rvec, center + rvec);
}

// ... Hit() 은 기존과 동일 ...

__device__ Aabb BoundingBox() const override { return mBBox; }

private:
    Point3 mCenter;
    double mRadius;
    Material* mMaterial;
    Aabb mBBox;   // 추가
```

***Listing 10:** [Sphere.h] 바운딩 박스를 가진 구*

움직이는 구(moving sphere)의 경우에는 **움직임 전체 범위를 감싸는 경계**가 필요하다. `time=0`에서의 박스와 `time=1`에서의 박스를 만든 다음, 그 두 박스를 모두 감싸는 박스를 계산하면 된다.

> 📄 **파일: `MovingSphere.h`** — 원서는 `sphere` 한 클래스에 정지/이동 생성자를 둘 다 두지만, 우리 프레임워크는 `Sphere`/`MovingSphere`로 분리되어 있어 여기에 적용한다.

```cpp
__device__ MovingSphere(
    Point3 center0, Point3 center1,
    double time0, double time1,
    double radius, Material* material)
    : mCenter0(center0)
    , mCenter1(center1)
    , mTime0(time0)
    , mTime1(time1)
    , mRadius(radius)
    , mMaterial(material)
{
    // 운동 구간 전체를 감싸려면 time0 위치 박스와 time1 위치 박스를 모두 포함해야 한다.
    Vector3 rvec(radius, radius, radius);
    Aabb box0(center0 - rvec, center0 + rvec);
    Aabb box1(center1 - rvec, center1 + rvec);
    mBBox = Aabb(box0, box1);
}

// ... Hit() 은 기존과 동일 ...

__device__ Aabb BoundingBox() const override { return mBBox; }

private:
    // ... 기존 멤버 ...
    Aabb mBBox;   // 운동 전체를 감싸는 박스
```

***Listing 11:** [MovingSphere.h] 바운딩 박스를 가진 움직이는 구*

이제 두 박스를 받아 새 박스를 만드는 `Aabb` 생성자가 필요한데, 그 전에 먼저 `Interval`에 **두 구간을 받아 감싸는 생성자**를 추가하자.

> 📄 **파일: `Interval.h`**

```cpp
// 두 구간을 빈틈없이 감싸는 구간을 생성한다(AABB 합집합용).
__host__ __device__ Interval(const Interval& a, const Interval& b)
    : Min(a.Min <= b.Min ? a.Min : b.Min)
    , Max(a.Max >= b.Max ? a.Max : b.Max)
{
}
```

***Listing 12:** [Interval.h] 두 Interval에서 Interval을 만드는 생성자*

그리고 이를 이용해 두 AABB를 입력으로 받아 그 둘을 감싸는 AABB를 만드는 생성자도 추가한다. (이미 위 `AABB.h`(Listing 8)에 포함해 두었다.)

```cpp
// AABB.h 안 — 두 AABB를 모두 감싸는 AABB
__host__ __device__ Aabb(const Aabb& box0, const Aabb& box1)
    : X(box0.X, box1.X)
    , Y(box0.Y, box1.Y)
    , Z(box0.Z, box1.Z)
{
}
```

***Listing 13:** [AABB.h] 두 AABB로부터 AABB를 만드는 생성자*

---

## 오브젝트 리스트의 바운딩 박스 만들기

이제 `HittableList`를 업데이트해서 자식들의 경계를 계산하자. 원서는 자식을 추가하는 `add()`에서 바운딩 박스를 누적해 갱신하지만, **우리 `HittableList`는 생성자에서 `Hittable**` 배열을 통째로 받으므로**, 누적을 생성자 루프에서 한 번에 처리한다.

> 📄 **파일: `HittableList.h`**

```cpp
__device__ HittableList(Hittable** list, int count)
    : mList(list), mCount(count)
{
    // 자식들의 경계 상자를 합쳐 리스트 전체의 경계 상자를 구한다.
    for (int i = 0; i < count; i++)
        mBBox = Aabb(mBBox, list[i]->BoundingBox());
}

// ... Hit() 은 기존과 동일 ...

__device__ Aabb BoundingBox() const override { return mBBox; }

private:
    Hittable** mList;
    int mCount;
    Aabb mBBox;   // 추가
```

***Listing 14:** [HittableList.h] 바운딩 박스를 가진 Hittable 리스트*

---

## BVH 노드 클래스

BVH도 하나의 `Hittable`이다. 리스트의 `Hittable`과 마찬가지로 컨테이너이지만, “레이가 너를 맞추니?”라는 질문에 답할 수 있다. 설계적으로는 트리 전체를 나타내는 클래스와 노드를 나타내는 클래스를 따로 둘 수도 있고, 클래스 하나만 만들고 루트도 그냥 노드로 취급할 수도 있다. `Hit` 함수는 꽤 단순하다. 먼저 해당 노드의 박스를 맞는지 보고, 맞으면 자식들을 검사해서 더 가까운 히트 정보를 정리하면 된다. 가능하다면 **하나의 클래스 설계**를 선호한다.

여기서 우리 구현이 원서와 크게 갈리는 두 지점이 있다. **빌드(메모리 관리)** 와 **순회 방식**이다.

### 빌드 — STL 없이 디바이스에서 트리 만들기

원서는 `make_shared`(`shared_ptr`)로 트리를 만들고 자동으로 해제한다. 디바이스에는 STL이 없으니 다음처럼 대체한다.

1. `std::vector` → 원본 `Hittable** list` 재사용(제자리 정렬)
2. `std::sort` → `DeviceSort`(단일 스레드 삽입 정렬)
3. `make_shared` → 디바이스 `new BvhNode(...)`
4. `shared_ptr` 자동 해제 → **노드 레지스트리(`nodeStorage[]`)** 에 등록해 두었다가 일괄 해제

> ⚠️ **더블 프리 주의**
> `objectSpan == 1`일 때 원서처럼 `mLeft = mRight = 같은 잎`으로 둔다(탐색을 부드럽게 하고 널 포인터 검사를 피하기 위해). 그러면 같은 잎 포인터가 두 번 참조될 수 있는데, `shared_ptr`가 없는 우리 환경에서 트리를 따라 재귀 `delete`하면 **같은 잎을 두 번 지우는** 더블 프리가 난다.
> → 그래서 트리로 해제하지 않는다. **잎(primitive)은 기존처럼 `list[]`를 통해 한 번씩**, **내부 `BvhNode`는 생성 시 `nodeStorage[]`에 등록해 두었다가 그 배열로 한 번씩** 해제한다. 누수도 더블 프리도 없다.

### 순회 — 재귀 대신 반복(iterative)

> ⚠️ **GPU 스택 오버플로 (가장 까다로웠던 부분)**
> 원서의 `hit`은 `left->hit(...)`, `right->hit(...)`를 부르는 **재귀** 다. CPU에선 문제없지만, GPU에서 재귀 + 가상함수로 BVH를 순회하면 **스레드당 스택 소비가 폭발**한다. 트리 깊이(~log₂N)만큼 가상 호출 프레임이 쌓이는데, 렌더 커널은 이미 `RayColor`/`Material::Scatter`/`Camera::GetRay` 인라인으로 프레임이 크다.
> 그 결과: 일부 스레드에서 스택이 넘쳐 결과가 **비결정적으로** 깨졌다(같은 입력인데 실행할 때마다 이미지가 달라짐 — 화면이 어둡고 노이즈가 낌). 스택 한도(`cudaLimitStackSize`)를 키우면 이번엔 점유(occupancy) 한계 때문에 **커널 실행 자체가 실패**했다. 즉 적당한 스택 크기가 존재하지 않았다.
> → 해결: 재귀를 버리고 **작은 고정 크기 명시적 스택(`stack[32]`)** 으로 반복 순회한다. 내부 노드는 스택에 펼치고, 잎만 가상 `Hit`로 검사한다. 이러면 재귀 깊이가 0이라 스택 부담이 사라지고, 결정적으로 동작하며, 속도도 빠르다. (검증: 무작위 레이 20,000개로 BVH vs 선형 비교 시 불일치 0건, 최종 렌더는 선형 버전과 픽셀 단위로 동일.)

이 두 가지를 반영한 `BvhNode`는 다음과 같다. (원서의 Listing 15·16·18·20을 한 파일에 통합했다.)

> 📄 **파일: `BvhNode.h` (신규)**

```cpp
#pragma once
#ifndef BVH_NODE_H
#define BVH_NODE_H

#include "Hittable.h"
#include "AABB.h"

class BvhNode : public Hittable
{
public:
    __device__ BvhNode() {}

    // objects     : 잎 포인터 배열. [start,end) 구간을 제자리 정렬한다.
    // nodeStorage : 생성된 모든 BvhNode 등록(해제용) — shared_ptr 대체.
    // nodeCount   : nodeStorage 등록 개수(공유 카운터).
    __device__ BvhNode(
        Hittable** objects, int start, int end,
        Hittable** nodeStorage, int* nodeCount)
    {
        // (Listing 20) span의 bbox를 먼저 만들고, 가장 긴 축으로 분할.
        mBBox = Aabb();
        for (int i = start; i < end; i++)
            mBBox = Aabb(mBBox, objects[i]->BoundingBox());

        int axis = mBBox.LongestAxis();
        int objectSpan = end - start;

        if (objectSpan == 1)
        {
            // 잎이 하나면 양쪽 자식에 같은 잎(널 포인터 검사 회피).
            mLeft = mRight = objects[start];
        }
        else if (objectSpan == 2)
        {
            mLeft = objects[start];
            mRight = objects[start + 1];
        }
        else
        {
            DeviceSort(objects, start, end, axis);            // std::sort 대체
            int mid = start + objectSpan / 2;

            BvhNode* leftNode  = new BvhNode(objects, start, mid, nodeStorage, nodeCount);
            BvhNode* rightNode = new BvhNode(objects, mid, end, nodeStorage, nodeCount);

            nodeStorage[(*nodeCount)++] = leftNode;           // 해제 레지스트리 등록
            nodeStorage[(*nodeCount)++] = rightNode;

            mLeft = leftNode;
            mRight = rightNode;
        }
    }

    // (Listing 15) hit — 재귀 대신 고정 크기 명시적 스택으로 반복 순회.
    __device__ bool Hit(
        const Ray& ray, double tMin, double tMax, HitRecord& hitRecord) const override
    {
        const BvhNode* stack[32];   // 균형 분할이라 트리 높이 ~log2(N), 32면 충분
        int sp = 0;

        const BvhNode* node = this;
        bool hitAnything = false;
        double closest = tMax;

        while (true)
        {
            // 현재 노드의 박스에 맞을 때만 자식을 살펴본다.
            if (node->mBBox.Hit(ray, Interval(tMin, closest)))
            {
                const BvhNode* next = nullptr;
                Hittable* kids[2] = { node->mLeft, node->mRight };

                for (int c = 0; c < 2; c++)
                {
                    Hittable* kid = kids[c];
                    if (kid->IsBvhNode())
                    {
                        // 내부 노드: 하나는 바로 내려가고 나머지는 스택에 보관.
                        const BvhNode* bn = static_cast<const BvhNode*>(kid);
                        if (next == nullptr) next = bn;
                        else if (sp < 32) stack[sp++] = bn;
                    }
                    else
                    {
                        // 잎(primitive): 직접 검사. 더 가까우면 closest 갱신.
                        if (kid->Hit(ray, tMin, closest, hitRecord))
                        {
                            hitAnything = true;
                            closest = hitRecord.T;
                        }
                    }
                }

                if (next != nullptr) { node = next; continue; }
            }

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

    // (Listing 18) 분할 축의 bbox 최솟값으로 a < b 비교.
    __device__ static bool BoxCompare(const Hittable* a, const Hittable* b, int axisIndex)
    {
        Interval aAxis = a->BoundingBox().AxisInterval(axisIndex);
        Interval bAxis = b->BoundingBox().AxisInterval(axisIndex);
        return aAxis.Min < bAxis.Min;
    }

    // std::sort 대체: [start,end) 구간을 분할 축 기준 제자리 삽입 정렬(단일 스레드).
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
```

***Listing 15:** [BvhNode.h] 경계 볼륨 계층 구조 (+ Listing 16·18·20 통합)*

---

## BVH 볼륨 분할하기

BVH를 포함한 어떤 가속 구조에서든 가장 복잡한 부분은 **“만드는(building) 과정”** 이다. 이건 생성자에서 한다. BVH의 멋진 점은 `BvhNode` 안의 물체 리스트가 두 개의 하위 리스트로만 나뉘기만 하면 `Hit` 함수는 제대로 동작한다는 것이다. 물론 분할을 잘하면(부모보다 더 작은 경계 박스를 갖게끔) 더 빨라지지만, 그건 속도 문제이지 정답/정확성의 문제는 아니다.

원서는 “중간 정도”의 방법으로 다음을 한다.

1. **임의의 축**을 선택한다 (`random_int`)
2. 프리미티브들을 정렬한다 (`std::sort`)
3. 반씩 나눠 각각 서브트리에 넣는다

입력 리스트가 둘이면 하나씩 배치하고 재귀를 끝낸다. 탐색이 부드럽게 동작하고 null 포인터 검사 같은 걸 안 하려면, 원소가 하나뿐일 때는 **양쪽 자식에 같은 원소를 복제**해 넣는다.

> 🔧 **우리 구현의 차이**: 우리는 1)의 “임의의 축” 단계를 건너뛰고, 처음부터 아래 “또 하나의 최적화”에 해당하는 **가장 긴 축(`LongestAxis()`)** 으로 분할한다. 따라서 원서 Listing 16의 `random_int(0,2)` + 세 비교자 분기 대신, `int axis = mBBox.LongestAxis();` 한 줄을 쓴다(위 `BvhNode` 생성자 참고). 정렬은 `std::sort` 대신 `DeviceSort`(삽입 정렬), 자식 생성은 `make_shared` 대신 `new`다.

참고로 원서의 “바운딩 박스가 존재하는지 검사”하는 부분은, 무한 평면처럼 바운딩 박스가 없는 물체를 넣었을 때를 대비한 것이다. 우리는 그런 프리미티브를 아직 안 쓰니, 그런 일이 생기는 건 나중에 추가할 때쯤일 것이다.

### 박스 비교 함수

원서는 `std::sort()`에 넘길 비교 함수를, “축 인덱스를 추가 인자로 받는 일반 비교 함수”와 그것을 쓰는 “축별 비교 함수(`box_x_compare` 등)”로 만든다. 우리는 `std::sort` 자체를 쓰지 않으므로, 일반 비교 함수 `BoxCompare(a, b, axis)` 하나만 두고, 그 함수를 직접 `DeviceSort`에 넘긴다(세 개의 축별 래퍼는 불필요). 이 둘은 이미 위 `BvhNode.h`(Listing 15)에 포함돼 있다.

```cpp
// (재게시) 분할 축의 bbox 최솟값으로 a < b 비교.
__device__ static bool BoxCompare(const Hittable* a, const Hittable* b, int axisIndex)
{
    Interval aAxis = a->BoundingBox().AxisInterval(axisIndex);
    Interval bAxis = b->BoundingBox().AxisInterval(axisIndex);
    return aAxis.Min < bAxis.Min;
}
```

***Listing 18:** [BvhNode.h] BVH 비교 함수*

> 📌 **원서 Listing 17 (`random_int`) 은 우리 프로젝트에선 필요 없다.** 분할 축을 랜덤이 아니라 `LongestAxis()`로 고르기 때문이다. 따라서 `RtWeekend.h`에 `random_int`를 추가하지 않는다.

---

## 이제 BVH를 사용해 보자

원서는 `main`에서 단 한 줄, `world = hittable_list(make_shared<bvh_node>(world));` 로 월드를 BVH로 감싼다. 하지만 우리는 월드를 GPU 커널에서 만들고 해제하므로, 이 한 줄이 **빌드 + 해제 + 호스트 플러밍**으로 흩어진다.

> 📄 **파일: `kernel.cu`** *(원서 Listing 19의 `main.cc`에 해당)*

### (1) `CreateWorld` — 구를 다 배치한 뒤 BVH로 묶기

```cpp
// CreateWorld 시그니처: BVH 노드 배열/개수 출력 인자 추가
__global__ void CreateWorld(
    Hittable** list, Hittable** world, Camera** camera,
    int imageWidth, int imageHeight, curandState* randState, int* outCount,
    Hittable** bvhNodes, int* outNodeCount)
{
    // ... 바닥/소형 구/대형 구를 list[0..i)에 배치 (기존과 동일) ...

    *outCount = i;   // 실제 배치된 구 개수

    // === BVH 빌드 ===
    // list[0..i)를 BVH로 묶는다. 빌드 중 list는 분할 축 기준으로 제자리 정렬되지만,
    // 잎 포인터는 그대로 보존되므로 FreeWorld의 list[] 해제에는 영향이 없다.
    int nodeCount = 0;
    BvhNode* root = new BvhNode(list, 0, i, bvhNodes, &nodeCount);
    bvhNodes[nodeCount++] = root;   // 루트도 해제 레지스트리에 등록
    *outNodeCount = nodeCount;
    *world = root;                  // 기존 HittableList 대신 BVH 루트를 월드로

    // ... 카메라 생성 (기존과 동일) ...
}
```

### (2) `FreeWorld` — 잎과 내부 노드를 각각 한 번씩 해제

```cpp
__global__ void FreeWorld(
    Hittable** list, int numHittables,
    Hittable** bvhNodes, int numNodes,
    Hittable** world, Camera** camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int i = 0; i < numHittables; i++)
            delete list[i];           // 잎(primitive)
        for (int i = 0; i < numNodes; i++)
            delete bvhNodes[i];       // 내부 노드(*world == 루트도 여기 포함)
        delete *camera;
    }
}
```

### (3) `main` — 노드 레지스트리 할당 / 개수 공유 / 호출

```cpp
// BVH 순회는 재귀가 아니라 명시적 스택(BvhNode::Hit)을 쓰므로 추가 스택은 필요 없다.
// (재귀로 두면 이 한도로도 일부 스레드에서 스택이 넘쳤다.)
checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 32768));

// ...

// n개 잎에 대한 BVH 내부 노드 수는 최대 2n 미만 → 2*maxHittables 로 넉넉히 잡는다.
Hittable** bvhNodes;
checkCudaErrors(cudaMalloc((void**)&bvhNodes, 2 * maxHittables * sizeof(Hittable*)));

int* d_numNodes;
checkCudaErrors(cudaMallocManaged((void**)&d_numNodes, sizeof(int)));
*d_numNodes = 0;

CreateWorld<<<1, 1>>>(list, world, camera, imageWidth, imageHeight,
                      randState2, d_numHittables, bvhNodes, d_numNodes);
checkCudaErrors(cudaGetLastError());
checkCudaErrors(cudaDeviceSynchronize());

int numHittables = *d_numHittables;
int numNodes     = *d_numNodes;

// ... RenderInit / Render ...

// 해제
FreeWorld<<<1, 1>>>(list, numHittables, bvhNodes, numNodes, world, camera);
checkCudaErrors(cudaGetLastError());
checkCudaErrors(cudaDeviceSynchronize());

checkCudaErrors(cudaFree(bvhNodes));
checkCudaErrors(cudaFree(d_numNodes));
// ... 나머지 cudaFree ...
```

***Listing 19:** [kernel.cu] BVH를 사용한 랜덤 구 장면*

렌더링 결과 이미지는 BVH를 쓰지 않은 버전과 **완전히 동일**해야 한다(우리는 MD5 해시까지 일치함을 확인했다). 하지만 시간을 재 보면 BVH 버전이 더 빠르다. 원서는 약 **6.5배** 향상을 보고하는데, 우리 환경(RTX 5070 Ti, 1440×720, 10 spp)에서는 **2.84s → 0.47s, 약 6배** 빨라졌다.

---

## 또 하나의 BVH 최적화 — 가장 긴 축

BVH를 조금 더 빠르게 만들 수 있다. 랜덤으로 분할 축을 선택하는 대신, 전체를 감싸는 바운딩 박스에서 **가장 긴 축(longest axis)** 을 기준으로 분할하면 보통 더 잘게 쪼개져 효율이 좋아진다.

원서는 이를 위해 두 가지를 한다.

1. BVH 생성자에서, 해당 구간(span)의 오브젝트들을 감싸는 박스를 먼저 만든다(빈 박스에서 시작해 각 오브젝트 박스로 확장).
2. 그 박스에서 가장 긴 변을 가진 축을 분할 축으로 고른다(`aabb::longest_axis()`).

> 🔧 **우리 구현은 처음부터 이 방식**으로 작성했다(위 `BvhNode` 생성자). 즉:
> - `mBBox = Aabb(); for(...) mBBox = Aabb(mBBox, objects[i]->BoundingBox());` 로 span의 박스를 만들고,
> - `int axis = mBBox.LongestAxis();` 로 가장 긴 축을 고른다.
> - 원서의 `static const aabb empty/universe` 전역 상수는 **만들지 않고**, 빈 박스가 필요한 곳은 기본 생성자 `Aabb()`로 대체한다.

`LongestAxis()` 자체는 이미 `AABB.h`(Listing 8)에 있다.

```cpp
__host__ __device__ int LongestAxis() const
{
    // 바운딩 박스에서 가장 긴 축의 인덱스를 반환.
    if (X.Size() > Y.Size())
        return X.Size() > Z.Size() ? 0 : 2;
    else
        return Y.Size() > Z.Size() ? 1 : 2;
}
```

***Listing 20·21:** [BvhNode.h / AABB.h] span의 bbox + longest_axis*

이전과 마찬가지로 결과 이미지는 동일하지만, 렌더링이 조금 더 빨라진다.

---

## 결과 & 검증

- **정확성**: BVH 미적용(선형 탐색) 렌더와 **MD5 해시까지 완전히 동일**(픽셀 단위 일치). 무작위 레이 20,000개로 BVH vs 선형 비교 시 불일치 0건.
- **속도**: 1440×720, 10 spp, RTX 5070 Ti 기준 **2.84s → 0.47s (약 6배)**.
- **결정성**: 반복 순회로 바꾼 뒤 매 실행 결과가 동일하다(이전 재귀판은 비결정적으로 깨졌다).

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| 7, 12 | `Interval.h` | `Expand`, 두-구간 생성자 |
| 8, 13, 21 | `AABB.h` *(신규)* | `Hit`은 분기 없는 `fmin/fmax`, `LongestAxis` |
| 9 | `Hittable.h` | `BoundingBox()` + `IsBvhNode()` |
| 10 | `Sphere.h` | 정지 구 bbox |
| 11 | `MovingSphere.h` | 별도 클래스에 적용 |
| 14 | `HittableList.h` | 생성자에서 누적 |
| 15·16·18·20 | `BvhNode.h` *(신규)* | 반복 순회 + 노드 레지스트리 + 가장 긴 축 |
| 17 | — | `random_int` 미사용(`LongestAxis`로 대체) |
| 19 | `kernel.cu` | `CreateWorld` / `FreeWorld` / `main` |

### CUDA 적용에서 꼭 기억할 3가지

1. **빌드는 디바이스에서**: STL 대신 `Hittable**` + 디바이스 정렬 + `new`, 해제는 노드 레지스트리(잎/내부 노드 각각 한 번씩 — 더블 프리 방지).
2. **`Aabb::Hit`는 분기 없는 형태로**: “루프 + 루프 내부 조기 return” 슬랩 검사는 nvcc 12.9에서 잘못 컴파일될 수 있다 → 세 축을 펼친 `fmin/fmax` 형태.
3. **순회는 반복(iterative)으로**: 재귀 + 가상함수는 GPU 스레드 스택을 초과해 비결정적 버그 또는 커널 실행 실패를 부른다 → 작은 명시적 스택으로 순회.
