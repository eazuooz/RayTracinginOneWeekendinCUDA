# Quadrilaterals (사각형) — CUDA 적용판

> *Ray Tracing: The Next Week* 6장(Quadrilaterals)을 우리 **CUDA + 레이트레이싱 프로젝트**에 맞춰 다시 정리한 문서.
> 원서의 설명을 그대로 살리되, 코드는 우리 프레임워크(GPU 디바이스 코드)에 맞춰 바꿔 적었다. 실제로 빌드·실행해 책의 Image 16과 동일한 결과를 확인했다.

---

지금까지 이 3권 시리즈의 절반 이상을 **구(sphere)** 하나만으로 진행해 왔다. 이제 두 번째 프리미티브 **사각형(quad)** 을 추가한다.

> 💡 **이 장에서 CUDA 때문에 달라지는 건 거의 없다**
> 사각형은 수학(평면 방정식 + 평면 좌표)으로 푸는 도형이라, 포팅이 거의 그대로다. 차이는 두 가지뿐이다.
> 1. `shared_ptr<material>` → raw `Material*` (다른 프리미티브와 동일).
> 2. `Hit` 시그니처가 `interval ray_t` 대신 `double tMin, double tMax` (우리 `Hittable` 인터페이스).
> 추가로, 평평한 사각형이 축 평면에 놓이면 AABB 한 축의 두께가 0이 되는 문제를 막기 위해 **`Aabb`에 패딩**을 넣는다(원서 Listing 48과 동일).

---

## 사각형 정의하기

이름은 `quad`지만 엄밀히는 **평행사변형**(마주보는 변이 평행)이다. 세 가지 기하 요소로 정의한다.

- **Q** : 시작 모서리.
- **u** : 첫 번째 변 벡터. `Q+u`가 인접 모서리.
- **v** : 두 번째 변 벡터. `Q+v`가 다른 인접 모서리.

대각 모서리는 `Q+u+v`다. 사각형 자체는 2D지만 이 값들은 3D다. 예를 들어 원점에 모서리를 두고 Z로 2, Y로 1만큼 뻗은 사각형은 `Q=(0,0,0), u=(0,0,2), v=(0,1,0)`이다.

![그림 5: 사각형 구성 요소](https://raytracing.github.io/images/fig-2.05-quad-def.jpg)

사각형은 평평하므로 XY/YZ/ZX 평면에 놓이면 AABB의 한 축 두께가 0이 되어 레이 교차에서 수치 문제가 생길 수 있다. 교차 결과는 그대로 두고, 경계 상자만 **작은 패딩**으로 넓혀 부피가 0이 되지 않게 한다.

> 📄 **파일: `AABB.h`** — 원서 Listing 48. `Aabb`의 두 생성자에서 `PadToMinimums()`를 호출한다. 우리 `Interval`에는 이미 `Expand`/`Size`가 있다(3장에서 추가).

```cpp
// (Interval×3) 와 (Point3,Point3) 생성자 끝에서 호출
__host__ __device__ Aabb(const Point3& a, const Point3& b)
{
    X = (a[0] <= b[0]) ? Interval(a[0], b[0]) : Interval(b[0], a[0]);
    Y = (a[1] <= b[1]) ? Interval(a[1], b[1]) : Interval(b[1], a[1]);
    Z = (a[2] <= b[2]) ? Interval(a[2], b[2]) : Interval(b[2], a[2]);
    PadToMinimums();
}

private:
// 어느 변도 delta보다 좁지 않도록 패딩(평면 도형의 두께 0 축 보정).
__host__ __device__ void PadToMinimums()
{
    double delta = 0.0001;
    if (X.Size() < delta) X = X.Expand(delta);
    if (Y.Size() < delta) Y = Y.Expand(delta);
    if (Z.Size() < delta) Z = Z.Expand(delta);
}
```

***Listing 48:** [AABB.h] `Aabb::PadToMinimums()` 추가*

> 🔧 **두 AABB 합성 생성자(`Aabb(box0, box1)`)에는 패딩을 넣지 않는다.** 그 경로는 자식들이 이미 패딩된 박스를 합치는 것이라 필요 없다(BVH 빌드/`Sphere`엔 영향 없음). 패딩은 사각형의 두-점/구간 생성자에서만 의미가 있다.

이제 `Quad` 클래스의 첫 스케치다(평면 값은 다음 절에서 채운다).

> 📄 **파일: `Quad.h` (신규)** — 원서 Listing 49. `shared_ptr<material>` → `Material*`.

```cpp
#pragma once
#ifndef QUAD_H
#define QUAD_H

#include "Hittable.h"

class Quad : public Hittable
{
public:
    __device__ Quad(const Point3& q, const Vector3& u, const Vector3& v, Material* material)
        : mQ(q), mU(u), mV(v), mMaterial(material)
    {
        SetBoundingBox();
    }

    // 네 꼭짓점을 감싸는 AABB = 두 대각선 박스의 합집합.
    __device__ void SetBoundingBox()
    {
        Aabb diag1 = Aabb(mQ, mQ + mU + mV);
        Aabb diag2 = Aabb(mQ + mU, mQ + mV);
        mBBox = Aabb(diag1, diag2);
    }

    __device__ Aabb BoundingBox() const override { return mBBox; }

    __device__ bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& rec) const override
    {
        return false; // 아래에서 구현
    }

private:
    Point3 mQ;
    Vector3 mU, mV;
    Material* mMaterial;
    Aabb mBBox;
};

#endif
```

***Listing 49:** [Quad.h] 2D 사각형(평행사변형) 클래스*

---

## 광선과 평면의 교차

레이-사각형 교차는 3단계다. ① 사각형을 포함하는 평면 찾기 → ② 레이-평면 교점 구하기 → ③ 교점이 사각형 내부인지 판정. 먼저 ②를 푼다.

평면의 암시적 방정식은 `Ax + By + Cz = D`이고, `(A,B,C)`가 법선 `n`이다. 벡터로 쓰면 평면 위 모든 점 `v`에 대해 `n·v = D`. 레이 `R(t) = P + t·d`를 대입해 `t`를 푼다:

```text
n·(P + t·d) = D
t = (D − n·P) / (n·d)
```

분모 `n·d`가 0이면 레이가 평면과 평행 → 미스. `t`가 레이 허용 구간 밖이면 역시 미스.

### 사각형을 포함하는 평면 찾기

법선은 두 변 벡터의 외적이다: `n = unit_vector(u×v)`. `Q`가 평면 위에 있으므로 `D = n·Q`. 이 두 값을 생성자에서 캐시한다.

> 📄 **파일: `Quad.h`** — 원서 Listing 50. 생성자에서 평면 값 계산.

```cpp
Vector3 n = Cross(u, v);
mNormal = UnitVector(n);
mD = Dot(mNormal, mQ);
```

***Listing 50:** [Quad.h] 평면 값 캐싱*

무한 평면에 대한 `Hit()` 중간 버전(아직 내부 판정 없음):

```cpp
__device__ bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& rec) const override
{
    double denom = Dot(mNormal, ray.Direction());
    if (fabs(denom) < 1e-8) return false;               // 평면과 평행 → 미스

    double t = (mD - Dot(mNormal, ray.Origin())) / denom;
    if (t < tMin || t > tMax) return false;             // 구간 밖 → 미스

    Point3 intersection = ray.At(t);
    rec.T = t;
    rec.P = intersection;
    rec.MaterialPtr = mMaterial;
    rec.SetFaceNormal(ray, mNormal);
    return true;
}
```

***Listing 51:** [Quad.h] 무한 평면용 `Hit()`*

![그림 6: 광선과 평면의 교차](https://raytracing.github.io/images/fig-2.06-ray-plane.jpg)

---

## 평면 위의 점 방향 잡기

교점은 평면 위에 있지만 사각형 *안*인지 *밖*인지는 아직 모른다. 이를 판정하고 텍스처 좌표를 주려면 평면에 **좌표계**를 세운다. 평면 원점 `Q`와 두 기저 `u, v`(직교일 필요 없음, 평행만 아니면 됨)로 임의의 점을 표현한다.

임의 점 `P`를 `P = Q + α·u + β·v`로 쓸 때, 평면 좌표 `α, β`는:

```text
p = P − Q
w = n / (n·n)          ← 사각형마다 상수 → 캐시
α = w·(p × v)
β = w·(u × p)
```

> 📄 **파일: `Quad.h`** — 원서 Listing 52. `w`도 생성자에서 캐시.

```cpp
mW = n / Dot(n, n);
```

***Listing 52:** [Quad.h] 사각형의 `w` 값 캐싱*

> 📐 **유도 요약**: `p = α·u + β·v`를 `u`, `v`와 각각 외적하면 자기 외적이 0이 되어 `v×p = α(v×u)`, `u×p = β(u×v)`. 양변에 `n`을 내적해 스칼라로 만든 뒤 나누면 위 식이 나온다. `a×b = −b×a`로 분모를 통일하고 `w = n/(n·n)`로 묶은 것이 최종형. (자세한 전개는 원서 "Deriving the Planar Coordinates" 참고.)

---

## UV 좌표로 내부 판정

평면 좌표 `(α, β)`로 사각형 내부 여부를 바로 판정한다. 다음을 만족하면 내부(=히트):

```text
0 ≤ α ≤ 1
0 ≤ β ≤ 1
```

![그림 7: 사각형 좌표](https://raytracing.github.io/images/fig-2.07-quad-coords.jpg)

내부 판정을 `IsInterior`로 분리하고, 내부면 `(α, β)`를 텍스처 좌표 `U, V`로 저장한다.

> 📄 **파일: `Quad.h`** — 원서 Listing 53. 최종 `Quad` 클래스.

```cpp
__device__ bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& rec) const override
{
    double denom = Dot(mNormal, ray.Direction());
    if (fabs(denom) < 1e-8) return false;

    double t = (mD - Dot(mNormal, ray.Origin())) / denom;
    if (t < tMin || t > tMax) return false;

    // 교점의 평면 좌표(alpha, beta)로 내부 판정
    Point3 intersection = ray.At(t);
    Vector3 planarHitptVector = intersection - mQ;
    double alpha = Dot(mW, Cross(planarHitptVector, mV));
    double beta  = Dot(mW, Cross(mU, planarHitptVector));

    if (!IsInterior(alpha, beta, rec)) return false;

    rec.T = t;
    rec.P = intersection;
    rec.MaterialPtr = mMaterial;
    rec.SetFaceNormal(ray, mNormal);
    return true;
}

// 내부면 U,V 채우고 true. (이 판정만 바꾸면 다른 평면 도형으로 확장)
__device__ bool IsInterior(double a, double b, HitRecord& rec) const
{
    Interval unitInterval(0.0, 1.0);
    if (!unitInterval.Contains(a) || !unitInterval.Contains(b)) return false;
    rec.U = a;
    rec.V = b;
    return true;
}
```

***Listing 53:** [Quad.h] 최종 사각형 클래스*

이제 사각형을 시연하는 장면을 추가한다. 앞 장들처럼 **`CreateWorld` 커널의 `sceneId == 4`** 분기로 넣는다.

> 📄 **파일: `kernel.cu`** *(`CreateWorld` 내부)* — 원서 Listing 54.

```cpp
else // sceneId == 4
{
    // 5색 사각형. 각 면을 다른 색 Lambertian으로.
    list[i++] = new Quad(Vector3(-3,-2,5), Vector3(0,0,-4), Vector3(0,4,0),
        new Lambertian(Color(1.0, 0.2, 0.2)));   // left  (red)
    list[i++] = new Quad(Vector3(-2,-2,0), Vector3(4,0,0), Vector3(0,4,0),
        new Lambertian(Color(0.2, 1.0, 0.2)));   // back  (green)
    list[i++] = new Quad(Vector3(3,-2,1), Vector3(0,0,4), Vector3(0,4,0),
        new Lambertian(Color(0.2, 0.2, 1.0)));   // right (blue)
    list[i++] = new Quad(Vector3(-2,3,1), Vector3(4,0,0), Vector3(0,0,4),
        new Lambertian(Color(1.0, 0.5, 0.0)));   // upper (orange)
    list[i++] = new Quad(Vector3(-2,-3,5), Vector3(4,0,0), Vector3(0,0,-4),
        new Lambertian(Color(0.2, 0.8, 0.8)));   // lower (teal)

    lookfrom = Vector3(0, 0, 9); vfov = 80.0; aperture = 0.0;
}
```

```cpp
// main(): int sceneId = 4;   // 사각형 장면 보기
```

***Listing 54:** [kernel.cu] 사각형을 포함한 새로운 장면*

빌드·실행 결과(우리 프로젝트, 실제 렌더):

![이미지 16: 쿼드](https://raytracing.github.io/images/img-2.16-quads.png)

> 🔧 **종횡비 메모**: 원서는 정사각형(`aspect 1.0`, 400×400)으로 렌더하지만, 우리 출력은 1440×720(2:1) 고정이라 카메라 `aspect`도 2:1이 된다. 같은 사각형들이 가로로 조금 더 넓게 보이지만, 배치·색·교차 판정은 책과 동일하다(중앙 초록, 좌 빨강, 우 파랑, 위 주황, 아래 청록).

---

## 추가 2D 도형 (확장)

`(α, β)` 좌표로 내부 판정을 하므로, **`IsInterior`만 바꾸면** 다른 평면 도형이 된다.

- **원판(disk)**: `sqrt(a*a + b*b) < r`
- **삼각형**: `a > 0 && b > 0 && a + b < 1`

텍스처 픽셀로 오려낸 스텐실이나 망델브로 형태도 가능하다. 우리 `Quad::IsInterior`도 같은 자리만 고치면 그대로 확장된다.

---

## 결과 & 검증

- **빌드/실행 확인**: VS2022 + CUDA 12.9로 **컴파일·링크·실행 성공**. `sceneId = 4`로 렌더 시 GPU 약 **0.27초**, 책의 Image 16과 **동일한 5색 사각형** 결과를 확인했다(중앙 초록/좌 빨강/우 파랑/위 주황/아래 청록).
- **BVH/텍스처와 호환**: `Quad`도 `Hittable`이라 기존 BVH·HitRecord(U,V)·재질 파이프라인에 그대로 들어간다.
- **AABB 패딩**: 평평한 사각형의 두께 0 축이 `PadToMinimums`로 보정되어, 축 평면에 놓인 사각형도 BVH 컬링이 정상 동작한다.

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| 48 | `AABB.h` | `PadToMinimums()` 추가, 두 생성자에서 호출 |
| 49, 50, 51, 52, 53 | `Quad.h` *(신규)* | 사각형 프리미티브. `shared_ptr`→`Material*`, `Hit`은 tMin/tMax |
| 54 | `kernel.cu` | `sceneId == 4`(quads) 분기 + main `sceneId` |
| — | `RayTracinginOneWeekend.vcxproj` | `Quad.h` 등록 |

### CUDA 적용에서 꼭 기억할 3가지

1. **수학 도형이라 포팅이 단순**: 평면 방정식·평면 좌표는 디바이스 수학(`Cross`/`Dot`/`fabs`)만 쓰므로 원서와 거의 동일. `shared_ptr`→`Material*`, `Hit`만 우리 시그니처로.
2. **두께 0 AABB 패딩 필수**: 축 평면에 놓인 사각형은 한 축 두께가 0 → BVH 슬랩 검사에서 문제. `Aabb::PadToMinimums`로 보정(원서 Listing 48).
3. **`IsInterior` 한 곳으로 확장**: `(α,β)` 판정만 바꾸면 원판/삼각형 등으로 확장된다.
```
