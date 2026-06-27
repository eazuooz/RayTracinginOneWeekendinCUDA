# Volumes (볼륨 / 참여 매질) — CUDA 적용판

> *Ray Tracing: The Next Week* 9장(Volumes)을 우리 **CUDA + 레이트레이싱 프로젝트**에 맞춰 다시 정리한 문서.
> 원서의 설명을 그대로 살리되, 코드는 우리 프레임워크(GPU 디바이스 코드)에 맞춰 바꿔 적었다. 실제로 VS2022 + CUDA 12.9로 빌드·실행해 연기/안개 상자가 든 코넬 박스를 확인했다.

---

레이 트레이서에 **연기·안개·박무**를 넣으면 표현력이 크게 는다. 이런 요소를 **볼륨(volume)** 또는 **참여 매질(participating media)** 이라 한다. 볼륨은 표면과 본질적으로 다른 존재라 보통 소프트웨어 구조를 복잡하게 만든다. 하지만 **볼륨을 "확률적 표면"으로 바꾸는** 깜찍한 기법이 있다 — 볼륨 안의 각 지점에서 확률적으로 있을 수도, 없을 수도 있는 표면으로 연기 덩어리를 대체하는 것이다.

> 💡 **이 장에서 CUDA 때문에 달라지는 큰 그림 — 가장 중요한 한 가지**
> `ConstantMedium::Hit`는 "레이가 매질 안 어디서 산란하는가"를 **난수**로 정한다(`hit_distance = -1/density · log(random)`). 그런데 우리 `Hittable::Hit`에는 난수원이 없었다. → **`Hit` 시그니처에 `curandState* randState`를 추가**해, RayColor가 가진 픽셀별 cuRAND 상태를 BVH·리스트·인스턴스를 거쳐 `ConstantMedium`까지 흘려보낸다. 볼륨이 아닌 표면(구/사각형…)은 이 인자를 받기만 하고 쓰지 않는다.

---

## 일정 밀도 매질 (Constant Density Medium)

일정한 밀도의 볼륨부터 시작한다. 볼륨을 지나는 레이는 **내부에서 산란**하거나, **그대로 통과**한다. 아주 작은 거리 ΔL 안에서 산란할 확률은 `C·ΔL`(C는 광학 밀도에 비례)이다. 미분방정식을 풀면, 균일 난수 하나로부터 **산란이 일어나는 거리**가 나온다. 그 거리가 볼륨 밖이면 "충돌 없음"이다. 일정 밀도 볼륨은 **밀도 C**와 **경계(boundary)** 만 있으면 되며, 경계는 또 다른 `Hittable`로 받는다.

> 📄 **파일: `ConstantMedium.h`** (신규) — 원서 Listing 71. `shared_ptr` → raw 포인터, `random_double()` → `curand_uniform(randState)`, `interval` → `(tMin, tMax)` 시그니처.

```cpp
class ConstantMedium : public Hittable
{
public:
    __device__ ConstantMedium(Hittable* boundary, double density, Texture* tex)
        : mBoundary(boundary), mNegInvDensity(-1.0 / density),
          mPhaseFunction(new Isotropic(tex)) {}

    __device__ ConstantMedium(Hittable* boundary, double density, const Color& albedo)
        : mBoundary(boundary), mNegInvDensity(-1.0 / density),
          mPhaseFunction(new Isotropic(albedo)) {}

    __device__ ~ConstantMedium() override { delete mBoundary; delete mPhaseFunction; }

    __device__ bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& rec,
                        curandState* randState) const override
    {
        HitRecord rec1, rec2;

        // 경계와의 두 교점(들어가는 곳/나가는 곳). 시작점이 매질 내부일 수도 있어
        // 전 구간(universe = -DBL_MAX..DBL_MAX)에서 검사한다.
        if (!mBoundary->Hit(ray, -DBL_MAX, DBL_MAX, rec1, randState)) return false;
        if (!mBoundary->Hit(ray, rec1.T + 0.0001, DBL_MAX, rec2, randState)) return false;

        if (rec1.T < tMin) rec1.T = tMin;       // 레이 구간 [tMin,tMax]로 자르기
        if (rec2.T > tMax) rec2.T = tMax;
        if (rec1.T >= rec2.T) return false;
        if (rec1.T < 0.0)    rec1.T = 0.0;

        double rayLength = ray.Direction().Length();
        double distanceInsideBoundary = (rec2.T - rec1.T) * rayLength;
        double hitDistance = mNegInvDensity * log(curand_uniform(randState));

        if (hitDistance > distanceInsideBoundary) return false;  // 매질을 통과(미스)

        rec.T = rec1.T + hitDistance / rayLength;   // 매질 내부에서 산란한 지점
        rec.P = ray.At(rec.T);
        rec.Normal = Vector3(1, 0, 0);   // 임의(등방성이라 의미 없음)
        rec.bFrontFace = true;           // 임의
        rec.MaterialPtr = mPhaseFunction;
        return true;
    }

    __device__ Aabb BoundingBox() const override { return mBoundary->BoundingBox(); }

private:
    Hittable* mBoundary;
    double mNegInvDensity;
    Material* mPhaseFunction;
};
```

***Listing 71:** [ConstantMedium.h] 상수 매질 클래스*

> 🔧 **`curand_uniform`과 `log(0)`**: 원서는 `random_double()`(보통 `[0,1)`)을 쓰지만, cuRAND의 `curand_uniform`은 **`(0, 1]`** 을 반환한다 — 0을 절대 내지 않으므로 `log(0) = -∞` 문제가 없다(되레 더 안전하다). 1.0을 낼 수는 있는데 `log(1)=0 → hit_distance=0`이라 무해하다.
>
> ⚠️ **볼록(convex) 경계 가정**: 이 구현은 레이가 경계를 벗어나면 바깥으로 영원히 진행한다고 가정한다(상자·구는 OK, 토러스·공동 포함 형태는 불가). 원서와 동일한 한계다.
>
> 🔧 **소유권/해제**: `ConstantMedium`이 경계 체인과 위상함수(Isotropic)를 **소유**한다. `FreeWorld`의 `delete list[i]`가 `~ConstantMedium`을 부르면, `delete mBoundary`가 8장에서 만든 `Translate→RotateY→HittableList→Quad 6개` 연쇄를 그대로 해제한다([[2권_8장_인스턴스]]의 가상 소멸자 덕분).

---

## 등방성 산란 (Isotropic)

볼륨이 산란할 때 쓰는 **위상 함수(phase function)**. 입사 방향과 무관하게 **균일한 무작위 방향**으로 산란시킨다.

> 📄 **파일: `Material.h`** — 원서 Listing 72. `random_unit_vector()` → `UnitVector(RandomInUnitSphere(randState))`. `Scatter`가 이미 `randState`를 받으므로 추가 변경은 없다.

```cpp
class Isotropic : public Material
{
public:
    __device__ Isotropic(const Color& albedo) : mTexture(new SolidColor(albedo)) {}
    __device__ Isotropic(Texture* texture)    : mTexture(texture) {}

    __device__ bool Scatter(const Ray& rayIn, const HitRecord& rec,
                           Color& attenuation, Ray& scattered,
                           curandState* randState) const override
    {
        scattered = Ray(rec.P, UnitVector(RandomInUnitSphere(randState)), rayIn.Time());
        attenuation = mTexture->Value(rec.U, rec.V, rec.P);
        return true;
    }

private:
    Texture* mTexture;
};
```

***Listing 72:** [Material.h] 등방성 클래스*

---

## CUDA 핵심: `Hit`에 난수원(randState) 전달

이 장의 **유일하면서 가장 큰 CUDA 구조 변경**이다. `ConstantMedium::Hit`가 `curand_uniform`을 호출해야 하므로, 모든 `Hittable`의 `Hit` 가상 함수에 `curandState* randState`를 추가했다.

> 📄 **파일: `Hittable.h`** — 베이스 시그니처 변경.

```cpp
class Hittable
{
public:
    __device__ virtual ~Hittable() {}
    __device__ virtual bool Hit(const Ray& ray, double tMin, double tMax,
                               HitRecord& hitRecord, curandState* randState) const = 0;
    __device__ virtual Aabb BoundingBox() const = 0;
    __device__ virtual bool IsBvhNode() const { return false; }
};
```

이 한 줄이 아래 호출 사슬을 따라 전파된다. **컨테이너/인스턴스는 자식 `Hit`로 그대로 넘기고**, 잎(표면)은 받기만 한다.

```text
RayColor (kernel.cu)  ──randState──►  BvhNode::Hit
   BvhNode::Hit        ──randState──►  kid->Hit (자식 노드/잎)
   HittableList::Hit   ──randState──►  mList[i]->Hit
   Translate::Hit      ──randState──►  mObject->Hit
   RotateY::Hit        ──randState──►  mObject->Hit
   ConstantMedium::Hit ──randState──►  mBoundary->Hit  +  curand_uniform(randState) 사용
   Sphere / MovingSphere / Quad::Hit   (받기만 함, 미사용)
```

| 파일 | 변경 |
|---|---|
| `Hittable.h` | 베이스 `Hit`에 `curandState*` 추가 |
| `BvhNode.h` | 시그니처 + `kid->Hit(..., randState)` 전달 |
| `HittableList.h` | 시그니처 + `mList[i]->Hit(..., randState)` 전달 |
| `Instance.h` (`Translate`/`RotateY`) | 시그니처 + 자식 `Hit`로 전달 |
| `Sphere.h` / `MovingSphere.h` / `Quad.h` | 시그니처만(미사용) |
| `kernel.cu` (`RayColor`) | `(*world)->Hit(currentRay, 0.001, DBL_MAX, rec, randState)` |

> 💡 `Render` 커널은 이미 픽셀별 `curandState localRandState`를 갖고 `RayColor`에 넘긴다. `RayColor`는 그 포인터를 그대로 `world->Hit`에 전달하므로, **새 RNG 상태를 만들 필요 없이** 기존 픽셀 RNG를 재사용한다(산란 표본과 같은 난수열을 공유).

---

## 연기·안개 코넬 박스 (sceneId == 8)

8장의 두 회전 상자를 `ConstantMedium`으로 감싼다. 더 빠른 수렴을 위해 **광원을 더 크고(113,554,127~) 어둡게((7,7,7))** 잡는다(원서와 동일). 빈 코넬 박스(scene 6)·회전 상자(scene 7)는 그대로 보존하고 새 `sceneId == 8`로 추가했다.

> 📄 **파일: `kernel.cu`** *(`CreateWorld`, `sceneId == 8`)* — 원서 Listing 73의 `cornell_smoke()`.

```cpp
else  // sceneId == 8, cornell_smoke
{
    Material* red   = new Lambertian(Color(0.65, 0.05, 0.05));
    Material* white = new Lambertian(Color(0.73, 0.73, 0.73));
    Material* green = new Lambertian(Color(0.12, 0.45, 0.15));
    Material* light = new DiffuseLight(Color(7, 7, 7));   // 더 크고 어두운 광원

    list[i++] = new Quad(Vector3(555,0,0), Vector3(0,555,0), Vector3(0,0,555), green);
    list[i++] = new Quad(Vector3(0,0,0),   Vector3(0,555,0), Vector3(0,0,555), red);
    list[i++] = new Quad(Vector3(113,554,127), Vector3(330,0,0), Vector3(0,0,305), light);
    list[i++] = new Quad(Vector3(0,555,0), Vector3(555,0,0), Vector3(0,0,555), white);
    list[i++] = new Quad(Vector3(0,0,0),   Vector3(555,0,0), Vector3(0,0,555), white);
    list[i++] = new Quad(Vector3(0,0,555), Vector3(555,0,0), Vector3(0,555,0), white);

    Hittable* box1 = MakeBox(Point3(0,0,0), Point3(165,330,165), white);
    box1 = new RotateY(box1, 15.0);
    box1 = new Translate(box1, Vector3(265,0,295));
    list[i++] = new ConstantMedium(box1, 0.01, Color(0,0,0));   // 어두운 연기

    Hittable* box2 = MakeBox(Point3(0,0,0), Point3(165,165,165), white);
    box2 = new RotateY(box2, -18.0);
    box2 = new Translate(box2, Vector3(130,0,65));
    list[i++] = new ConstantMedium(box2, 0.01, Color(1,1,1));   // 밝은 안개

    background = Color(0,0,0);
    lookfrom = Vector3(278,278,-800); lookat = Vector3(278,278,0); vfov = 40.0;
}
```

***Listing 73:** [kernel.cu] 연기가 있는 코넬 박스*

> 🔧 **BVH 관점**: `list[]`에 들어가는 잎은 6벽 + **2개의 `ConstantMedium`** = 8개다. 각 `ConstantMedium`의 `BoundingBox()`는 경계 상자(회전·이동 반영)를 그대로 돌려주므로, BVH가 평소처럼 잎 하나로 묶는다.
>
> 🔧 **샘플/종횡비**: scene 8도 빛 장면이라 `main()`에서 샘플을 200으로 잡는다(`sceneId 5~8`). 원서는 1:1로 렌더하지만 우리 출력은 1440×720(2:1)이라 가로로 늘어나 보인다.

---

## 결과 & 검증

VS2022 + CUDA 12.9로 **컴파일·링크·실행 성공**(1440×720, 200 spp). 코넬 박스 안에서 **키 큰 상자가 어두운 연기**로, **키 작은 상자가 밝은 안개**로 보이며, 경계가 또렷한 표면이 아니라 **부드럽게 흩어지는 입자**로 렌더되는 것을 확인했다(원서 Image 22와 동일 구도).

![이미지 22: 연기 블록이 있는 코넬 박스](https://raytracing.github.io/images/img-2.21-cornell-smoke.png)

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| 71 | `ConstantMedium.h` (신규) | 볼륨 = 확률적 표면. `random_double` → `curand_uniform`, 경계/위상함수 소유·연쇄 해제 |
| 72 | `Material.h` (`Isotropic`) | 균일 무작위 방향 산란, 감쇠는 텍스처 |
| — | `Hittable.h` | **`Hit`에 `curandState*` 추가**(이 장의 핵심 구조 변경) |
| — | `BvhNode.h`·`HittableList.h`·`Instance.h` | `randState`를 자식 `Hit`로 전달 |
| — | `Sphere.h`·`MovingSphere.h`·`Quad.h` | `Hit` 시그니처만 맞춤(미사용) |
| 73 | `kernel.cu` (scene 8) | cornell_smoke: 두 상자를 연기/안개로(scene 6·7은 보존) |

### CUDA 적용에서 꼭 기억할 3가지

1. **볼륨 = 확률적 표면**: 매질 안 산란 지점을 `hit_distance = -1/density·log(random)`으로 정한다. 그 거리가 경계 밖이면 레이는 통과(미스).
2. **난수원을 `Hit`로 흘려보낸다**: `ConstantMedium`이 `Hit`에서 난수를 쓰므로, `Hittable::Hit`에 `curandState*`를 추가하고 BVH·리스트·인스턴스가 자식으로 전달한다. RayColor가 가진 **픽셀 RNG를 그대로 재사용**한다.
3. **`curand_uniform`은 `(0,1]`**: `log(0)=-∞` 위험이 없어 원서의 `random_double`보다 안전하다. 소유권은 가상 소멸자로 연쇄 해제([[2권_8장_인스턴스]]와 동일 패턴).
```
