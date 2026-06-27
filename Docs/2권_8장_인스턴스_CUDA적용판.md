# Instances (인스턴스) — CUDA 적용판

> *Ray Tracing: The Next Week* 8장(Instances)을 우리 **CUDA + 레이트레이싱 프로젝트**에 맞춰 다시 정리한 문서.
> 원서의 설명을 그대로 살리되, 코드는 우리 프레임워크(GPU 디바이스 코드)에 맞춰 바꿔 적었다. 실제로 VS2022 + CUDA 12.9로 빌드·실행해 회전·이동된 두 상자가 들어간 코넬 박스를 확인했다.

---

코넬 박스에는 보통 **두 개의 직육면체(블록)** 가 들어간다. 이 블록들은 벽에 대해 **회전**되어 있다. 먼저 6개의 사각형으로 직육면체를 만드는 함수를 만들고, 그 다음 **인스턴스(instance)** 로 이동·회전시킨다.

> 💡 **이 장에서 CUDA 때문에 달라지는 큰 그림**
> 1. 인스턴스는 새 프리미티브를 만들지 않는다 — **레이를 반대로 변환**해 같은 메시를 옮기거나 돌린다(좌표계 변환). 이건 CPU/GPU가 동일하다.
> 2. 원서의 `shared_ptr<hittable>` 대신 **raw `Hittable*`**. 인스턴스가 자식을 **소유**하고, **가상 소멸자**로 `delete`가 자식까지 연쇄되게 한다 → `Hittable`에 가상 소멸자를 추가했다.
> 3. 상자(`MakeBox`)는 6개 Quad를 **소유한 `HittableList`** 로 만든다. 이 리스트를 `RotateY`/`Translate`로 감싸 `list[]`에 한 칸으로 넣는다. **BVH는 이 최상위 래퍼를 잎(leaf) 하나**로 본다.
> 4. `Pi`/`Infinity`는 호스트 전용 `RtWeekend.h`에 있어 디바이스에서 못 쓴다 → 라디안은 상수를 직접 곱하고, 무한대 대용으로 `DBL_MAX`를 쓴다.

---

## 상자 만들기 (Box)

두 대각 꼭짓점 `a`, `b`로 정의되는 직육면체(6면)를 Quad 6개로 만든다. 원서는 `box()`가 `hittable_list`를 반환한다. 우리는 6개 Quad와 **그 포인터 배열을 소유하는** `HittableList*`를 반환한다(소멸자 연쇄 해제를 위해).

> 📄 **파일: `Instance.h`** — 원서 Listing 62의 `box()`. `make_shared<quad>` → 디바이스 `new Quad`, 반환은 `bOwns=true` 리스트.

```cpp
__device__ inline Hittable* MakeBox(const Point3& a, const Point3& b, Material* mat)
{
    Point3 min(fmin(a.X(), b.X()), fmin(a.Y(), b.Y()), fmin(a.Z(), b.Z()));
    Point3 max(fmax(a.X(), b.X()), fmax(a.Y(), b.Y()), fmax(a.Z(), b.Z()));

    Vector3 dx(max.X() - min.X(), 0, 0);
    Vector3 dy(0, max.Y() - min.Y(), 0);
    Vector3 dz(0, 0, max.Z() - min.Z());

    Hittable** sides = new Hittable*[6];
    sides[0] = new Quad(Point3(min.X(), min.Y(), max.Z()),  dx,  dy, mat); // front
    sides[1] = new Quad(Point3(max.X(), min.Y(), max.Z()), -dz,  dy, mat); // right
    sides[2] = new Quad(Point3(max.X(), min.Y(), min.Z()), -dx,  dy, mat); // back
    sides[3] = new Quad(Point3(min.X(), min.Y(), min.Z()),  dz,  dy, mat); // left
    sides[4] = new Quad(Point3(min.X(), max.Y(), max.Z()),  dx, -dz, mat); // top
    sides[5] = new Quad(Point3(min.X(), min.Y(), min.Z()),  dx,  dz, mat); // bottom

    return new HittableList(sides, 6, true);  // true = 6개 Quad + 배열을 소유
}
```

***Listing 62:** [Instance.h] 직육면체 객체*

> 🔧 **소유권/해제 메모 (CUDA 핵심)**
> 원서는 `shared_ptr`가 자동으로 메모리를 관리하지만, 우리는 디바이스 `new`를 직접 쓴다. 그래서 `HittableList`에 **소유 플래그**를 추가했다. `bOwns=true`면 소멸자가 내부 Quad 6개와 포인터 배열을 직접 해제한다. 이 리스트를 `RotateY`→`Translate`로 감싸 `list[]`에 넣으면, `FreeWorld`가 **최상위 `Translate`만 `delete`** 해도 소멸자 연쇄로 `RotateY → HittableList(소유) → Quad 6개`까지 한 번씩 깔끔히 풀린다(누수·더블 프리 없음).

회전 없이 두 상자만 넣으면 다음과 같다(원서 Listing 63). 우리 프로젝트에서는 이 단계가 `sceneId == 7`의 중간 형태에 해당한다.

```cpp
list[i++] = MakeBox(Point3(130, 0, 65),  Point3(295, 165, 230), white);
list[i++] = MakeBox(Point3(265, 0, 295), Point3(430, 330, 460), white);
```

---

## 인스턴스란? — 레이를 반대로 옮긴다

**인스턴스**는 장면에 배치된 기하 프리미티브의 복사본이다. 각 복사본은 독립적으로 **이동·회전**할 수 있다. 레이트레이싱이 특히 편한 점은 **물체를 실제로 옮길 필요가 없다**는 것이다. 대신 **레이를 반대 방향으로 옮긴다**.

예를 들어 분홍 상자를 원점에서 x로 +2 옮기고 싶다면, 상자를 그대로 두고 **레이의 원점에서 x를 −2** 하면 된다. 이걸 "이동"으로 보든 "좌표계 변환"으로 보든 자유다.

---

## 이동 인스턴스 (Translate)

추론 순서는 이렇다:
1. 입사 레이를 **−offset** 만큼 뒤로 옮긴다(월드 → 오브젝트 공간).
2. 옮긴 레이로 교차가 있는지(있다면 어디인지) 판정한다.
3. 교점을 **+offset** 만큼 앞으로 되돌린다(오브젝트 → 월드 공간).

3번을 빼먹으면 교점이 옮겨진 레이 경로에 남아 실제 입사 레이와 어긋난다.

> 📄 **파일: `Instance.h`** — 원서 Listing 64·65. `interval ray_t` → 우리 `(tMin, tMax)` 시그니처. `ray.time()` 그대로 보존.

```cpp
class Translate : public Hittable
{
public:
    __device__ Translate(Hittable* object, const Vector3& offset)
        : mObject(object), mOffset(offset)
    {
        // 자식 경계 상자도 옮겨 둔다. 안 옮기면 BVH 슬랩 검사가 엉뚱한 곳을
        // 보고 레이를 조기 기각한다.
        mBBox = object->BoundingBox() + offset;
    }

    __device__ ~Translate() override { delete mObject; }   // 소멸자 연쇄

    __device__ bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& rec) const override
    {
        Ray offsetRay(ray.Origin() - mOffset, ray.Direction(), ray.Time());  // 1) 레이 -offset
        if (!mObject->Hit(offsetRay, tMin, tMax, rec))                        // 2) 교차 검사
            return false;
        rec.P += mOffset;                                                    // 3) 교점 +offset
        return true;
    }

    __device__ Aabb BoundingBox() const override { return mBBox; }

private:
    Hittable* mObject;
    Vector3 mOffset;
    Aabb mBBox;
};
```

***Listing 64·65:** [Instance.h] 이동 인스턴스*

경계 상자도 **반드시 offset만큼 옮겨야** 한다. 안 그러면 BVH가 엉뚱한 위치의 상자를 보고 교차를 trivial하게 기각한다. `object->BoundingBox() + offset`을 위해 `Aabb + Vector3` 연산자를 추가한다.

> 📄 **파일: `AABB.h`** — 원서 Listing 66. 세 축 구간을 각각 민다.

```cpp
__host__ __device__ inline Aabb operator+(const Aabb& bbox, const Vector3& offset)
{
    return Aabb(bbox.X + offset.X(), bbox.Y + offset.Y(), bbox.Z + offset.Z());
}
__host__ __device__ inline Aabb operator+(const Vector3& offset, const Aabb& bbox)
{
    return bbox + offset;
}
```

***Listing 66:** [AABB.h] `aabb + offset` 연산자*

AABB의 각 축은 `Interval`이므로, `Interval`에도 덧셈 연산자가 필요하다.

> 📄 **파일: `Interval.h`** — 원서 Listing 67.

```cpp
__host__ __device__ inline Interval operator+(const Interval& ival, double displacement)
{
    return Interval(ival.Min + displacement, ival.Max + displacement);
}
__host__ __device__ inline Interval operator+(double displacement, const Interval& ival)
{
    return ival + displacement;
}
```

***Listing 67:** [Interval.h] `interval + displacement` 연산자*

---

## 회전 인스턴스 (RotateY)

회전은 이동보다 식이 까다로워, **좌표계 변환**으로 다루는 편이 안전하다(부호 하나만 틀려도 깨진다). Y축을 기준으로 θ만큼 **반시계** 회전하는 식은:

```text
x' = cosθ·x + sinθ·z
z' = -sinθ·x + cosθ·z
```

월드 → 오브젝트 공간으로 보낼 때는 **−θ** 회전(`cos(−θ)=cosθ`, `sin(−θ)=−sinθ`)을 쓴다. 회전은 교점뿐 아니라 **표면 법선**도 같이 돌려야 반사/굴절 방향이 맞는다(법선도 벡터와 같은 식으로 회전).

> 📄 **파일: `Instance.h`** — 원서 Listing 68·69. `degrees_to_radians`는 호스트 전용이라 디바이스에서 **상수를 직접 곱**한다. `infinity` 대신 `DBL_MAX`.

```cpp
class RotateY : public Hittable
{
public:
    __device__ RotateY(Hittable* object, double angle) : mObject(object)
    {
        double radians = angle * 3.1415926535897932385 / 180.0;  // 디바이스 라디안 변환
        mSinTheta = sin(radians);
        mCosTheta = cos(radians);
        mBBox = object->BoundingBox();

        // 자식 AABB의 8개 꼭짓점을 모두 회전시켜, 회전 후를 감싸는 새 AABB를 구한다.
        Point3 min(DBL_MAX, DBL_MAX, DBL_MAX);
        Point3 max(-DBL_MAX, -DBL_MAX, -DBL_MAX);
        for (int i = 0; i < 2; i++)
          for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                double x = i*mBBox.X.Max + (1-i)*mBBox.X.Min;
                double y = j*mBBox.Y.Max + (1-j)*mBBox.Y.Min;
                double z = k*mBBox.Z.Max + (1-k)*mBBox.Z.Min;
                double newx =  mCosTheta*x + mSinTheta*z;
                double newz = -mSinTheta*x + mCosTheta*z;
                Vector3 tester(newx, y, newz);
                for (int c = 0; c < 3; c++) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        mBBox = Aabb(min, max);
    }

    __device__ ~RotateY() override { delete mObject; }

    __device__ bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& rec) const override
    {
        // 레이를 월드 → 오브젝트 공간으로 회전(-θ).
        Point3 origin(
            (mCosTheta*ray.Origin().X()) - (mSinTheta*ray.Origin().Z()),
            ray.Origin().Y(),
            (mSinTheta*ray.Origin().X()) + (mCosTheta*ray.Origin().Z()));
        Vector3 direction(
            (mCosTheta*ray.Direction().X()) - (mSinTheta*ray.Direction().Z()),
            ray.Direction().Y(),
            (mSinTheta*ray.Direction().X()) + (mCosTheta*ray.Direction().Z()));
        Ray rotatedRay(origin, direction, ray.Time());

        if (!mObject->Hit(rotatedRay, tMin, tMax, rec))
            return false;

        // 교점을 오브젝트 → 월드 공간으로 되돌린다(+θ).
        rec.P = Point3(
            (mCosTheta*rec.P.X()) + (mSinTheta*rec.P.Z()),
            rec.P.Y(),
            (-mSinTheta*rec.P.X()) + (mCosTheta*rec.P.Z()));
        // 법선도 같은 식으로 회전.
        rec.Normal = Vector3(
            (mCosTheta*rec.Normal.X()) + (mSinTheta*rec.Normal.Z()),
            rec.Normal.Y(),
            (-mSinTheta*rec.Normal.X()) + (mCosTheta*rec.Normal.Z()));
        return true;
    }

    __device__ Aabb BoundingBox() const override { return mBBox; }

private:
    Hittable* mObject;
    double mSinTheta;
    double mCosTheta;
    Aabb mBBox;
};
```

***Listing 68·69:** [Instance.h] Y축 회전 인스턴스*

> ⚠️ **스케일은 다루지 않는다**: 이동·회전에서는 법선이 벡터와 똑같이 변환되지만, **스케일링**은 법선이 표면에 직교하도록 별도 처리(역전치 행렬)가 필요하다. 이 장 범위 밖이다.

---

## 가상 소멸자 — CUDA 메모리 해제의 핵심

`Translate`/`RotateY`가 자식을 소멸자에서 재귀적으로 `delete` 하려면, 베이스 포인터(`Hittable*`)로 `delete` 할 때 파생 소멸자가 호출돼야 한다. 그래서 베이스에 **가상 소멸자**를 추가했다.

> 📄 **파일: `Hittable.h`** — 원서에는 없는, CUDA 수동 메모리 관리용 추가.

```cpp
class Hittable
{
public:
    __device__ virtual ~Hittable() {}   // Translate→RotateY→HittableList→Quad 연쇄 해제
    __device__ virtual bool Hit(const Ray&, double, double, HitRecord&) const = 0;
    __device__ virtual Aabb BoundingBox() const = 0;
    __device__ virtual bool IsBvhNode() const { return false; }
};
```

이로써 `FreeWorld`의 `delete list[i]` 한 줄이 상자 한 채의 모든 내부 객체를 정확히 한 번씩 해제한다. BVH 내부 노드(`bvhNodes[]`)는 종전대로 따로 해제하므로 더블 프리도 없다.

---

## 코넬 박스에 회전 상자 추가 (sceneId == 7)

원서는 `cornell_box()`를 수정해 회전 상자를 넣는다. 우리는 빈 코넬 박스(`sceneId == 6`)를 **그대로 보존**하고, 회전 상자가 든 버전을 **`sceneId == 7`** 로 새로 추가했다(샘플 수도 200으로 함께 잡는다).

> 📄 **파일: `kernel.cu`** *(`CreateWorld`, `sceneId == 7`)* — 원서 Listing 70.

```cpp
else  // sceneId == 7, cornell_box (두 회전 상자)
{
    Material* red   = new Lambertian(Color(0.65, 0.05, 0.05));
    Material* white = new Lambertian(Color(0.73, 0.73, 0.73));
    Material* green = new Lambertian(Color(0.12, 0.45, 0.15));
    Material* light = new DiffuseLight(Color(15, 15, 15));

    // 5벽 + 천장 광원 (scene 6과 동일)
    list[i++] = new Quad(Vector3(555,0,0), Vector3(0,555,0), Vector3(0,0,555), green);
    list[i++] = new Quad(Vector3(0,0,0),   Vector3(0,555,0), Vector3(0,0,555), red);
    list[i++] = new Quad(Vector3(343,554,332), Vector3(-130,0,0), Vector3(0,0,-105), light);
    list[i++] = new Quad(Vector3(0,0,0),       Vector3(555,0,0), Vector3(0,0,555), white);
    list[i++] = new Quad(Vector3(555,555,555), Vector3(-555,0,0), Vector3(0,0,-555), white);
    list[i++] = new Quad(Vector3(0,0,555),     Vector3(555,0,0), Vector3(0,555,0), white);

    // 키 큰 상자: 15° 회전 후 (265,0,295)로 이동
    Hittable* box1 = MakeBox(Point3(0,0,0), Point3(165,330,165), white);
    box1 = new RotateY(box1, 15.0);
    box1 = new Translate(box1, Vector3(265, 0, 295));
    list[i++] = box1;

    // 키 작은 상자: -18° 회전 후 (130,0,65)로 이동
    Hittable* box2 = MakeBox(Point3(0,0,0), Point3(165,165,165), white);
    box2 = new RotateY(box2, -18.0);
    box2 = new Translate(box2, Vector3(130, 0, 65));
    list[i++] = box2;

    background = Color(0, 0, 0);
    lookfrom = Vector3(278,278,-800); lookat = Vector3(278,278,0); vfov = 40.0;
}
```

***Listing 70:** [kernel.cu] Y축 회전 상자가 든 코넬 박스*

> 🔧 **BVH와 인스턴스**: `list[]`에 들어가는 건 **상자 한 채당 최상위 `Translate` 하나**다(코넬은 6벽 + 2상자 = **8개 잎**). BVH는 각 `Translate`의 `BoundingBox()`(=회전·이동까지 반영된 상자)를 잎으로 보고 묶는다. `DeviceSort`가 `list[]`를 제자리 정렬해도 8개 포인터가 모두 남아 있어 `FreeWorld` 해제에 문제없다.
>
> 🔧 **종횡비 메모**: 원서는 정사각형(1:1)으로 렌더하지만 우리 출력은 1440×720(2:1)이라 코넬 박스가 가로로 늘어나 보인다. 배치·조명·회전각은 동일하다.

---

## 결과 & 검증

VS2022 + CUDA 12.9로 **컴파일·링크·실행 성공**(약 33초, 1440×720, 200 spp). 검은 배경의 코넬 박스 안에 **15° 회전한 키 큰 상자**와 **−18° 회전한 키 작은 상자**가 천장 광원에 조명되는 것을 확인했다(원서 Image 21과 동일 구도).

![이미지 21: 표준 코넬 박스 (회전 상자 2개)](https://raytracing.github.io/images/img-2.20-cornell-standard.png)

작은 광원 탓에 노이즈가 남아 있는데, 이는 다음 장(볼륨)·다음 권(중요도 샘플링)에서 개선된다.

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| 62 | `Instance.h` (`MakeBox`) | 6 Quad를 **소유**하는 `HittableList*` 반환(`bOwns=true`) |
| 64, 65 | `Instance.h` (`Translate`) | 레이 −offset / 교점 +offset, 경계 상자 이동, 소멸자 연쇄 |
| 66 | `AABB.h` | `Aabb + Vector3` 연산자 |
| 67 | `Interval.h` | `Interval + double` 연산자 |
| 68, 69 | `Instance.h` (`RotateY`) | Y축 회전(교점+법선), bbox 8꼭짓점 재계산, `DBL_MAX`/상수 라디안 |
| — | `Hittable.h` | **가상 소멸자** 추가(인스턴스 연쇄 해제용, CUDA 전용) |
| — | `HittableList.h` | **소유 플래그**(`bOwns`) + 소멸자(자식·배열 해제) |
| 70 | `kernel.cu` (scene 7) | 회전 상자 2개가 든 코넬 박스(scene 6은 빈 박스로 보존) |

### CUDA 적용에서 꼭 기억할 3가지

1. **인스턴스 = 레이 역변환**: 물체를 옮기지 않고 레이를 −offset/−θ로 보내 교차한 뒤, 교점(과 법선)을 +offset/+θ로 되돌린다. 이건 CPU와 동일하다.
2. **경계 상자도 같이 변환**: 안 옮기면 BVH가 엉뚱한 곳을 보고 레이를 조기 기각한다. `Translate`는 `+offset`, `RotateY`는 8꼭짓점을 회전해 새 AABB를 만든다.
3. **소유권은 가상 소멸자로**: `shared_ptr`가 없으니 인스턴스가 자식을 소유하고, `Hittable`의 가상 소멸자로 `delete` 한 번이 `Translate→RotateY→HittableList→Quad`까지 연쇄 해제하게 했다.
```
