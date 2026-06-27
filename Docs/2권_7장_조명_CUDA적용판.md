# Lights (조명) — CUDA 적용판

> *Ray Tracing: The Next Week* 7장(Lights)을 우리 **CUDA + 레이트레이싱 프로젝트**에 맞춰 다시 정리한 문서.
> 원서의 설명을 그대로 살리되, 코드는 우리 프레임워크(GPU 디바이스 코드)에 맞춰 바꿔 적었다. 실제로 빌드·실행해 직사각형 광원·구 광원·코넬 박스를 확인했다.

---

조명은 레이트레이싱의 핵심이다. 초기의 단순한 레이트레이서는 공간상의 점이나 방향 같은 **추상적 광원**을 썼지만, 현대적 방식은 **위치와 크기를 가진 물리 기반 조명**을 쓴다. 이런 광원을 만들려면 **아무 일반 물체나 빛을 방출하는 물체로 바꿀 수 있어야** 한다.

> 💡 **이 장에서 CUDA 때문에 달라지는 큰 그림**
> 1. 원서의 재귀 `ray_color`를 우리는 이미 **반복(iterative) 루프**로 풀고 있다. 여기에 **방출광 누적**을 더한다(아래 RayColor).
> 2. 배경색은 카메라에 저장해 디바이스에서 읽는다(`(*camera)->Background()`).
> 3. PPM 출력 시 **색 클램프가 필수**가 된다 — 광원 색이 1.0을 넘어, 클램프하지 않으면 PPM에 255 초과 값이 찍혀 파일이 깨진다(실제로 부딪힌 버그, 본문 콜아웃).

---

## 발광 재질 (Emissive Materials)

빛을 방출하는 재질을 만든다. 재질 기본 클래스에 `Emitted(u, v, p)`를 추가한다. 배경처럼 레이에 **색만 알려주고 반사는 하지 않는다**. 비발광 재질이 모두 `Emitted`를 구현하지 않아도 되도록, 기본 클래스는 **검정**을 반환한다.

> 📄 **파일: `Material.h`** — 원서 Listing 55·56. `shared_ptr<texture>` → `Texture*`.

```cpp
// 재질 기본 클래스: Emitted 기본값은 검정(비발광).
class Material
{
public:
    __device__ virtual Color Emitted(double u, double v, const Point3& p) const
    {
        return Color(0.0, 0.0, 0.0);
    }

    __device__ virtual bool Scatter(
        const Ray& rayIn, const HitRecord& rec,
        Color& attenuation, Ray& scattered, curandState* randState) const = 0;
};

// 확산 광원: 방출색을 텍스처로 조회하고, 산란은 하지 않는다.
class DiffuseLight : public Material
{
public:
    __device__ DiffuseLight(Texture* texture) : mTexture(texture) {}
    __device__ DiffuseLight(const Color& emit) : mTexture(new SolidColor(emit)) {}

    __device__ Color Emitted(double u, double v, const Point3& p) const override
    {
        return mTexture->Value(u, v, p);
    }

    __device__ bool Scatter(const Ray&, const HitRecord&, Color&, Ray&, curandState*) const override
    {
        return false;   // 빛은 산란하지 않음
    }

private:
    Texture* mTexture;
};
```

***Listing 55·56:** [Material.h] 확산 광원 + 기본 Emitted*

---

## RayColor에 배경색 + 방출광 추가

장면의 빛이 **광원에서만** 나오도록, 레이가 아무것도 못 맞추면 **배경색**을 반환한다(빛 장면은 검정). 그리고 매 히트마다 **방출광을 누적**한다.

원서는 재귀로 짧게 쓰지만, 우리 `RayColor`는 GPU 스택을 아끼려 **반복 루프**다. 재귀식

```text
result = emit_0 + atten_0·(emit_1 + atten_1·(emit_2 + ...))
```

를 누적변수 두 개(`throughput`=누적 감쇠, `accumulated`=모은 방출광)로 펼친다.

> 📄 **파일: `kernel.cu`** — 원서 Listing 57에 대응.

```cpp
__device__ Color RayColor(const Ray& r, const Color& background, Hittable** world, curandState* randState)
{
    Ray currentRay = r;
    Color throughput(1.0, 1.0, 1.0);
    Color accumulated(0.0, 0.0, 0.0);

    for (int i = 0; i < 50; i++)
    {
        HitRecord rec;
        if (!(*world)->Hit(currentRay, 0.001, DBL_MAX, rec))
        {
            accumulated += throughput * background;   // 아무것도 안 맞음 → 배경
            return accumulated;
        }

        Color emission = rec.MaterialPtr->Emitted(rec.U, rec.V, rec.P);
        accumulated += throughput * emission;         // 방출광 누적

        Ray scattered;
        Color attenuation;
        if (!rec.MaterialPtr->Scatter(currentRay, rec, attenuation, scattered, randState))
            return accumulated;                       // 빛/흡수 → 모은 색 반환

        throughput = throughput * attenuation;
        currentRay = scattered;
    }
    return accumulated;
}
```

***Listing 57:** [kernel.cu] 배경색 + 발광 재질을 포함한 RayColor*

> 🔧 **기존 하늘 그라데이션은 제거**된다. 원서처럼 장면마다 **단색 배경**을 넘긴다. 그래서 기존 장면(0~4)도 하늘이 그라데이션 → **납작한 푸르스름한 흰색**(0.70, 0.80, 1.00)으로 바뀐다(원서 Listing 58과 동일).

배경은 카메라에 저장해 두고 디바이스에서 읽는다.

> 📄 **파일: `Camera.h`** — 배경색 멤버 + 게터 추가. 생성자 마지막 인자로 받는다(기본값 하늘색).

```cpp
__device__ Camera(..., double time1 = 0.0,
                  Color background = Color(0.70, 0.80, 1.00))
{
    mBackground = background;
    /* ... 기존 ... */
}

__device__ const Color& Background() const { return mBackground; }
```

`Render` 커널은 `Color background = (*camera)->Background();`로 읽어 `RayColor`에 넘긴다.

> ⚠️ **PPM 클램프 버그 (실제로 부딪힌 문제)**
> 광원 색은 `(4,4,4)`·`(15,15,15)`처럼 1.0을 훨씬 넘는다. 감마 보정 후에도 픽셀 값이 1.0을 초과할 수 있는데, 기존 PPM 출력은 `int(255.99 * col)`로 **클램프 없이** 변환해서 **255를 넘는 값(예: 511)** 이 파일에 찍혔다. 그러면 PPM 헤더(maxval 255)와 안 맞아 **파일이 깨진다**(뷰어가 "Channel value too large" 에러). 비발광 장면(0~4)에선 색이 ≤1이라 안 드러나다가, 빛 장면에서 처음 터졌다.
> → 해결: 출력 시 각 채널을 `[0, 0.999]`로 **클램프**한 뒤 `int(256.0 * c)`로 변환한다(원서 `write_color`의 `interval(0.000,0.999).clamp`와 동일).

```cpp
double r = col.X() < 0.0 ? 0.0 : (col.X() > 0.999 ? 0.999 : col.X());
// g, b 동일 ...
int ir = int(256.0 * r);
```

---

## 물체를 빛으로 — 직사각형/구 광원

직사각형(quad)을 광원으로 두고, 펄린 구 장면에 올린다. 배경은 검정이라 장면의 유일한 빛은 이 광원뿐이다. 광원은 `(1,1,1)`보다 밝아야 주변을 비출 수 있어 `(4,4,4)`를 준다.

> 📄 **파일: `kernel.cu`** *(`CreateWorld`, `sceneId == 5`)* — 원서 Listing 59·60. 구 광원까지 추가한 최종형.

```cpp
else if (sceneId == 5)   // simple_light
{
    Texture* pertext = new NoiseTexture(4.0, &localRandState);
    list[i++] = new Sphere(Vector3(0,-1000,0), 1000, new Lambertian(pertext));
    list[i++] = new Sphere(Vector3(0, 2, 0), 2, new Lambertian(pertext));

    Material* diffLight = new DiffuseLight(Color(4, 4, 4));
    list[i++] = new Sphere(Vector3(0, 7, 0), 2, diffLight);                 // 구 광원
    list[i++] = new Quad(Vector3(3,1,-2), Vector3(2,0,0), Vector3(0,2,0), diffLight); // 사각형 광원

    background = Color(0, 0, 0);
    lookfrom = Vector3(26, 3, 6); lookat = Vector3(0, 2, 0); vfov = 20.0;
}
```

***Listing 59·60:** [kernel.cu] 직사각형 광원 + 조명용 구*

![이미지 17: 직사각형 광원이 있는 장면](https://raytracing.github.io/images/img-2.17-rect-light.png)

![이미지 18: 직사각형 + 구 광원이 있는 장면](https://raytracing.github.io/images/img-2.18-rect-sphere-light.png)

---

## 빈 코넬 박스 (Cornell Box)

"코넬 박스"는 1984년 확산 표면 간 빛 상호작용을 모델링하려고 도입된 고전 장면이다. 5개 벽과 천장 광원을 만든다.

> 📄 **파일: `kernel.cu`** *(`CreateWorld`, `sceneId == 6`)* — 원서 Listing 61.

```cpp
else   // sceneId == 6, cornell_box
{
    Material* red   = new Lambertian(Color(0.65, 0.05, 0.05));
    Material* white = new Lambertian(Color(0.73, 0.73, 0.73));
    Material* green = new Lambertian(Color(0.12, 0.45, 0.15));
    Material* light = new DiffuseLight(Color(15, 15, 15));

    list[i++] = new Quad(Vector3(555,0,0), Vector3(0,555,0), Vector3(0,0,555), green);
    list[i++] = new Quad(Vector3(0,0,0),   Vector3(0,555,0), Vector3(0,0,555), red);
    list[i++] = new Quad(Vector3(343,554,332), Vector3(-130,0,0), Vector3(0,0,-105), light);
    list[i++] = new Quad(Vector3(0,0,0),       Vector3(555,0,0), Vector3(0,0,555), white);
    list[i++] = new Quad(Vector3(555,555,555), Vector3(-555,0,0), Vector3(0,0,-555), white);
    list[i++] = new Quad(Vector3(0,0,555),     Vector3(555,0,0), Vector3(0,555,0), white);

    background = Color(0, 0, 0);
    lookfrom = Vector3(278,278,-800); lookat = Vector3(278,278,0); vfov = 40.0;
}
```

***Listing 61:** [kernel.cu] 빈 코넬 박스 장면*

![이미지 19: 빈 코넬 박스](https://raytracing.github.io/images/img-2.19-cornell-empty.png)

이 이미지는 **광원이 작아** 대부분의 무작위 레이가 광원에 닿지 못하므로 노이즈가 심하다.

> 🔧 **샘플 수 메모**: 빛 장면은 노이즈가 심해, 우리 `main()`에서 `sceneId`가 5·6이면 픽셀당 샘플을 **200**으로 올린다(다른 장면은 10). 원서도 100~200을 쓴다. 그래도 코넬 박스는 노이즈가 남는데, 이는 다음 장(중요도 샘플링)에서 개선된다.
>
> 🔧 **종횡비 메모**: 원서는 정사각형(`aspect 1.0`)으로 렌더하지만 우리 출력은 1440×720(2:1)이라 코넬 박스가 가로로 늘어나 보인다. 배치·조명은 동일하다.

---

## 결과 & 검증

- **빌드/실행 확인**: VS2022 + CUDA 12.9로 **컴파일·링크·실행 성공**. 직사각형 광원/구 광원(scene 5), 코넬 박스(scene 6)가 검은 배경에서 광원만으로 조명되는 것을 확인했다.
- **PPM 클램프**: 광원 색이 1.0을 넘어도 출력이 깨지지 않도록 `[0,0.999]` 클램프를 추가(빛 장면 필수).
- **기존 장면 영향**: 하늘 그라데이션이 단색 배경으로 바뀌어 scene 0~4의 하늘이 납작한 푸른빛이 된다(원서와 동일).

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| 55, 56 | `Material.h` | 기본 `Emitted`(검정) + `DiffuseLight` 클래스 |
| 57 | `kernel.cu` (RayColor) | 배경색 인자 + 방출광 누적(반복형) |
| 57 | `Camera.h` | 배경색 멤버 + `Background()` 게터 |
| 58 | `kernel.cu` (CreateWorld) | 장면별 `background` 설정, 카메라에 전달 |
| 59, 60 | `kernel.cu` (scene 5) | simple_light: 사각형/구 광원 |
| 61 | `kernel.cu` (scene 6) | cornell_box: 5벽 + 천장 광원 |
| — | `kernel.cu` (PPM 출력) | 색 `[0,0.999]` 클램프(광원 대응) |

### CUDA 적용에서 꼭 기억할 3가지

1. **방출광은 반복 루프에 누적**: 재귀 `ray_color`를 `throughput`/`accumulated` 두 변수로 펼쳐 방출광을 더한다. `Emitted` 기본값이 검정이라 비발광 재질은 자동으로 무영향.
2. **배경색은 카메라에 실어 디바이스로**: 장면마다 다른 배경(빛 장면=검정)을 카메라 멤버로 저장하고 `Render`가 읽는다.
3. **PPM 출력 클램프 필수**: 광원 색 > 1.0 → 클램프 없으면 255 초과 값으로 파일이 깨진다. `[0,0.999]`로 자른다.
```
