# Perlin Noise (펄린 노이즈) — CUDA 적용판

> *Ray Tracing: The Next Week* 5장(Perlin Noise)을 우리 **CUDA + 레이트레이싱 프로젝트**에 맞춰 다시 정리한 문서.
> 원서의 단계적 설명을 그대로 살리되, 코드는 우리 프레임워크(GPU 디바이스 코드)에 맞춰 바꿔 적었다. 핵심 차이는 **격자 데이터를 호스트 RNG가 아니라 디바이스 `curand`로 생성**한다는 점이다.

---

멋진 절차적(solid) 텍스처를 만들 때 대부분 **펄린 노이즈(Perlin noise)** 의 일종을 쓴다. 발명자 켄 펄린(Ken Perlin)의 이름에서 왔다. 펄린 텍스처는 아래 같은 **백색 노이즈(white noise)** 를 반환하지 *않는다*:

![이미지 6: 백색 노이즈](https://raytracing.github.io/images/img-2.06-white-noise.jpg)

대신 **흐릿한 백색 노이즈**와 비슷한 것을 반환한다:

![이미지 7: 흐릿한 백색 노이즈](https://raytracing.github.io/images/img-2.07-white-noise-blurred.jpg)

펄린 노이즈의 핵심은 **재현 가능성**이다. 3D 점을 입력하면 항상 같은 "랜덤스러운" 값을 돌려주고, 인접한 점은 비슷한 값을 준다. 또 하나의 특징은 **단순하고 빠르다**는 점이다. 그래서 보통 일종의 "핵(hack)"으로 구현한다. 앤드류 켄슬러(Andrew Kensler)의 설명을 바탕으로 단계적으로 쌓아 올린다.

> 💡 **이 장에서 CUDA 때문에 달라지는 큰 그림**
> 원서의 `perlin` 생성자는 호스트 RNG(`random_double` / `random_int`)로 격자 데이터를 채운다. 하지만 우리는 월드를 **`CreateWorld` 커널 안에서** 만들기 때문에, `Perlin` 생성자가 **`curandState*` 를 받아 디바이스에서** 랜덤 단위 벡터와 순열(permutation)을 만든다. 나머지(노이즈 해싱·보간·난류)는 전부 디바이스 수학 함수(`floor`/`sin`/`fabs`)로 그대로 포팅된다.

---

## 무작위 숫자 블록 사용하기

공간 전체를 무작위 숫자 3D 배열로 타일링해 블록 단위로 쓸 수도 있다. 그러면 반복이 뚜렷한 블록 형태가 나온다:

![이미지 8: 타일링된 무작위 패턴](https://raytracing.github.io/images/img-2.08-tile-random.jpg)

타일링 대신 **해싱**으로 섞는다. 이를 위한 보조 코드가 필요하다. 원서 Listing 34를 우리 디바이스 코드로 옮긴다.

> 📄 **파일: `Perlin.h` (신규)** — 원서 `perlin` 클래스. `random_double`/`random_int` 대신 `curand`. 모든 메서드는 `__device__`.

```cpp
#pragma once
#ifndef PERLIN_H
#define PERLIN_H

#include "Vec3.h"
#include <curand_kernel.h>

class Perlin
{
public:
    // 원서: randfloat[i] = random_double(); + 세 축 순열 생성
    // 우리: curandState로 디바이스에서 채운다(아래는 "랜덤 벡터" 최종형 미리 반영)
    __device__ Perlin(curandState* randState)
    {
        for (int i = 0; i < POINT_COUNT; i++)
            mRandVec[i] = UnitVector(RandomVector(randState, -1.0, 1.0));

        GeneratePerm(mPermX, randState);
        GeneratePerm(mPermY, randState);
        GeneratePerm(mPermZ, randState);
    }

    // (초기형) 블록 해싱 noise — 이후 보간형으로 대체된다.
    // auto i = int(4*p.x()) & 255; ... return randfloat[perm_x[i]^perm_y[j]^perm_z[k]];

private:
    static const int POINT_COUNT = 256;
    Vector3 mRandVec[POINT_COUNT];   // 원서 randfloat[] → 최종적으로 벡터로 바뀜
    int mPermX[POINT_COUNT];
    int mPermY[POINT_COUNT];
    int mPermZ[POINT_COUNT];

    __device__ static void GeneratePerm(int* p, curandState* s)
    {
        for (int i = 0; i < POINT_COUNT; i++) p[i] = i;
        Permute(p, POINT_COUNT, s);
    }

    // 원서 random_int(0,i)를 curand로 대체한 Fisher–Yates 셔플
    __device__ static void Permute(int* p, int n, curandState* s)
    {
        for (int i = n - 1; i > 0; i--)
        {
            int target = int(curand_uniform(s) * (i + 1));  // [0, i]
            if (target > i) target = i;                      // curand_uniform이 1.0일 때 보호
            int tmp = p[i]; p[i] = p[target]; p[target] = tmp;
        }
    }
};
#endif
```

***Listing 34:** [Perlin.h] 펄린 클래스와 보조 함수 (curand 기반)*

> 🔧 **`random_int` → `curand` 변환**: 원서는 `random_int(0, i)`로 `[0,i]` 정수를 뽑는다. `curand_uniform`은 `(0,1]`을 주므로 `int(curand_uniform * (i+1))` 로 `[0,i]`를 만들고, 경계에서 `1.0`이 나올 때만 `i`로 클램프한다. 그래서 우리 `RtWeekend.h`에 별도의 `random_int`를 추가하지 않는다.

이제 0~1 사이 값을 받아 회색을 만드는 텍스처를 만든다.

> 📄 **파일: `Texture.h`** — 원서 Listing 35. `make_shared` 대신 멤버 `Perlin`을 값으로 보유.

```cpp
#include "Perlin.h"

class NoiseTexture : public Texture
{
public:
    // CUDA: Perlin 생성자가 curandState를 받으므로 NoiseTexture도 받는다.
    __device__ NoiseTexture(double scale, curandState* randState)
        : mNoise(randState), mScale(scale) {}

    __device__ Color Value(double u, double v, const Point3& p) const override
    {
        return Color(1, 1, 1) * mNoise.Noise(p);   // (초기형) 회색 노이즈
    }

private:
    Perlin mNoise;
    double mScale;
};
```

***Listing 35:** [Texture.h] 노이즈 텍스처*

이 텍스처를 구 몇 개에 입힌다. 원서 Listing 36의 `perlin_spheres()` 장면을 우리는 **`CreateWorld` 커널의 `sceneId == 3`** 분기로 추가한다(앞 장에서 만든 장면 switch에 한 칸 더).

> 📄 **파일: `kernel.cu`** *(`CreateWorld` 내부)* — 원서 Listing 36.

```cpp
else // sceneId == 3
{
    // perlin_spheres: 두 Lambertian이 같은 NoiseTexture를 공유. scale=4.
    // NoiseTexture 생성자가 localRandState로 격자 벡터/순열을 디바이스에서 만든다.
    Texture* pertext = new NoiseTexture(4.0, &localRandState);
    list[i++] = new Sphere(Vector3(0, -1000, 0), 1000, new Lambertian(pertext));
    list[i++] = new Sphere(Vector3(0,  2,    0),    2, new Lambertian(pertext));

    lookfrom = Vector3(13, 2, 3); vfov = 20.0; aperture = 0.0;
}
```

```cpp
// main()
int sceneId = 3;   // 0: 체커바닥 / 1: 체커구 / 2: 지구 / 3: 펄린(대리석)
```

***Listing 36:** [kernel.cu] 펄린 텍스처 구 두 개로 구성된 장면*

해싱은 기대대로 잘 섞인다:

![이미지 9: 해시된 랜덤 텍스처](https://raytracing.github.io/images/img-2.09-hash-random.png)

---

## 결과 부드럽게 만들기 (Smoothing)

부드럽게 만들려면 **삼선형 보간(trilinear interpolation)** 을 쓴다. `noise()`가 둘러싼 8개 격자 값을 가져와 보간한다.

> 📄 **파일: `Perlin.h`** — 원서 Listing 37.

```cpp
__device__ double Noise(const Point3& p) const
{
    double u = p.X() - floor(p.X());
    double v = p.Y() - floor(p.Y());
    double w = p.Z() - floor(p.Z());

    int i = int(floor(p.X()));
    int j = int(floor(p.Y()));
    int k = int(floor(p.Z()));
    double c[2][2][2];

    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++)
            for (int dk = 0; dk < 2; dk++)
                c[di][dj][dk] = mRandFloat[
                    mPermX[(i + di) & 255] ^
                    mPermY[(j + dj) & 255] ^
                    mPermZ[(k + dk) & 255]];

    return TrilinearInterp(c, u, v, w);
}

__device__ static double TrilinearInterp(double c[2][2][2], double u, double v, double w)
{
    double accum = 0.0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                accum += (i*u + (1-i)*(1-u))
                       * (j*v + (1-j)*(1-v))
                       * (k*w + (1-k)*(1-w))
                       * c[i][j][k];
    return accum;
}
```

***Listing 37:** [Perlin.h] 삼선형 보간을 적용한 펄린*

![이미지 10: 삼선형 보간 펄린 텍스처](https://raytracing.github.io/images/img-2.10-perlin-trilerp.png)

---

## 에르미트 스무딩 (Hermitian Smoothing)

보간은 더 나아졌지만 격자 무늬가 보인다. 일부는 색 선형 보간에서 생기는 지각적 아티팩트인 **마하 밴드(Mach bands)** 다. 표준 트릭은 **에르미트 3차 곡선**으로 보간을 둥글려 주는 것이다.

> 📄 **파일: `Perlin.h`** — 원서 Listing 38. `Noise()`에서 u/v/w를 미리 가공한다.

```cpp
double u = p.X() - floor(p.X());
double v = p.Y() - floor(p.Y());
double w = p.Z() - floor(p.Z());
u = u*u*(3 - 2*u);   // 에르미트 스무딩
v = v*v*(3 - 2*v);
w = w*w*(3 - 2*w);
```

***Listing 38:** [Perlin.h] 에르미트 스무딩을 적용한 펄린*

![이미지 11: 삼선형 보간 + 스무딩](https://raytracing.github.io/images/img-2.11-perlin-trilerp-smooth.png)

---

## 주파수 조정 (Tweaking The Frequency)

무늬가 다소 저주파다. 입력 점을 **스케일**해 더 빨리 변하게 만든다.

> 📄 **파일: `Texture.h`** — 원서 Listing 39·40. `mScale`을 곱해 호출한다.

```cpp
__device__ Color Value(double u, double v, const Point3& p) const override
{
    return Color(1, 1, 1) * mNoise.Noise(mScale * p);
}
```

`pertext`를 `new NoiseTexture(4.0, &localRandState)`로 만들면(위 Listing 36) scale=4가 적용된다.

![이미지 12: 고주파 펄린 텍스처](https://raytracing.github.io/images/img-2.12-perlin-hifreq.png)

---

## 격자 점에 무작위 벡터 사용하기

아직 약간 블록처럼 보이는 이유는 패턴의 최솟/최댓값이 항상 정수 격자 위에 정확히 떨어지기 때문이다. 켄 펄린의 기발한 트릭은 격자 점에 단순 float 대신 **무작위 단위 벡터**를 놓고, **내적(dot product)** 으로 최솟/최댓값을 격자에서 벗어나게 하는 것이다. 그래서 `randfloat` 를 `randvec`(단위 벡터)로 바꾼다.

> 📄 **파일: `Perlin.h`** — 원서 Listing 41·42. 우리는 처음부터 이 최종형으로 작성했다(`mRandVec`).

```cpp
// 생성자: 격자마다 무작위 "단위 벡터"
for (int i = 0; i < POINT_COUNT; i++)
    mRandVec[i] = UnitVector(RandomVector(randState, -1.0, 1.0));

// [min,max]^3 안의 랜덤 벡터 (균일일 필요 없음). 원서 vec3::random(-1,1) 대응.
__device__ static Vector3 RandomVector(curandState* s, double min, double max)
{
    double range = max - min;
    return Vector3(min + range*curand_uniform(s),
                   min + range*curand_uniform(s),
                   min + range*curand_uniform(s));
}
```

`Noise()`는 이제 `double c[2][2][2]` 대신 `Vector3 c[2][2][2]`를 모아 `PerlinInterp`로 보간한다.

> 📄 **파일: `Perlin.h`** — 원서 Listing 43.

```cpp
__device__ static double PerlinInterp(const Vector3 c[2][2][2], double u, double v, double w)
{
    double uu = u*u*(3 - 2*u);
    double vv = v*v*(3 - 2*v);
    double ww = w*w*(3 - 2*w);
    double accum = 0.0;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
            {
                Vector3 weightV(u - i, v - j, w - k);
                accum += (i*uu + (1-i)*(1-uu))
                       * (j*vv + (1-j)*(1-vv))
                       * (k*ww + (1-k)*(1-ww))
                       * Dot(c[i][j][k], weightV);
            }
    return accum;
}
```

***Listing 41·42·43:** [Perlin.h] 무작위 단위 벡터 + 격자 벡터 보간*

펄린 보간의 출력은 **음수**가 될 수 있다. 이 값이 감마 보정 함수(양수만 기대)로 가면 곤란하므로, `[-1,1]` 을 `[0,1]` 로 매핑한다.

> 📄 **파일: `Texture.h`** — 원서 Listing 44.

```cpp
__device__ Color Value(double u, double v, const Point3& p) const override
{
    return Color(1, 1, 1) * 0.5 * (1.0 + mNoise.Noise(mScale * p));
}
```

***Listing 44:** [Texture.h] [-1,1] → [0,1] 매핑*

![이미지 13: 정수에서 벗어난 펄린 텍스처](https://raytracing.github.io/images/img-2.13-perlin-shift.png)

---

## 난류 (Turbulence)

여러 주파수를 합산한 복합 노이즈를 흔히 **난류(turbulence)** 라 부른다. 노이즈를 주파수를 2배씩 올리며 가중 합산한다.

> 📄 **파일: `Perlin.h`** — 원서 Listing 45.

```cpp
__device__ double Turb(const Point3& p, int depth) const
{
    double accum = 0.0;
    Point3 tempP = p;
    double weight = 1.0;

    for (int i = 0; i < depth; i++)
    {
        accum += weight * Noise(tempP);
        weight *= 0.5;
        tempP *= 2.0;     // Vec3::operator*=(double) 사용
    }
    return fabs(accum);
}
```

***Listing 45:** [Perlin.h] 난류 함수*

> 📄 **파일: `Texture.h`** — 원서 Listing 46. 난류를 직접 쓰면 위장막 같은 무늬가 된다.

```cpp
__device__ Color Value(double u, double v, const Point3& p) const override
{
    return Color(1, 1, 1) * mNoise.Turb(p, 7);
}
```

***Listing 46:** [Texture.h] 난류를 적용한 노이즈 텍스처*

![이미지 14: 난류가 적용된 펄린 텍스처](https://raytracing.github.io/images/img-2.14-perlin-turb.png)

---

## 위상 조정 — 대리석 (Adjusting the Phase)

보통 난류는 **간접적으로** 쓴다. 절차적 솔리드 텍스처의 "헬로 월드"는 단순한 **대리석(marble)** 무늬다. 색을 `sin` 함수에 비례시키고, 난류로 그 **위상(phase)** 을 흔들어 줄무늬가 물결치게 만든다. 이것이 우리 프로젝트에 **실제로 적용한 최종형**이다.

> 📄 **파일: `Texture.h`** — 원서 Listing 47.

```cpp
__device__ Color Value(double u, double v, const Point3& p) const override
{
    // 대리석: color(.5,.5,.5) * (1 + sin(scale*z + 10*turb(p,7)))
    return Color(0.5, 0.5, 0.5)
        * (1.0 + sin(mScale * p.Z() + 10.0 * mNoise.Turb(p, 7)));
}
```

***Listing 47:** [Texture.h] 대리석 무늬 노이즈 텍스처*

![이미지 15: 펄린 노이즈, 대리석 질감](https://raytracing.github.io/images/img-2.15-perlin-marble.png)

> 🔧 **단계 전환은 한 줄**: 위 `Value`의 반환식만 바꾸면 챕터의 각 단계(부드러운 노이즈 / 난류 직접 / 대리석)를 오갈 수 있다. 우리 `Texture.h`에는 최종형(대리석)을 기본으로 두고, 나머지 두 형태를 주석으로 함께 남겨 두었다.

---

## 결과 & 검증

- **빌드/실행 확인**: VS2022 + CUDA 12.9로 **컴파일·링크·실행 성공**. (커맨드라인 `nvcc`는 VS2026 환경에서 `cudafe++`가 죽는 별개 이슈가 있으니, **VS2022 솔루션으로 빌드**해야 한다.)
- **디바이스 RNG**: 격자 벡터/순열이 `curand`로 디바이스에서 채워지므로, 월드를 GPU에서 만드는 우리 구조와 자연스럽게 맞물린다.
- **객체 크기**: `Perlin`은 256개 벡터 + 3×256개 정수(약 9KB)를 멤버로 갖는다. 디바이스 `new`로 할당되며 기본 디바이스 힙(8MB)에 충분하다. 두 구가 **하나의 `NoiseTexture`를 공유**하므로 Perlin은 한 번만 만들어진다.
- **호환성**: 앞 장의 장면 switch에 `sceneId == 3`만 추가했고, 텍스처/BVH/모션블러 파이프라인과 그대로 호환된다.

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| 34, 37, 38, 41, 42, 43, 45 | `Perlin.h` *(신규)* | `Perlin` 클래스. `random_int`→`curand`, 보간/스무딩/난류 |
| 35, 39, 44, 46, 47 | `Texture.h` | `NoiseTexture` 추가. 생성자가 `curandState` 보유, 기본은 대리석 |
| 36, 40 | `kernel.cu` | `sceneId == 3`(perlin_spheres) 분기 + main `sceneId` |
| 17 (`random_int`) | — | 미사용. `curand_uniform`으로 대체 |

### CUDA 적용에서 꼭 기억할 3가지

1. **격자 데이터는 디바이스에서 `curand`로**: 호스트 `random_double`/`random_int` 대신, `Perlin` 생성자가 `curandState*`를 받아 단위 벡터와 순열을 디바이스에서 만든다.
2. **`random_int(0,i)` 대체**: `int(curand_uniform * (i+1))` + 경계 클램프로 `[0,i]` 정수를 얻는다.
3. **노이즈 자체는 순수 디바이스 수학**: `floor`/`sin`/`fabs`/`Dot`만 쓰므로 보간·난류·대리석은 원서와 동일하게 포팅된다. 단계 전환은 `NoiseTexture::Value`의 반환식 한 줄.
```
