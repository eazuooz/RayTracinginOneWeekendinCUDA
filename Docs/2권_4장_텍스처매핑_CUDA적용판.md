# Texture Mapping (텍스처 매핑) — CUDA 적용판

> *Ray Tracing: The Next Week* 4장(Texture Mapping)을 우리 **CUDA + 레이트레이싱 프로젝트**에 맞춰 다시 정리한 문서.
> 원서의 설명을 그대로 살리되, 코드는 우리 프레임워크(GPU 디바이스 코드)에 맞춰 바꿔 적었고, 이미지 텍스처에서 실제로 부딪힌 GPU/툴체인 문제와 해결까지 함께 담았다.

---

컴퓨터 그래픽스에서 **텍스처 매핑(texture mapping)** 은 장면 속 물체에 어떤 재질 효과를 입히는 과정이다. "텍스처(texture)"는 그 효과(색, 반짝임, 요철 등)이고, "매핑(mapping)"은 한 공간을 다른 공간으로 대응시키는 수학적 의미다. 이 효과는 색뿐 아니라 광택, 범프(요철, Bump Mapping), 심지어 표면의 존재 여부(잘라내기)까지 무엇이든 될 수 있다.

가장 흔한 텍스처 매핑은 **이미지를 물체 표면에 입히는 것**으로, 표면의 각 점에서의 색을 정의한다. 실제 구현은 이를 **역방향**으로 한다. 즉, 물체 위의 어떤 점이 주어지면 텍스처 맵이 그 점에 정의해 둔 색을 **조회(look up)** 한다.

텍스처 조회를 하려면 **텍스처 좌표(texture coordinate)** 가 필요하다. 관례적으로 `u`, `v` 라고 부르는 2차원 좌표다. 단색 텍스처는 모든 `(u,v)`에 대해 같은 색을 주므로 좌표를 무시해도 되지만, 다른 텍스처들은 이 좌표가 필요하므로 메서드 인터페이스에는 항상 `(u, v)` 와 점의 위치 `p`를 함께 넘긴다.

> 💡 **이 장에서 CUDA 때문에 달라지는 큰 그림**
> 1. 원서는 `shared_ptr<texture>`로 텍스처를 관리하지만, 우리는 다른 디바이스 오브젝트(Material/Hittable)와 똑같이 **커널 안에서 `new`** 로 만들고 raw 포인터(`Texture*`)로 참조한다.
> 2. **체커(절차적) 텍스처**는 전부 디바이스에서 계산되므로 포팅이 거의 그대로다.
> 3. **이미지 텍스처**가 가장 큰 차이다. stb_image는 **호스트(CPU) 전용**이라, 이미지를 호스트에서 디코딩한 뒤 픽셀 버퍼를 **`cudaMemcpy`로 디바이스 글로벌 메모리에 업로드**하고, 디바이스의 `ImageTexture`가 그 버퍼를 인덱싱한다. 게다가 stb 구현부를 `.cu`에 넣으면 nvcc 디바이스 프런트엔드가 죽어서, **별도 `.cpp`로 분리**해야 했다(본문 콜아웃 참고).

---

## 단색 텍스처 (Constant Color Texture)

먼저 텍스처의 추상 인터페이스와, 가장 단순한 **단색 텍스처**부터 만든다. 핵심 메서드는 `Value(u, v, p)` 하나로, 텍스처 좌표 `(u,v)`와 점 `p`를 받아 그 위치의 색을 반환한다.

원서는 상수 RGB 색과 텍스처를 보통 다른 클래스로 둔다고 했지만, 저자(그리고 우리)는 **"모든 색을 텍스처로 다룬다"** 는 설계를 강하게 선호한다. 어떤 색이든 텍스처로 만들 수 있으면 재질 코드가 단순하고 일반적이 되기 때문이다.

> 📄 **파일: `Texture.h` (신규)** — 원서의 `texture` / `solid_color` 에 해당. 모든 메서드는 디바이스에서 호출되므로 `__device__` 다. `shared_ptr` 대신 raw `Texture*`를 쓴다.

```cpp
#pragma once
#ifndef TEXTURE_H
#define TEXTURE_H

#include "Vec3.h"
#include "Interval.h"

// 텍스처 인터페이스: (u,v) 좌표와 점 p를 받아 그 위치의 색을 반환.
class Texture
{
public:
    __device__ virtual ~Texture() {}
    __device__ virtual Color Value(double u, double v, const Point3& p) const = 0;
};

// 단색 텍스처: (u,v)와 무관하게 항상 같은 색. "모든 색은 텍스처"의 출발점.
class SolidColor : public Texture
{
public:
    __device__ SolidColor(const Color& albedo) : mAlbedo(albedo) {}
    __device__ SolidColor(double r, double g, double b) : mAlbedo(Color(r, g, b)) {}

    __device__ Color Value(double u, double v, const Point3& p) const override
    {
        return mAlbedo;
    }

private:
    Color mAlbedo;
};
```

***Listing 22:** [Texture.h] 텍스처 인터페이스 + 단색 텍스처*

이어서 `HitRecord`에 표면 텍스처 좌표 `u, v`를 추가한다. 레이가 맞은 지점이 2D 텍스처 상에서 어디인지를 각 `Hittable`이 채워 넣고, 재질이 그 좌표로 색을 조회한다.

> 📄 **파일: `Hittable.h`**

```cpp
struct HitRecord
{
    Point3 P;
    Vector3 Normal;
    double T;
    // 표면 텍스처 좌표 (u, v) — [0,1] x [0,1]. 각 Hittable의 Hit()이 채운다.
    double U;
    double V;
    bool bFrontFace;
    Material* MaterialPtr;

    __device__ void SetFaceNormal(const Ray& ray, const Vector3& outwardNormal) { /* 기존과 동일 */ }
};
```

***Listing 23:** [Hittable.h] HitRecord에 `U, V` 추가*

---

## 공간 텍스처: 체커 텍스처 (A Checker Texture)

**공간(solid) 텍스처** 는 3D 공간상의 점 위치만으로 색을 결정한다. 마치 공간 자체를 색칠해 둔 것처럼 생각하면 된다. 물체는 그 색 공간 속을 "지나가며" 무늬가 드러난다.

체커 무늬는 입력 점의 각 좌표에 `floor()`를 적용해 정수로 만든 뒤, 세 정수의 합을 `2`로 나눈 나머지(홀짝)로 두 색을 번갈아 고른다. `floor`(버림이 아니라 항상 음의 무한대 쪽으로 내림)를 쓰는 이유는, 단순 절단(truncation)을 쓰면 `0` 주변에서 양쪽이 같은 색이 되어 무늬가 깨지기 때문이다. 마지막으로 `invScale`(= `1/scale`)로 체커 칸의 크기를 조절한다.

> 📄 **파일: `Texture.h`** — 원서의 `checker_texture`. `even`/`odd`는 또 다른 텍스처를 가리킬 수 있어(Pat Hanrahan의 셰이더 네트워크 정신) `Texture*` 두 개를 받는다.

```cpp
// 공간 체커 텍스처: 점 p의 위치만으로 3D 체커 무늬를 만든다(u,v는 무시).
class CheckerTexture : public Texture
{
public:
    __device__ CheckerTexture(double scale, Texture* even, Texture* odd)
        : mInvScale(1.0 / scale), mEven(even), mOdd(odd) {}

    __device__ Color Value(double u, double v, const Point3& p) const override
    {
        int xInteger = int(floor(mInvScale * p.X()));
        int yInteger = int(floor(mInvScale * p.Y()));
        int zInteger = int(floor(mInvScale * p.Z()));

        bool isEven = ((xInteger + yInteger + zInteger) % 2) == 0;
        return isEven ? mEven->Value(u, v, p) : mOdd->Value(u, v, p);
    }

private:
    double mInvScale;
    Texture* mEven;
    Texture* mOdd;
};
```

***Listing 24:** [Texture.h] 체커 텍스처*

---

## 절차적 텍스처를 지원하도록 Lambertian 확장

이제 **`Lambertian`이 단일 색(`Color`) 대신 `Texture*`** 를 들고 있게 한다. 단색 생성자(`Color`)를 주면 내부적으로 `SolidColor`로 감싸므로, 기존 호출부(`new Lambertian(Color(...))`)는 **그대로 두어도** 동작한다. 산란 시 히트 지점의 `(u, v, p)`로 텍스처 색을 조회해 감쇠색으로 쓴다.

> 📄 **파일: `Material.h`** — 원서 Listing 25. `make_shared<solid_color>` 대신 디바이스 `new SolidColor`.

```cpp
#include "Texture.h"
#include "Hittable.h"   // rec.U/V/P 접근을 위해 완전한 HitRecord 정의 필요

class Lambertian : public Material
{
public:
    // 단색 → SolidColor 텍스처로 감싼다.
    __device__ Lambertian(const Color& albedo)
        : mTexture(new SolidColor(albedo)) {}

    // 임의 텍스처(체커/이미지)를 직접 받는 생성자.
    __device__ Lambertian(Texture* texture)
        : mTexture(texture) {}

    __device__ bool Scatter(
        const Ray& rayIn, const HitRecord& rec,
        Color& attenuation, Ray& scattered, curandState* randState) const override
    {
        Vector3 scatterDirection = rec.Normal + RandomInUnitSphere(randState);
        if (scatterDirection.NearZero())
            scatterDirection = rec.Normal;

        scattered = Ray(rec.P, scatterDirection, rayIn.Time());
        // 히트 지점의 텍스처 좌표로 색을 조회한다(단색이면 항상 같은 값).
        attenuation = mTexture->Value(rec.U, rec.V, rec.P);
        return true;
    }

private:
    Texture* mTexture;
};
```

***Listing 25:** [Material.h] 텍스처를 사용하는 Lambertian*

> ⚠️ **CUDA 메모리 메모 (의도된 누수)**
> `Lambertian(Color)`가 내부에서 만드는 `SolidColor`, 그리고 장면에서 만드는 `CheckerTexture`/`ImageTexture`는 디바이스 `new`로 할당된다. 우리 `FreeWorld`는 기존부터 **프리미티브(`Sphere` 등)와 BVH 노드, 카메라만** 해제하고 **Material/Texture는 따로 해제하지 않는다**(단발성 렌더라 프로그램 종료 시 GPU 메모리가 회수된다). 텍스처도 같은 정책을 따른다 — 즉 약간의 의도된 누수가 있지만, 한 번 렌더하고 끝나는 이 프로그램에서는 안전하고 단순하다. (체커의 두 `SolidColor`를 두 `Lambertian`이 **공유**해도 더블 프리 걱정이 없는 이유이기도 하다.)

이제 장면의 바닥을 체커로 바꿔 보자. 원서 Listing 26과 동일하게, 단색 바닥 대신 두 색을 번갈아 쓰는 체커를 깐다.

> 📄 **파일: `kernel.cu`** *(scene 0, `CreateWorld` 내부)* — 원서 Listing 26.

```cpp
// 기존: list[0] = new Sphere(..., new Lambertian(Color(0.5, 0.5, 0.5)));
// 변경: 바닥을 체커 텍스처로
Texture* checker = new CheckerTexture(
    0.32,
    new SolidColor(Color(0.2, 0.3, 0.1)),    // even (어두운 초록)
    new SolidColor(Color(0.9, 0.9, 0.9)));   // odd  (밝은 회색)
list[i++] = new Sphere(
    Vector3(0.0, -1000.0, -1.0), 1000.0, new Lambertian(checker));
```

***Listing 26:** [kernel.cu] 체커 바닥을 깐 장면*

![그림 1: 체커 바닥 위의 구들](https://raytracing.github.io/images/img-2.02-checker-ground.png)

---

## 장면 선택 — switch로 여러 장면 다루기

원서는 책을 진행하며 장면을 계속 추가하고, `main()`의 `switch`로 어떤 장면을 렌더할지 고른다. 우리도 같은 방식을 쓰되, 우리는 **월드를 GPU 커널(`CreateWorld`)에서 만들기** 때문에 `switch`를 커널 안으로 가져온다. `main`은 `sceneId`만 정해 커널로 넘긴다.

> 📄 **파일: `kernel.cu`** *(`CreateWorld` 시그니처 + main)* — 원서 Listing 27·28.

```cpp
// CreateWorld 시그니처: sceneId + (이미지 텍스처용) 디바이스 버퍼/크기 추가
__global__ void CreateWorld(
    Hittable** list, Hittable** world, Camera** camera,
    int imageWidth, int imageHeight, curandState* randState, int* outCount,
    Hittable** bvhNodes, int* outNodeCount,
    int sceneId, const unsigned char* earthData, int earthW, int earthH)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState localRandState = *randState;
        int i = 0;

        // 장면별 카메라 파라미터(아래 분기에서 채움)
        Vector3 lookfrom(13, 2, 3), lookat(0, 0, 0);
        double vfov = 20.0, aperture = 0.0, distToFocus = 10.0;
        double shutterOpen = 0.0, shutterClose = 0.0;

        if (sceneId == 0) {
            // bouncing_spheres: 바닥을 체커로 (Listing 26) + 488개 랜덤 구 + 모션 블러
            // ... (기존 최종 장면, 바닥만 CheckerTexture로 교체) ...
            lookfrom = Vector3(13, 2, 3); vfov = 30.0; aperture = 0.1;
            shutterOpen = 0.0; shutterClose = 1.0;
        }
        else if (sceneId == 1) {
            // checkered_spheres: 체커 구 2개 (Listing 28)
            Texture* checker = new CheckerTexture(0.32,
                new SolidColor(Color(0.2, 0.3, 0.1)),
                new SolidColor(Color(0.9, 0.9, 0.9)));
            list[i++] = new Sphere(Vector3(0, -10, 0), 10, new Lambertian(checker));
            list[i++] = new Sphere(Vector3(0,  10, 0), 10, new Lambertian(checker));
            lookfrom = Vector3(13, 2, 3); vfov = 20.0; aperture = 0.0;
        }
        else {
            // earth: 이미지 텍스처 구 1개 (Listing 33)
            Texture* earthTex = new ImageTexture(earthData, earthW, earthH);
            list[i++] = new Sphere(Vector3(0, 0, 0), 2, new Lambertian(earthTex));
            lookfrom = Vector3(0, 0, 12); vfov = 20.0; aperture = 0.0;
        }

        *randState = localRandState;
        *outCount = i;

        // === BVH 빌드 (모든 장면 공통) ===
        int nodeCount = 0;
        BvhNode* root = new BvhNode(list, 0, i, bvhNodes, &nodeCount);
        bvhNodes[nodeCount++] = root;
        *outNodeCount = nodeCount;
        *world = root;

        // === 카메라 (장면별 파라미터로 공통 생성) ===
        *camera = new Camera(lookfrom, lookat, Vector3(0, 1, 0), vfov,
            double(imageWidth) / double(imageHeight),
            aperture, distToFocus, shutterOpen, shutterClose);
    }
}
```

```cpp
// main() 에서 장면 선택
int sceneId = 0;   // 0: 체커 바닥 최종 장면 / 1: 체커 구 2개 / 2: 지구
```

***Listing 27·28:** [kernel.cu] 커널 내부 장면 switch + main의 sceneId*

이제 `sceneId = 1`로 두면 위아래로 놓인 체커 구 두 개가 보인다.

![그림 2: 체커 구 두 개](https://raytracing.github.io/images/img-2.03-checker-spheres.png)

결과가 조금 이상해 보일 수 있다. 체커는 **공간(solid) 텍스처**라서, 우리가 보고 있는 건 구의 표면이 3차원 체커 공간을 "가르고 지나가는" 단면이다. 표면에 일관된 무늬를 입히고 싶다면 다음에 다룰 `(u,v)` 매핑이 필요하다.

---

## 구의 텍스처 좌표 (Texture Coordinates for Spheres)

이제 `(u,v)` 좌표를 본격적으로 쓸 차례다. 구의 텍스처 좌표는 보통 **경도/위도(구면 좌표)** 로 정한다. 점에 대한 구면 좌표 `(θ, φ)`를 구하는데, `θ`는 바닥 극(-Y)에서 위로 잰 각, `φ`는 Y축 둘레의 각(-X → +Z → +X → -Z)이다.

`(θ, φ)`를 `[0,1]` 범위의 `(u, v)`로 정규화한다:

```text
u = φ / (2π)
v = θ / π
```

원점 중심 단위 구 위의 점에 대해, 데카르트 좌표는 다음과 같다:

```text
y = −cos(θ)
x = −cos(φ)·sin(θ)
z =  sin(φ)·sin(θ)
```

이를 역으로 풀면 (`atan2` 덕분에) 다음을 얻는다. `atan2`가 `0→π` 뒤 `−π→0`으로 점프하는 걸 `0→2π` 연속으로 만들기 위해 `atan2(−z, x) + π` 형태를 쓴다:

```text
θ = arccos(−y)
φ = atan2(−z, x) + π
```

> 📄 **파일: `Sphere.h`** — 원서 Listing 29·30. 단위 구 기준 법선(`outwardNormal`)을 그대로 넘겨 `(u,v)`를 구하고, `Hit()`의 두 분기 모두에서 `HitRecord`에 채운다.

```cpp
// 원점 중심 단위 구 위의 점 p에 대한 (u,v) 텍스처 좌표.
//   <1 0 0> -> <0.50 0.50>   <-1 0 0> -> <0.00 0.50>
//   <0 1 0> -> <0.50 1.00>   <0 -1 0> -> <0.50 0.00>
//   <0 0 1> -> <0.25 0.50>   <0 0 -1> -> <0.75 0.50>
__device__ static void GetSphereUV(const Point3& p, double& u, double& v)
{
    const double pi = 3.1415926535897932385;
    double theta = acos(-p.Y());
    double phi   = atan2(-p.Z(), p.X()) + pi;
    u = phi / (2.0 * pi);
    v = theta / pi;
}

// Hit() 안, 법선을 set 한 직후:
Vector3 outwardNormal = (hitRecord.P - mCenter) / mRadius;
hitRecord.SetFaceNormal(ray, outwardNormal);
GetSphereUV(outwardNormal, hitRecord.U, hitRecord.V);   // ← 추가
hitRecord.MaterialPtr = mMaterial;
```

***Listing 29·30:** [Sphere.h] 구의 `(u,v)` 계산과 적용*

> 🔧 **`MovingSphere`에도 동일 적용**: 우리 프레임워크는 `Sphere`/`MovingSphere`가 분리돼 있으므로, 움직이는 구에도 같은 `GetSphereUV`를 추가해 두 분기에서 `(u,v)`를 채운다. 표면 좌표 자체는 중심 이동과 무관하게 법선 방향으로 결정된다.

---

## 이미지 텍스처 데이터 다루기 (Accessing Texture Image Data)

이제 이미지를 담는 텍스처를 만든다. 원서는 이미지 유틸리티 **stb_image** 로 파일을 읽어 픽셀 배열을 얻고, `rtw_image` 헬퍼가 이를 감싼다. 여기가 **CUDA에서 가장 크게 달라지는 지점**이다.

> ⚠️ **CUDA의 핵심 차이 — stb_image는 호스트 전용**
> stb_image의 디코딩 함수(`stbi_loadf` 등)는 **CPU(호스트)에서만** 돈다. GPU 커널 안에서 직접 호출할 수 없다. 그래서 파이프라인을 이렇게 나눈다:
> 1. **호스트**에서 `stbi_loadf`로 이미지를 **선형(linear) float [0,1]** 로 디코딩한다(우리 모든 셰이딩 연산은 선형 색공간에서 한다).
> 2. 호스트에서 `[0,255]` 바이트로 변환한다(원서 `rtw_image`의 `bdata`와 같은 레이아웃: 픽셀당 R,G,B 3바이트, 왼→오 / 위→아래 연속).
> 3. `cudaMalloc` + `cudaMemcpy`로 그 바이트 버퍼를 **디바이스 글로벌 메모리에 업로드**한다.
> 4. 디바이스 포인터/크기를 커널로 넘겨, 디바이스의 `ImageTexture`가 그 버퍼를 인덱싱한다.

> ⚠️ **툴체인 함정 — stb 구현부를 `.cu`에 넣지 말 것 (실제로 부딪힌 문제)**
> stb_image.h의 **구현부**(`#define STB_IMAGE_IMPLEMENTATION`, 약 5000줄의 순수 호스트 C 코드)를 `kernel.cu`(.cu) 안에 포함시키면, nvcc의 디바이스 분리 프런트엔드 **`cudafe++`가 파싱 중 죽는다**(`ACCESS_VIOLATION`). 심지어 **선언부만** include 해도 같은 TU에서 불안정했다.
> → 해결:
> 1. stb 구현부는 **별도 호스트 파일 `StbImageImpl.cpp`** 에서만 빌드한다(이 파일은 nvcc가 아니라 **cl**이 컴파일). 
> 2. `.cu`가 보는 `RtwImage.h`에서는 stb_image.h를 아예 include하지 않고, 우리가 쓰는 **두 함수만 `extern "C"`로 전방 선언**한다. 정의는 링크 시 `StbImageImpl.cpp`와 연결된다.
> 3. 추가로, stb 헤더+CJK(한글) 멀티바이트 주석이 같은 TU에 섞이면 EDG 프런트엔드의 렉서가 어긋나는 사례가 있어, **`RtwImage.h`의 주석만 ASCII**로 유지했다(나머지 헤더의 한글 주석은 그대로 둔다).

> 📄 **파일: `StbImageImpl.cpp` (신규)** — 호스트 전용 번역 단위. 여기서만 구현부를 빌드한다.

```cpp
// stb_image 구현부 전용 호스트 TU (cl이 컴파일, cudafe++ 우회)
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "external/stb_image.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
```

***Listing 31a:** [StbImageImpl.cpp] stb 구현부 분리*

> 📄 **파일: `RtwImage.h` (신규)** — 원서 `rtw_image`에 대응하는 **호스트** 헬퍼. 로드 → 바이트 변환 → 디바이스 업로드를 담당하고, 디바이스 포인터/크기를 보관한다(소멸 시 `cudaFree`). 주석은 위 콜아웃 이유로 영어로 유지.

```cpp
#pragma once
#ifndef RTW_IMAGE_H
#define RTW_IMAGE_H

// stb 구현부는 StbImageImpl.cpp에 있다. 여기서는 쓰는 두 함수만 전방 선언
// (stb_image.h 자체를 .cu 경로에 들이지 않는다 → cudafe++ 크래시 회피).
extern "C" {
    float* stbi_loadf(char const* filename, int* x, int* y,
                      int* channels_in_file, int desired_channels);
    void   stbi_image_free(void* retval_from_stbi_load);
}

#include "cuda_runtime.h"
#include <iostream>

// 호스트에서 이미지를 로드해 RGB 바이트로 변환 후 디바이스로 업로드한다.
class RtwImage
{
public:
    RtwImage() {}
    explicit RtwImage(const char* filename) { Load(filename); }
    ~RtwImage() { if (mDeviceData) cudaFree(mDeviceData); }

    bool Load(const char* filename)
    {
        int channelsInFile = 0;
        // 선형 float [0,1]로 디코딩 (모든 연산은 선형 색공간에서)
        float* fdata = stbi_loadf(filename, &mWidth, &mHeight, &channelsInFile, 3);
        if (fdata == nullptr) {
            std::cerr << "ERROR: Could not load image '" << filename << "'.\n";
            mWidth = mHeight = 0;
            return false;
        }

        int totalBytes = mWidth * mHeight * 3;

        // 호스트: 선형 float -> 바이트
        unsigned char* hostBytes = new unsigned char[totalBytes];
        for (int k = 0; k < totalBytes; k++)
            hostBytes[k] = FloatToByte(fdata[k]);
        stbi_image_free(fdata);

        // 디바이스 글로벌 메모리로 업로드
        cudaMalloc((void**)&mDeviceData, totalBytes);
        cudaMemcpy(mDeviceData, hostBytes, totalBytes, cudaMemcpyHostToDevice);
        delete[] hostBytes;
        return true;
    }

    const unsigned char* DeviceData() const { return mDeviceData; }
    int Width()  const { return mWidth; }
    int Height() const { return mHeight; }

private:
    unsigned char* mDeviceData = nullptr;  // 디바이스 버퍼
    int mWidth = 0, mHeight = 0;

    static unsigned char FloatToByte(float value)
    {
        if (value <= 0.0f) return 0;
        if (1.0f <= value) return 255;
        return static_cast<unsigned char>(256.0f * value);
    }
};
#endif
```

***Listing 31b:** [RtwImage.h] 호스트 로더 + 디바이스 업로드 (원서 `rtw_image` 대응)*

이제 디바이스에서 그 버퍼를 인덱싱하는 `image_texture`다. 원서 Listing 32와 같지만, 데이터 출처가 파일이 아니라 **디바이스 글로벌 메모리 버퍼**다.

> 📄 **파일: `Texture.h`** — 원서 Listing 32.

```cpp
class ImageTexture : public Texture
{
public:
    // devData : 디바이스 글로벌 메모리의 RGB 바이트 버퍼(RtwImage가 업로드).
    __device__ ImageTexture(const unsigned char* devData, int width, int height)
        : mData(devData), mWidth(width), mHeight(height) {}

    __device__ Color Value(double u, double v, const Point3& p) const override
    {
        // 데이터가 없으면 디버깅용 청록색
        if (mHeight <= 0 || mData == nullptr) return Color(0.0, 1.0, 1.0);

        // (u,v)를 [0,1]로 클램프, v는 이미지 좌표계(위가 0)로 뒤집는다
        u = Interval(0.0, 1.0).Clamp(u);
        v = 1.0 - Interval(0.0, 1.0).Clamp(v);

        int i = int(u * mWidth);
        int j = int(v * mHeight);
        if (i >= mWidth)  i = mWidth - 1;
        if (j >= mHeight) j = mHeight - 1;

        const unsigned char* pixel = mData + (j * mWidth + i) * 3;
        double s = 1.0 / 255.0;
        return Color(s * pixel[0], s * pixel[1], s * pixel[2]);
    }

private:
    const unsigned char* mData;   // 디바이스 글로벌 메모리 RGB 바이트
    int mWidth, mHeight;
};
```

***Listing 32:** [Texture.h] 이미지 텍스처 클래스*

---

## 이미지 텍스처 렌더링 (Rendering The Image Texture)

저자처럼 웹에서 적당한 지구 맵(`earthmap.jpg`)을 가져온다. 어떤 표준 투영이든 상관없다.

![그림 3: earthmap.jpg](https://raytracing.github.io/images/earthmap.jpg)

원서는 `earth()` 장면에서 이미지를 읽어 diffuse 재질에 입힌다. 우리는 ① **호스트(`main`)에서 이미지를 디바이스로 업로드**하고, ② 디바이스 버퍼 포인터/크기를 커널로 넘겨 `ImageTexture`를 만든다.

> 📄 **파일: `kernel.cu`** *(main — 이미지 업로드 + 커널 호출)* — 원서 Listing 33.

```cpp
// === 이미지 텍스처 업로드 (scene 2에서만) ===
// RtwImage 소멸자가 디바이스 버퍼를 해제하므로, 렌더가 끝날 때까지 살아 있도록
// main 스코프에 둔다. 파일이 없으면 DeviceData()==nullptr → 커널이 청록색을 표시.
RtwImage earthImage;
const unsigned char* earthData = nullptr;
int earthW = 0, earthH = 0;
if (sceneId == 2) {
    earthImage.Load("earthmap.jpg");
    earthData = earthImage.DeviceData();
    earthW = earthImage.Width();
    earthH = earthImage.Height();
}

CreateWorld<<<1, 1>>>(list, world, camera, imageWidth, imageHeight,
    randState2, d_numHittables, bvhNodes, d_numNodes,
    sceneId, earthData, earthW, earthH);
```

***Listing 33:** [kernel.cu] 지구 텍스처 맵으로 렌더링*

`sceneId = 2`로 두고 빌드하면 지구가 입혀진 구가 나온다. 만약 큰 **청록색** 구가 나오면 `earthmap.jpg`를 찾지 못한 것이다 — 실행 파일과 같은 디렉터리(혹은 작업 디렉터리)에 이미지를 두어야 한다.

![그림 4: 지구가 매핑된 구](https://raytracing.github.io/images/img-2.05-earth-sphere.png)

여기서 **"모든 색은 텍스처"** 설계의 힘이 드러난다. `Lambertian`은 자기가 단색인지, 체커인지, 이미지인지 전혀 몰라도 된다. 그냥 `Texture*`에 색을 물어볼 뿐이다.

---

## 빌드 설정 (Visual Studio)

새 파일들을 프로젝트에 추가한다.

> 📄 **파일: `RayTracinginOneWeekend.vcxproj`**

```xml
<!-- 컴파일 대상: stb 구현부는 cl이 빌드 -->
<ItemGroup>
  <ClCompile Include="StbImageImpl.cpp" />
</ItemGroup>
<!-- 헤더 -->
<ItemGroup>
  <ClInclude Include="Texture.h" />
  <ClInclude Include="RtwImage.h" />
  <ClInclude Include="external\stb_image.h" />
  <!-- ... 기존 헤더들 ... -->
</ItemGroup>
```

또한 `external/stb_image.h`(공개 도메인, nothings/stb)를 프로젝트의 `external/` 폴더에 둔다. `earthmap.jpg`는 실행 시 작업 디렉터리에서 찾으므로 프로젝트/출력 디렉터리에 복사해 둔다(scene 2에서만 필요).

---

## 결과 & 검증

- **체커 바닥/체커 구(scene 0·1)**: 절차적 텍스처라 외부 의존성 없이 디바이스에서 전부 계산된다. 기존 모션 블러 + BVH 파이프라인과 그대로 호환된다.
- **이미지 텍스처(scene 2)**: 호스트 디코딩 → 디바이스 업로드 → 디바이스 샘플링. `earthmap.jpg`가 없으면 청록색으로 **안전하게 degrade** 한다.
- **호환성**: `Lambertian(Color)` 기존 호출부는 내부에서 `SolidColor`로 감싸지므로 **수정 없이** 동작한다(Metal/Dielectric은 변화 없음).

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| 22, 24, 32 | `Texture.h` *(신규)* | `Texture`/`SolidColor`/`CheckerTexture`/`ImageTexture`, 전부 `__device__` + raw 포인터 |
| 23 | `Hittable.h` | `HitRecord`에 `U, V` 추가 |
| 25 | `Material.h` | `Lambertian`이 `Texture*` 보유, 단색은 `SolidColor`로 감쌈 |
| 26, 27, 28, 33 | `kernel.cu` | 체커 바닥 + 커널 내 장면 switch + 이미지 업로드 플러밍 |
| 29, 30 | `Sphere.h` / `MovingSphere.h` | `GetSphereUV` + `Hit()`에서 `(u,v)` 채움 |
| 31 (`rtw_image`) | `RtwImage.h` *(신규)* + `StbImageImpl.cpp` *(신규)* | 호스트 로드 → 디바이스 업로드, stb 구현부 분리 |
| — | `external/stb_image.h` | 공개 도메인 이미지 로더(호스트) |

### CUDA 적용에서 꼭 기억할 3가지

1. **이미지는 호스트에서 디코딩 → 디바이스로 업로드**: stb_image는 GPU에서 못 돈다. `cudaMemcpy`로 올린 바이트 버퍼를 디바이스 `ImageTexture`가 인덱싱한다.
2. **stb 구현부는 별도 `.cpp`로**: `STB_IMAGE_IMPLEMENTATION`을 `.cu`에 넣으면 `cudafe++`가 죽는다. 구현부는 `StbImageImpl.cpp`(cl 컴파일), `.cu`에는 함수 전방 선언만.
3. **텍스처도 디바이스 `new`**: `shared_ptr` 대신 raw `Texture*`. 해제는 기존 Material과 같은 정책(단발성 렌더라 의도된 누수). 단색은 `SolidColor`로 감싸 기존 호출부 호환.
```
