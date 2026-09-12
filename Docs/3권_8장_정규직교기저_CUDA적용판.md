# Orthonormal Bases (정규직교 기저) — CUDA 적용판

> *Ray Tracing: The Rest of Your Life* 8장을 우리 **CUDA + 레이트레이싱 프로젝트** 기준으로 정리한 문서.
> 원서의 흐름(상대 좌표 → ONB 만들기 → ONB 클래스 → 재질에 적용)을 따라가되 설명은 요약·재구성했고, 코드는 전부 우리 GPU 코드다. 실제로 빌드·렌더해서 결과가 8장 이전과 같은지 확인했다.
> 원서: <https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html> (v4.0.2) · 코드 커밋 `6f46903`

---

> 💡 **이 장에서 CUDA 때문에 달라지는 큰 그림**
> 1. `Onb.h`를 새로 만든다. 힙을 쓰지 않는 작은 값 타입이라 **산란 함수 안에서 스택에 그대로** 만들어 쓴다(디바이스에서 `new` 금지).
> 2. `Material::Scatter`에 **`double& pdf` 출력 인자**가 붙는다. 재질이 "이 방향을 어떤 밀도로 뽑았는지"를 직접 알려주고, `RayColor`는 더 이상 그 값을 스스로 가정하지 않는다.
> 3. Metal/Dielectric은 `pdf = 0`(델타 분포), Isotropic은 `1/4π`를 돌려준다. 우리는 6장에서 만든 "pdf가 0이면 감쇠만" 규칙 덕분에 1·2권 장면이 계속 정상이다.
> 4. 결과 이미지는 6장과 **사실상 동일**(선형 평균 0.0783 vs 0.0784) — 리팩터링이 동작을 바꾸지 않았다는 확인이다.
> 5. ⚠️ 속도는 오히려 느려졌다(32.3초 → 40.9초). 원인을 측정으로 좁혀 봤고, 결론은 "8장의 이득은 속도가 아니라 **pdf를 정확히 아는 것**"이다(아래 측정 표).

---

## 왜 기저가 필요한가

7장에서 만든 방향들은 **z축이 법선**이라고 가정한 좌표계의 값이다. 실제 표면의 법선은 제각각이니, 그 좌표계를 법선에 맞춰 돌려 줘야 한다.

**정규직교 기저(ONB)** 는 서로 수직인 단위 벡터 세 개다. xyz 축이 그 예다. 위치를 (3, −2, 7)이라고 말할 때 우리는 사실 "원점에서 x축으로 3, y축으로 −2, z축으로 7만큼"이라고 말하는 것이다. 기저가 $\mathbf{u}, \mathbf{v}, \mathbf{w}$ 로 바뀌면 같은 방식으로

$$ \text{방향} = x\,\mathbf{u} + y\,\mathbf{v} + z\,\mathbf{w} $$

로 옮기면 된다. 방향에는 원점이 필요 없으므로(방향은 상대적인 양이다) 기저 세 개만 있으면 충분하다.

법선 $\mathbf{n}$ 하나에서 나머지 두 축을 만드는 흔한 방법은, $\mathbf{n}$과 나란하지 않은 아무 축 $\mathbf{a}$를 골라 외적을 쓰는 것이다. 외적 $\mathbf{n} \times \mathbf{a}$ 는 두 벡터 모두에 수직이기 때문이다. $\mathbf{a}$가 하필 $\mathbf{n}$과 나란하면 0벡터가 되므로, $|\mathbf{n}_x| > 0.9$이면 y축을, 아니면 x축을 고른다.

$$ \mathbf{s} = \operatorname{unit}(\mathbf{n} \times \mathbf{a}), \qquad \mathbf{t} = \mathbf{n} \times \mathbf{s} $$

$\mathbf{n}$과 $\mathbf{s}$가 단위이고 서로 수직이므로 $\mathbf{t}$는 정규화할 필요가 없다.

> 📄 **파일: `Onb.h`** *(신규)* — 원서 Listing `class-onb`에 대응

```cpp
class Onb
{
public:
    __device__ Onb() {}

    __device__ Onb(const Vector3& n)
    {
        mAxis[2] = UnitVector(n);
        Vector3 a = (fabs(mAxis[2].X()) > 0.9) ? Vector3(0.0, 1.0, 0.0) : Vector3(1.0, 0.0, 0.0);
        mAxis[1] = UnitVector(Cross(mAxis[2], a));
        mAxis[0] = Cross(mAxis[2], mAxis[1]);
    }

    __device__ const Vector3& U() const { return mAxis[0]; }
    __device__ const Vector3& V() const { return mAxis[1]; }
    __device__ const Vector3& W() const { return mAxis[2]; }

    // 기저 좌표 → 월드 좌표
    __device__ Vector3 Transform(const Vector3& v) const
    {
        return v.X() * mAxis[0] + v.Y() * mAxis[1] + v.Z() * mAxis[2];
    }

private:
    Vector3 mAxis[3];
};
```

> 🔧 **CUDA 메모**: 디바이스 코드에서 `new`는 피하는 게 좋다(느리고 힙이 작다). `Onb`는 벡터 세 개짜리 값 타입이라 `Scatter` 안에서 지역 변수로 만들면 레지스터/로컬 메모리에 올라갔다 사라진다. 원서처럼 클래스로 묶어도 GPU에서 추가 비용이 없다.

---

## 재질에 적용하기

### `Scatter`가 pdf를 알려준다

6장까지 `RayColor`는 "재질이 pScatter와 같은 분포로 뽑았겠지"라고 가정하고 `pdfValue = scatteringPdf`로 두었다. 이제 재질이 **자기가 실제로 쓴 밀도**를 직접 돌려준다. 원서가 `scatter()`에 `double& pdf`를 추가한 것과 같다.

> 📄 **파일: `Material.h`** — 원서 Listing `scatter-onb`에 대응

```cpp
// 재질이 "어떤 밀도로 이 방향을 뽑았는지"(pdf)를 함께 알려준다. 방향이 사실상 하나로
// 정해지는 재질(Metal/Dielectric)은 pdf = 0을 돌려주고, 그 경우 RayColor가 감쇠만 곱한다.
__device__ virtual bool Scatter(
    const Ray& rayIn, const HitRecord& rec,
    Color& attenuation, Ray& scattered, double& pdf, curandState* randState) const = 0;
```

### Lambertian: ONB + 코사인 역변환 샘플링

7장의 `RandomCosineDirection`은 z축 기준 방향이다. 여기에 ONB를 씌우면 곧바로 이 표면의 cos/π 분포 산란 방향이 된다. 밀도도 그 자리에서 정확히 안다.

```cpp
__device__ bool Scatter(
    const Ray& rayIn, const HitRecord& rec,
    Color& attenuation, Ray& scattered, double& pdf, curandState* randState) const override
{
    Onb uvw(rec.Normal);
    Vector3 scatterDirection = UnitVector(uvw.Transform(RandomCosineDirection(randState)));

    scattered = Ray(rec.P, scatterDirection, rayIn.Time());
    attenuation = mTexture->Value(rec.U, rec.V, rec.P);
    pdf = Dot(uvw.W(), scatterDirection) / kPi;   // cos(theta)/pi
    return true;
}
```

6장의 "법선 + 구 위의 점"과 **같은 분포**지만 두 가지가 달라진다.

- 거절법 루프가 사라져 워프 발산이 없다(7장).
- **뽑은 방향의 밀도를 그 자리에서 정확히 안다.** 10장에서 여러 PDF를 섞으려면 이 값이 반드시 필요하다.

### 나머지 재질

| 재질 | pdf | 메모 |
|---|---|---|
| `Metal`, `Dielectric` | `0.0` | 방향이 (거의) 하나인 델타 분포 → 나눌 수 없다. `RayColor`가 감쇠만 곱한다 |
| `Isotropic` | `1/4π` | 구 전체 균일. `ScatteringPdf`도 `1/4π` |
| `DiffuseLight` | — | 산란하지 않음(`false` 반환) |

> 📄 **파일: `kernel.cu`** *(`RayColor`)* — 원서 Listing `scatter-ray-color`에 대응

```cpp
// 재질이 "이 방향을 뽑은 밀도"를 pdfValue로 알려준다(델타 분포 재질은 0).
double pdfValue = 0.0;
if (!rec.MaterialPtr->Scatter(currentRay, rec, attenuation, scattered, pdfValue, randState))
	return accumulated;

double scatteringPdf = rec.MaterialPtr->ScatteringPdf(currentRay, rec, scattered);
if (scatteringPdf > 0.0 && pdfValue > 0.0)
{
	throughput = throughput * attenuation * (scatteringPdf / pdfValue);
}
else
{
	throughput = throughput * attenuation;   // 델타 분포 재질
}
```

---

## 결과: 그림은 그대로

![이미지 6: ONB 산란을 쓴 코넬 박스 (원서)](https://raytracing.github.io/images/img-3.06-cornell-ortho.jpg)

![우리 렌더: ONB + 코사인 역변환 샘플링, 961 spp](images/book3/ch08_cornell_onb.png)

6장 기준 이미지와 비교하면 밝기도 노이즈도 사실상 같다. 리팩터링이 결과를 바꾸지 않았다는 뜻이다.

| 이미지 | 선형 평균 밝기 | 뒷벽 표준편차 |
|---|---|---|
| 6장 (법선 + 구 위의 점, 거절법) | 0.0784 | 17.26 |
| 8장 (ONB + 코사인 역변환) | 0.0783 | 17.19 |

---

## ⚠️ 속도는 왜 느려졌나 (측정 기록)

7장 데모에서 역변환법이 거절법보다 2.19배 빨랐으니 렌더도 빨라질 것 같지만, 실제로는 **32.3초 → 40.9초로 느려졌다**. 원인을 좁히려고 두 가지를 측정했다(같은 장면, 961 spp).

| 버전 | 시간 | 이미지 |
|---|---|---|
| 6장: 법선 + 구 위의 점(거절법, float 난수, 삼각함수 없음) | 32.3초 | 기준 |
| 8장: ONB + 코사인 역변환 (double 난수·삼각함수) | 40.9초 | 동일 |
| 실험 A: 난수만 float(`curand_uniform`)로 | 43.6초 | 동일 |
| 실험 B: 방향 계산 전체를 float + `__sincosf`(하드웨어 SFU)로 | 38.4초 | 동일 |

- **난수 정밀도는 원인이 아니다**(실험 A는 오히려 더 느렸다 — 측정 잡음 범위).
- **삼각함수 정밀도도 주된 원인이 아니다**(실험 B가 6%쯤 줄였을 뿐).
- 남는 차이는 **바운스마다 늘어난 벡터 연산**이다. 6장 경로는 float 난수 3개를 평균 1.9번 뽑는 것이 전부였고 삼각함수가 아예 없었다. 8장 경로는 ONB를 만들고(정규화 2회, 외적 2회) `sin`·`cos`·`sqrt`를 부른 뒤 다시 정규화한다. 7장 데모의 2.19배는 **생성만 따로**, 그것도 양쪽 모두 double로 재서 나온 값이라 렌더러에는 그대로 적용되지 않는다.

그래서 실험 B는 채택하지 않고 **원서와 같은 double 버전을 커밋**했다(프로젝트 전체가 double이고, 몇 %를 위해 구조를 흐트러뜨릴 이유가 없다). 8장의 진짜 소득은 속도가 아니라 **밀도를 정확히 아는 것**이다 — 10장에서 광원 PDF와 섞으려면 반드시 필요하다.

> 📝 나중에 성능이 급하면 볼 만한 선택지: 렌더러 전체를 float로 바꾸기(FP64가 1/64 속도인 소비자용 GPU에서 가장 큰 한 방), 또는 `Onb`를 히트 레코드에 캐시해 재사용하기.

---

## 결과 & 검증

- **빌드/실행 확인**: VS2022 + CUDA 12.9 Release로 컴파일·링크·실행 성공.
- **동작 보존**: 8장 렌더가 6장과 선형 평균 0.0783 vs 0.0784, 뒷벽 노이즈 17.19 vs 17.26으로 사실상 동일.
- **1·2권 장면**: `Scatter` 시그니처가 바뀌었지만 Metal/Dielectric은 `pdf = 0`이라 기존과 같은 경로로 렌더된다.

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| `class-onb` | `Onb.h` *(신규)* | 디바이스 ONB 값 타입, `Transform` |
| `scatter-onb` | `Material.h` | `Scatter`에 `double& pdf`, Lambertian은 ONB + 코사인 역변환 |
| `class-isotropic-impsample` | `Material.h` | Isotropic `pdf = 1/4π`, `ScatteringPdf = 1/4π` |
| `scatter-onb` | `Metal.h`, `Dielectric.h` | 델타 분포 → `pdf = 0` |
| `scatter-ray-color` | `kernel.cu` (`RayColor`) | 재질이 알려준 pdf로 나눈다 |
| — | `.vcxproj` | `Onb.h` 등록 |

### CUDA 적용에서 꼭 기억할 3가지

1. **작은 값 타입은 스택에**: `Onb`처럼 벡터 몇 개짜리 객체는 디바이스에서 `new` 없이 지역 변수로 만들면 된다.
2. **"누가 밀도를 아는가"를 분명히**: 재질이 뽑은 분포의 밀도를 재질이 돌려주는 구조라야 10장의 혼합 PDF로 갈 수 있다.
3. **데모의 속도 이득이 렌더러의 이득은 아니다**: 생성만 떼어 재면 역변환법이 2배 빠르지만, 렌더러에서는 바운스당 추가 연산 때문에 되레 느려질 수 있다. 반드시 실제 장면으로 재 보자.
