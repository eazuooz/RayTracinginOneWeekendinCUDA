# Generating Random Directions (무작위 방향 만들기) — CUDA 적용판

> *Ray Tracing: The Rest of Your Life* 7장을 우리 **CUDA + 레이트레이싱 프로젝트** 기준으로 정리한 문서.
> 원서의 논지를 빠짐없이 따라가되 설명은 우리 말로 다시 썼고, 코드는 전부 우리 GPU 코드다. 실제로 빌드·실행해 수치를 확인했다.
> 원서: <https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html> (v4.0.2) · 코드 커밋 `a83ec5f`

---

> 💡 **이 장에서 CUDA 때문에 달라지는 큰 그림**
> 1. 렌더러에는 `RandomCosineDirection` 하나만 추가한다(실제로 쓰는 것은 8장). 실험은 `--demo dirs`.
> 2. 역변환법은 **루프가 없다** → 4장에서 본 워프 발산이 사라진다. 같은 1,677만 방향을 만드는 데 거절법 6.97 ms, 역변환법 3.18 ms로 **2.19배** 빨랐다.
> 3. ⚠️ 마이크로벤치마크 함정: 생성 결과를 일부만 쓰면 컴파일러가 나머지 계산을 지워 버린다. 처음에 z성분만 더했다가 **27배라는 엉터리 수치**가 나왔다.
> 4. 코사인 PDF는 cos³ 적분에서 샘플당 표준편차를 **1.78 → 0.91**로 줄인다. 중요도 샘플링의 맛보기다.

---

## z축 기준으로 방향 만들기

이 장부터 세 장에 걸쳐 하는 일은 새 기능을 붙이는 것이 아니라 **이해와 도구를 단단하게 만드는 것**이다. 먼저 무작위 방향을 만드는 방법부터 정리한다.

우리에게는 이미 **거절법**(뽑고 버리기)으로 방향을 만드는 방법이 있다. 이번에는 3장에서 배운 **역변환법**으로 만들어 본다. 계산을 단순하게 하려고 **z축을 표면 법선으로 두고**, θ는 법선에서 잰 각으로 둔다. 이 장에서는 모든 것을 z축 기준으로 세우고, **다음 장(8장)에서 실제 법선 방향에 맞춰 돌린다.**

그리고 우리가 다룰 분포는 전부 **z축에 대해 회전 대칭**이다. 즉 방향의 PDF는 φ에 무관하고 θ만의 함수다.

$$ p(\omega) = f(\theta) $$

이때 φ와 θ 각각의 1차원 PDF는 다음과 같다.

$$ a(\phi) = \frac{1}{2\pi}, \qquad b(\theta) = 2\pi f(\theta)\sin\theta $$

$b(\theta)$에 $\sin\theta$가 붙는 이유는 4장에서 본 대로 구면 좌표의 넓이 조각이 $d\omega = \sin\theta\,d\theta\,d\phi$ 이기 때문이다. 극 근처(θ가 0에 가까움)는 같은 θ 폭이라도 실제 넓이가 작다.

### φ: 직관 그대로

균일 난수 $r_1$, $r_2$ 를 받아 **CDF를 풀고 뒤집는다**. φ부터 하자.

$$ r_1 = \int_0^{\phi} a(\phi')\,d\phi' = \int_0^{\phi}\frac{1}{2\pi}\,d\phi' = \frac{\phi}{2\pi} $$

뒤집으면

$$ \boxed{\phi = 2\pi\,r_1} $$

이건 직관과 정확히 일치한다 — φ의 범위가 $[0, 2\pi]$ 이므로 `[0,1]` 난수에 $2\pi$ 를 곱하면 전 범위를 덮는다. θ 쪽은 직관이 잘 안 서므로 식을 차근차근 따라가 보자.

### θ: CDF를 적분해서 뒤집기

$$ r_2 = \int_0^{\theta} b(\theta')\,d\theta' = \int_0^{\theta} 2\pi f(\theta')\sin\theta'\,d\theta' $$

$f$ 자리에 원하는 분포를 넣고 풀면 된다. 세 가지를 차례로 해 보자.

**① 구 전체 균일.** 단위 구의 넓이가 $4\pi$ 이므로 $p(\omega) = f(\theta) = 1/4\pi$ 다.

$$ r_2 = \int_0^{\theta} 2\pi\cdot\frac{1}{4\pi}\sin\theta'\,d\theta' = \int_0^{\theta}\frac{1}{2}\sin\theta'\,d\theta' = \frac{-\cos\theta}{2} - \frac{-\cos 0}{2} = \frac{1-\cos\theta}{2} $$

$$ \boxed{\cos\theta = 1 - 2r_2} $$

> 📝 **θ를 직접 구하지 않는다.** 어차피 직교 좌표로 바꿀 때 필요한 것은 $\cos\theta$ 와 $\sin\theta$ 뿐이다. 여기서 `acos`를 불러 θ를 얻고 다시 `cos`을 부르는 것은 순전한 낭비다. GPU에서는 더욱 그렇다 — 초월 함수는 SFU(special function unit)를 쓰거나 소프트웨어로 풀리므로, 한 번 아끼는 것이 그대로 처리량이 된다.

**② 반구 균일.** $p(\omega) = f(\theta) = 1/2\pi$ 로 상수만 바꾸면 된다.

$$ r_2 = \int_0^{\theta} 2\pi\cdot\frac{1}{2\pi}\sin\theta'\,d\theta' = 1 - \cos\theta \;\Longrightarrow\; \boxed{\cos\theta = 1 - r_2} $$

$r_2$ 가 `[0,1]` 이므로 $\cos\theta$ 는 1에서 0까지, 즉 θ는 0에서 $\pi/2$ 까지만 움직인다. **수평선 아래로는 아무것도 가지 않는다**는 뜻이다.

**③ 코사인 가중(Lambert).** $p(\omega) = f(\theta) = \cos\theta/\pi$ 다.

$$ r_2 = \int_0^{\theta} 2\pi\cdot\frac{\cos\theta'}{\pi}\sin\theta'\,d\theta' = \int_0^{\theta} 2\cos\theta'\sin\theta'\,d\theta' = 1 - \cos^2\theta $$

$$ \boxed{\cos\theta = \sqrt{1 - r_2}} $$

### 직교 좌표로 바꾸기

$(\theta, \phi)$ 방향의 단위 벡터는 공통 공식으로 만든다.

$$ x = \cos\phi\,\sin\theta, \qquad y = \sin\phi\,\sin\theta, \qquad z = \cos\theta $$

$\sin\theta$ 는 항등식 $\cos^2 + \sin^2 = 1$ 로 구한다. 구 전체 균일을 예로 들면

$$ x = \cos(2\pi r_1)\sqrt{1 - (1-2r_2)^2}, \quad y = \sin(2\pi r_1)\sqrt{1 - (1-2r_2)^2}, \quad z = 1 - 2r_2 $$

이고, $(1-2r_2)^2 = 1 - 4r_2 + 4r_2^2$ 이므로 루트 안이 $4r_2(1-r_2)$ 로 정리된다.

$$ x = \cos(2\pi r_1)\cdot 2\sqrt{r_2(1-r_2)}, \quad y = \sin(2\pi r_1)\cdot 2\sqrt{r_2(1-r_2)}, \quad z = 1 - 2r_2 $$

코사인 분포에서도 비슷한 정리가 가능하다. $z = \sqrt{1-r_2}$ 이므로

$$ \sin\theta = \sqrt{1 - z^2} = \sqrt{1 - (1-r_2)} = \sqrt{r_2} $$

즉 **루트 두 번이면 끝**이고, 제곱·뺄셈을 한 번 더 아낀다.

요약하면 이렇다.

| 분포 | $f(\theta)$ | $\cos\theta$ | $\sin\theta$ |
|---|---|---|---|
| 균일 구면 | $1/4\pi$ | $1 - 2r_2$ | $2\sqrt{r_2(1-r_2)}$ |
| 균일 반구 | $1/2\pi$ | $1 - r_2$ (수평선 아래로 가지 않음) | $\sqrt{1 - (1-r_2)^2}$ |
| 코사인(Lambert) | $\cos\theta/\pi$ | $\sqrt{1 - r_2}$ | $\sqrt{r_2}$ |

> 📄 **파일: `MonteCarloDemo.cu`** — 원서 Listing `rand-unit-sphere-plot`, `random-cosine-direction`에 대응

```cpp
__device__ inline Vector3 RandomUnitVectorInversion(DemoRng* rng)
{
	double r1 = RandomDouble(rng);
	double r2 = RandomDouble(rng);
	double z = 1.0 - 2.0 * r2;                      // cos(theta)
	double r = sqrt(fmax(0.0, 1.0 - z * z));        // sin(theta)
	double phi = 2.0 * kPi * r1;
	return Vector3(cos(phi) * r, sin(phi) * r, z);
}

__device__ inline Vector3 RandomHemisphereDirectionInversion(DemoRng* rng)
{
	double r1 = RandomDouble(rng);
	double r2 = RandomDouble(rng);
	double z = 1.0 - r2;                            // cos(theta), [0,1] → 수평선 위쪽만
	double r = sqrt(fmax(0.0, 1.0 - z * z));
	double phi = 2.0 * kPi * r1;
	return Vector3(cos(phi) * r, sin(phi) * r, z);
}

__device__ inline Vector3 RandomCosineDirectionDemo(DemoRng* rng)
{
	double r1 = RandomDouble(rng);
	double r2 = RandomDouble(rng);
	double z = sqrt(1.0 - r2);                      // cos(theta)
	double r = sqrt(r2);                            // sin(theta) = sqrt(1 - z^2)
	double phi = 2.0 * kPi * r1;
	return Vector3(cos(phi) * r, sin(phi) * r, z);
}
```

원서는 이렇게 만든 점 200개를 plot.ly에 올려 구면에 고르게 퍼지는지 눈으로 확인한다. 우리도 `--demo dirs`가 점 200개를 CSV로 찍어 주고, 그것을 그대로 그렸다.

![역변환법으로 만든 단위 구 위의 무작위 방향 200개](images/book3/ch07_sphere_points.png)

---

## 적분으로 검증하기

눈으로 보는 것(plot.ly에서 돌려 보며 "고르게 퍼졌네" 하는 것)보다 확실한 방법은 **답을 아는 적분**을 풀어 보는 것이다. 원서는 반구에서 $\cos^3$ 을 고른다 — 특별한 의미가 있어서가 아니라 그냥 해석해가 있는 함수라서다. 먼저 손으로 풀어 두자.

$$ \int_{\text{hemi}} \cos^3\theta\,dA = \int_0^{2\pi}\!\!\int_0^{\pi/2}\cos^3\theta\,\sin\theta\,d\theta\,d\phi = 2\pi\int_0^{\pi/2}\cos^3\theta\,\sin\theta\,d\theta = 2\pi\cdot\frac{1}{4} = \frac{\pi}{2} $$

그리고 구 전체에서 $\cos^2$ 을 적분하면 4장에서 본 대로 $4\pi/3$ 이다. 이제 이 값들을 **서로 다른 PDF로** 추정해 본다. 반구 균일이면 $p(\omega) = 1/2\pi$ 이므로 $f/p = \cos^3\theta \big/ \tfrac{1}{2\pi}$ 를 평균내고, 코사인 분포면 $p(\omega) = \cos\theta/\pi$ 로 나눈다.

`--demo dirs`로 세 경우를 각각 1,677만 샘플씩 돌렸다.

```text
[dirs] inversion-method directions, N = 16777216 per case
case                                      estimate           exact         |error|   stddev/sample
cos^2 over sphere,   p = 1/4pi         4.188496164     4.188790205     0.000294041     3.747001311
cos^3 over hemi,     p = 1/2pi         1.570411040     1.570796327     0.000385287     1.780752191
cos^3 over hemi,     p = cos/pi        1.570754462     1.570796327     0.000041865     0.906914153
```

- 셋 다 정답에 수렴한다(4장의 거절법 결과와도 일치).
- 같은 적분 $\pi/2$를 균일 반구로 풀면 샘플당 표준편차가 1.78, **코사인 분포로 풀면 0.91**이다. 분산은 표준편차의 제곱이므로 약 **3.9배** 작다 — 같은 노이즈를 얻는 데 샘플이 약 1/4만 있으면 된다는 뜻이다. 피적분 함수 $\cos^3$과 PDF $\cos$의 모양이 더 닮았기 때문이다.

---

## GPU 관점: 루프 없는 생성기의 값어치

4장에서 거절법의 실제 비용은 "워프 안 최대 시도 횟수"(레인 활용률 32%)라고 확인했다. 역변환법은 난수 두 개로 바로 계산하므로 **모든 레인이 정확히 같은 일을 같은 횟수만큼** 한다. 같은 개수의 방향을 만드는 시간을 CUDA 이벤트로 쟀다.

```text
generating 16777216 directions
  rejection method  :     6.97 ms   (  2.41 G dir/s)
  inversion method  :     3.18 ms   (  5.28 G dir/s)
  speedup           :     2.19x
```

역변환법은 `sin`, `cos`, `sqrt`를 (그것도 double로) 부르는데도 **2.19배 빠르다**. 거절법이 평균 1.91회 난수 3개를 뽑고, 워프는 그중 가장 느린 레인을 기다리기 때문이다.

> ⚠️ **마이크로벤치마크 함정 (실제로 겪음)**: 처음에는 생성한 방향의 **z성분만** 더해서 시간을 쟀다. 그랬더니 역변환법이 27배 빠르다고 나왔다. 이유는 간단하다 — 역변환법에서 z는 `1 - 2*r2` 라서, **컴파일러가 φ·sin·cos·sqrt 계산을 전부 죽은 코드로 보고 지워 버린다**. 거절법은 조건 검사에 x, y, z가 다 필요해서 지울 수 없으니 불공정한 비교가 된 것이다. 세 성분을 모두 더하도록 고치니 2.19배가 나왔다.
>
> ```cpp
> // 어느 쪽도 계산을 건너뛸 수 없게 세 성분을 모두 쓴다
> sum += d.X() + d.Y() + d.Z();
> ```
>
> GPU 마이크로벤치마크에서는 "결과를 반드시 소비하기"가 기본이고, **무엇을 소비하느냐**까지 신경 써야 한다.

---

## 렌더러에 들어간 것

이 장에서 렌더러에 추가하는 것은 코사인 분포 방향 생성기 하나다. 아직 z축 기준이라 그대로는 못 쓰고, 8장에서 정규직교 기저(ONB)로 실제 법선에 맞춰 돌린다.

> 📄 **파일: `Material.h`**

```cpp
// 거절법과 달리 루프가 없다. 난수 두 개로 바로 계산하므로 워프 안 모든 레인이
// 같은 시간에 끝난다(GPU에 유리).
//   phi = 2 pi r1,  cos(theta) = sqrt(1 - r2)  ← cos(theta)/pi 분포의 CDF를 뒤집은 것
__device__ inline Vector3 RandomCosineDirection(curandState* randState)
{
    double r1 = curand_uniform_double(randState);
    double r2 = curand_uniform_double(randState);

    double phi = 2.0 * kPi * r1;
    double z = sqrt(1.0 - r2);     // cos(theta)
    double r = sqrt(r2);           // sin(theta)

    return Vector3(cos(phi) * r, sin(phi) * r, z);
}
```

---

## 결과 & 검증

- **빌드/실행 확인**: VS2022 + CUDA 12.9 Release로 컴파일·링크·실행 성공.
- **적분 검증**: cos²(구) 4.188496 / cos³(반구) 1.570411(균일), 1.570754(코사인) — 모두 정답과 일치.
- **분산**: 코사인 PDF가 균일 PDF보다 샘플당 표준편차 약 절반(분산 약 1/4).
- **생성 비용**: 거절법 6.97 ms vs 역변환법 3.18 ms (1,677만 방향, 2.19배).

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| `rand-unit-sphere-plot` | `MonteCarloDemo.cu` | `RandomUnitVectorInversion`, 점 200개 CSV 출력 |
| `cos-cubed` | `MonteCarloDemo.cu` | 균일 반구 PDF로 cos³ 적분 |
| `random-cosine-direction`, `cos-density` | `MonteCarloDemo.cu` | 코사인 분포 생성기와 그 PDF로 cos³ 적분 |
| — | `MonteCarloDemo.cu` | 거절법 vs 역변환법 생성 비용(CUDA 이벤트) |
| `random-cosine-direction` | `Material.h` | 렌더러용 `RandomCosineDirection` (8장에서 사용) |

### CUDA 적용에서 꼭 기억할 3가지

1. **역변환법은 GPU와 궁합이 좋다**: 루프가 없어 워프 발산이 없고, `acos` 없이 cos θ·sin θ만 구하면 된다. 실측 2.19배.
2. **벤치마크는 결과를 전부 소비하자**: 일부만 쓰면 컴파일러가 나머지를 지워 엉뚱한 수치(27배)가 나온다.
3. **PDF가 피적분 함수를 닮을수록 분산이 준다**: cos³ 적분에서 코사인 PDF가 분산을 약 1/4로 줄였다. 다음 장부터 이 원리를 광원 샘플링에 쓴다.
