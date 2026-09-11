# Monte Carlo Integration on the Sphere of Directions (방향 구 위의 몬테카를로 적분) — CUDA 적용판

> *Ray Tracing: The Rest of Your Life* 4장을 우리 **CUDA + 레이트레이싱 프로젝트** 기준으로 정리한 문서.
> 원서의 흐름(1차원 → 방향 → 구면 PDF → 입체각)을 따라가되 설명은 요약·재구성했고, 코드는 전부 우리 GPU 코드다. 실제로 빌드·실행해 수치를 확인했다.
> 원서: <https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html> (v4.0.2) · 코드 커밋 `da48cf0`

---

> 💡 **이 장에서 CUDA 때문에 달라지는 큰 그림**
> 1. 렌더러는 그대로다. 방향 공간 위의 적분을 `--demo sphere`로 확인한다.
> 2. 1·2권의 무작위 방향은 **거절법**(`while` 루프)으로 만든다. GPU에서는 워프가 가장 늦은 스레드를 기다려야 해서, 평균 1.91회 시도로 끝나는 일이 실제로는 **워프 최대 5.98회** 비용이 든다(**레인 활용률 32%**). 직접 재 보았다.
> 3. 방향 벡터는 렌더러의 `Vector3`(`Vec3.h`)를 데모에서도 그대로 쓴다.

---

## 1차원에서 방향으로

지난 장의 도구(PDF, $f/p$의 평균, 역함수 샘플링)는 차원이 늘어도 그대로 통한다. 2차원, 3차원 공간에서 점을 뽑고 그 선택에 PDF로 가중치를 주면 된다. 레이트레이싱에서 가장 중요한 경우는 **무작위 방향**을 만드는 일이다.

1·2권은 무작위 벡터를 뽑아 단위 공 밖이면 버리고, 안이면 정규화해서 방향으로 썼다. 이렇게 "뽑고, 조건에 맞지 않으면 버리는" 방식을 **거절법(rejection method)** 이라 하고, 3장처럼 CDF를 뒤집어 만드는 방식을 **역변환법(inversion method)** 이라 한다. 방향의 역변환법은 7장에서 나온다.

## 방향 = 단위 구 위의 점

3차원의 모든 방향은 단위 구 위의 점 하나에 대응한다(원점에서 그 점으로 가는 벡터). 그러니 방향을 뽑는 것은 구 표면이라는 **2차원 곡면 위의 점**을 뽑는 것과 같다. 좌표는 무엇을 써도 좋다 — 극좌표 $(\theta, \phi)$로 쓰든 단위 벡터 $d$로 쓰든, PDF가 지켜야 할 규칙은 똑같다.

- 구 표면 전체에서 적분하면 **1**이어야 한다.
- PDF 값은 그 방향이 뽑힐 **상대적인** 가능성이다.

구 전체에서 균일하게 뽑는다면 밀도는 어디서나 같고, 전체 적분이 1이 되려면 구의 넓이로 나누면 된다.

$$ p(d) = \frac{1}{\text{area of unit sphere}} = \frac{1}{4\pi} $$

---

## cos²θ를 구 전체에서 적분하기

피적분 함수는 $f(\theta,\phi) = \cos^2\theta$, θ는 z축과 이루는 각이다. 단위 벡터 d의 z성분은 바로 $\cos\theta$이므로(길이 1인 벡터의 z축 사영) $f(d) = d_z^2$로 간단히 쓸 수 있다. 몬테카를로 추정은 균일 방향 $d$를 뽑아 $f(d)/p(d) = 4\pi\,d_z^2$를 평균내는 것이다.

정답은 구면 적분(면적 요소 $d\omega = \sin\theta\,d\theta\,d\phi$)으로 구할 수 있다.

$$ \int_{S^2}\cos^2\theta\,d\omega = \int_0^{2\pi}\!\!\int_0^{\pi}\cos^2\theta\,\sin\theta\,d\theta\,d\phi = 2\pi\cdot\frac{2}{3} = \frac{4\pi}{3} \approx 4.18879 $$

> 📄 **파일: `MonteCarloDemo.cu`** — 원서 Listing `main-sphereimp`에 대응

```cpp
// 거절법(rejection method)으로 균일한 무작위 방향을 만든다(1·2권 방식).
// [-1,1]^3 정육면체에서 점을 뽑아 단위 공 안에 들어오면 정규화해 단위 구 위의 점,
// 즉 방향으로 쓴다. 받아들여질 확률은 (공 부피)/(정육면체 부피) = pi/6 (약 52%)이라,
// 평균 6/pi (약 1.91)번 시도한다. tries에 이번 샘플의 시도 횟수를 돌려준다.
__device__ inline Vector3 RandomUnitVectorRejection(DemoRng* rng, int& tries)
{
	tries = 0;
	while (true)
	{
		tries++;
		Vector3 p(RandomDouble(rng, -1.0, 1.0), RandomDouble(rng, -1.0, 1.0), RandomDouble(rng, -1.0, 1.0));
		double lengthSquared = p.LengthSquared();

		// 원점에 너무 가까운 점은 정규화할 때 0으로 나누게 되므로 버린다.
		if (1e-160 < lengthSquared && lengthSquared <= 1.0)
			return p / sqrt(lengthSquared);
	}
}

__global__ void SphereIntegrateKernel(
	unsigned long long seed, int numThreads, int samplesPerThread,
	double* partialSums, double* partialTries, double* partialWarpTries)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	bool bValid = (id < numThreads);   // 워프 셔플은 모든 레인이 참여해야 하므로 return 대신 플래그

	DemoRng rng;
	curand_init(seed, id, 0, &rng);

	const double pdf = 1.0 / (4.0 * kPi);   // 균일 구면 밀도 = 1 / (구의 넓이)
	double sum = 0.0;
	double triesTotal = 0.0;
	double warpTriesTotal = 0.0;

	for (int k = 0; k < samplesPerThread; k++)
	{
		int tries = 0;
		if (bValid)
		{
			Vector3 d = RandomUnitVectorRejection(&rng, tries);
			double cosineSquared = d.Z() * d.Z();   // cos(theta) = d_z
			sum += cosineSquared / pdf;
		}
		triesTotal += tries;

		// 워프 안 최대 시도 횟수 (나비 모양 셔플 리덕션)
		int warpMax = tries;
		for (int offset = 16; offset > 0; offset >>= 1)
			warpMax = max(warpMax, __shfl_xor_sync(0xffffffffu, warpMax, offset));
		warpTriesTotal += warpMax;
	}

	if (bValid)
	{
		partialSums[id] = sum;
		partialTries[id] = triesTotal;
		partialWarpTries[id] = warpTriesTotal;
	}
}
```

결과(`--demo sphere`, 2²⁰ 스레드 × 16 = 약 1,678만 방향):

```text
[sphere] integral of cos^2(theta) over the unit sphere, N = 16777216
  Estimate          = 4.188736061391
  Exact (4/3 pi)    = 4.188790204786
  |error|           = 0.000054143395
rejection-method cost per sample
  mean tries per thread      = 1.9101   (theory 6/pi = 1.9099)
  mean tries the warp waited = 5.9776   (max over the 32 lanes)
  lane utilization           = 32.0%
```

주의할 점은 **모든 적분과 확률이 단위 구 위에서 정의된다**는 것이다. PDF도, $f/p$의 평균도 전부 "방향들의 공간"에서 계산한다.

---

## 입체각 (Solid Angle)

방향 하나는 단위 구 위의 점 하나이고, **방향들의 범위**는 그 방향들이 단위 구 위에 차지하는 **넓이**로 잰다. 이 넓이를 **입체각(solid angle)** 이라 부르고 기호로 $\omega$를 쓴다. "방향", "단위 구 위의 넓이", "입체각"은 결국 같은 것을 다르게 부르는 이름이다.

- 2차원 각도는 단위원 둘레 위의 **호의 길이**(단위: 라디안)다. 원 전체는 $2\pi$.
- 입체각은 그 2차원 확장으로, 단위 구 위의 **넓이**(단위: 스테라디안, sr)다. 구 전체는 $4\pi$ sr, 반구는 $2\pi$ sr.
- 단위 구는 3차원 물체지만 그 표면은 2차원이다. 그래서 입체각은 θ와 φ 두 변수로 표현된다.

앞으로 나오는 "방향에 대한 적분"은 전부 **입체각에 대한 적분**($d\omega$)이다. 각도가 익숙하지 않다면 "그 방향들이 지나가는 단위 구 위의 넓이"를 떠올리면 된다.

![그림 9: 구의 입체각 / 투영 넓이](https://raytracing.github.io/images/fig-3.09-solid-angle.jpg)

---

## GPU 관점: 거절법 루프의 진짜 비용

거절법은 CPU에서는 아무 문제가 없다. 평균 시도 횟수는 정육면체 대비 공의 부피 비 $\pi/6 \approx 52\%$의 역수, 즉 **약 1.91회**다. 실측값 1.9101이 이론값 1.9099와 맞는다.

GPU는 사정이 다르다. 워프(32스레드)는 같은 명령을 함께 실행하므로, `while` 루프는 **32개 레인 중 마지막 하나가 통과할 때까지** 계속 돈다. 이미 통과한 레인은 놀고 있다. 그래서 실제 비용은 워프 안 시도 횟수의 **최대값**이다. 이것을 재려고 매 샘플마다 `__shfl_xor_sync`로 워프 최대값을 구했다(레지스터끼리 값을 교환하는 나비 모양 리덕션, 5단계).

- 레인 평균 시도: **1.91회**
- 워프가 실제로 기다린 횟수(32개 중 최대): **5.98회**
- 레인 활용률: 1.91 / 5.98 ≈ **32%**

즉 방향 하나를 얻는 데 평균의 약 3배 비용을 치르고 있다. 우리 렌더러도 같은 패턴을 이미 쓰고 있다 — `Lambertian`·`Metal`·`Isotropic`의 `RandomInUnitSphere`, 카메라의 `RandomInUnitDisk`가 모두 거절법 루프다. 매 바운스마다 이 값을 치르는 셈이다. 7장의 **역변환법**(난수 두 개로 바로 방향을 계산)은 루프가 없어 모든 레인이 똑같은 시간에 끝난다. GPU에서 역변환법이 특히 반가운 이유다.

> ⚠️ **워프 셔플도 "모든 레인이 참여"**: `__shfl_xor_sync(0xffffffff, ...)`는 마스크의 모든 레인이 같이 호출해야 한다. 2장의 `__syncthreads_count`와 마찬가지로 범위 밖 스레드를 `return`으로 빼지 않고 `bValid` 플래그로 계산만 건너뛴다.

---

## 결과 & 검증

- **빌드/실행 확인**: VS2022 + CUDA 12.9 Release로 컴파일·링크·실행 성공.
- **`--demo sphere`**: $\int \cos^2\theta\,d\omega$ 추정 4.188736 (정답 4π/3 = 4.188790, 오차 5×10⁻⁵).
- **거절법 비용**: 레인 평균 1.91회(이론 6/π와 일치), 워프 최대 5.98회, 레인 활용률 32%.

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| `main-sphereimp` | `MonteCarloDemo.cu` | `SphereIntegrateKernel`, 균일 구면 PDF 1/(4π) |
| `random_unit_vector` (1권) | `MonteCarloDemo.cu` | `RandomUnitVectorRejection`, 시도 횟수 반환 |
| — | `MonteCarloDemo.cu` | `__shfl_xor_sync` 워프 최대값으로 레인 활용률 측정 |

### CUDA 적용에서 꼭 기억할 3가지

1. **방향 적분은 단위 구 위의 적분**: 균일 방향의 PDF는 1/(4π), 반구라면 1/(2π). 방향의 범위는 입체각(단위 구 위의 넓이)으로 잰다.
2. **GPU에서 거절법의 비용은 "워프 최대"**: 평균 1.91회가 아니라 약 6회를 치른다. 루프 없는 역변환법(7장)이 GPU에 잘 맞는다.
3. **워프 단위 연산은 모든 레인이 참여**: 셔플·블록 집계 앞에서는 조기 `return` 대신 플래그.
