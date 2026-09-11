# One Dimensional Monte Carlo Integration (1차원 몬테카를로 적분) — CUDA 적용판

> *Ray Tracing: The Rest of Your Life* 3장을 우리 **CUDA + 레이트레이싱 프로젝트** 기준으로 정리한 문서.
> 원서의 흐름(넓이 → 기댓값 → PDF → 역함수 샘플링 → 중요도 샘플링)을 따라가되 설명은 요약·재구성했고, 코드는 전부 우리 GPU 코드다. 실제로 빌드·실행해 수치를 확인했다.
> 원서: <https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html> (v4.0.2) · 코드 커밋 `5da8fe8`

---

> 💡 **이 장에서 CUDA 때문에 달라지는 큰 그림**
> 1. 이 장은 렌더러를 건드리지 않는다. 원서의 적분 실험을 GPU 데모 세 개로 옮겼다: `--demo integrate`, `--demo halfway`, `--demo importance`.
> 2. double용 `atomicAdd`는 sm_60 이상에서만 된다(우리는 compute_52 빌드). 그래서 **스레드별 부분합 배열 → `thrust::reduce`** 로 합친다.
> 3. 원서가 `std::sort`로 푸는 "넓이 절반 지점"은 GPU 병렬 기본 연산 세 개(**정렬 → 누적합 → 이진 탐색**)로 바뀐다.
> 4. `curand_uniform_double`이 `(0,1]`을 주는 덕분에 x = 0에서 생기는 0으로 나누기·`ln 0` 문제가 저절로 사라진다.
> 5. 원서의 2차 PDF 역함수 표기가 틀린 것을 발견해 바른 식으로 구현했다(아래 ⚠️).
> 6. 빌드 이슈: **LF 줄끝 + 한글 주석 = nvcc 파싱 오류**. 새 소스는 CRLF로 저장해야 한다(맨 아래).

---

## 넓이를 재는 도구로서의 몬테카를로

2장의 π 추정은 사실 **"두 넓이의 비율"** 을 잰 것이다. 반대로 원의 넓이 공식을 모른다고 치면, 같은 실험이 원의 넓이를 재는 방법이 된다. 외접 정사각형의 넓이는 $4r^2$이므로

$$ \frac{\operatorname{area}(\text{circle})}{(2r)^2} = \frac{\pi}{4} \;\Rightarrow\; \operatorname{area}(\text{circle}) = \pi r^2 $$

즉 몬테카를로는 본질적으로 **넓이(= 적분)를 재는 도구**다. 코드에서는 출력 문구만 "Estimate of Pi" → "Estimated area of unit circle"로 바꾸면 끝이다(단위원이면 두 값이 같다).

---

## 기댓값 (Expected Value)

조금 일반화해 보자. 구간 `[a, b]`에서 입력 $x_i$를 무작위로 뽑아 함수 값 $f(x_i)$들의 평균을 낸다. 샘플이 늘수록 이 평균은 그 구간에서 함수가 갖는 평균값, 즉 **기댓값**에 수렴한다.

- **평균(average)**: 집합에서 골라낸 일부 값들의 평균. 고를 때마다 달라진다.
- **기댓값(expected value)**: 집합 전체의 평균. 하나뿐이다. 무작위 표본이 커질수록 평균은 기댓값으로 간다.

무작위 대신 간격 $\Delta x = (b-a)/N$으로 고르게 찍어도 같은 결론이 나오는데, 그 합의 극한이 바로 리만 적분이다. 그래서 구간 평균과 적분은 다음처럼 이어진다.

$$ E[f(x) \mid a \le x \le b] = \lim_{N\to\infty}\frac{1}{N}\sum_{i=0}^{N-1} f(x_i) = \frac{1}{b-a}\int_a^b f(x)\,dx $$

따라서 **적분값 = (구간 길이) × (무작위 샘플의 평균)** 이다. 적분은 무한히 얇은 조각을 더하고, 몬테카를로는 점점 늘어나는 무작위 점을 더해 같은 값에 다가간다. 닫힌 형태(해석해)가 있다면 적분이 가장 깔끔하고, 없다면 몬테카를로가 남는다.

---

## x² 적분하기 (Integrating x²)

고전적인 예 $I = \int_0^2 x^2\,dx = 8/3$ 를 몬테카를로로 풀면 $I = 2 \cdot \operatorname{average}(x^2)$ 이다. GPU에서는 스레드마다 샘플 몇 개의 합을 만들어 배열에 쓰고, 배열을 병렬 리덕션으로 합친다.

> 📄 **파일: `MonteCarloDemo.cu`** — 원서 Listing `integ-xsq-1`, `integ-sin5`, `integ-ln-sin`에 대응

```cpp
__device__ inline double EvalIntegrand(int which, double x)
{
	if (which == kIntegrandXSquared) return x * x;
	if (which == kIntegrandSin5) return pow(sin(x), 5.0);
	return log(sin(x));
}

// 구간 [a,b]에서 균일하게 뽑은 f(x)의 합을 스레드별로 구한다.
// 적분값 = (b - a) * (합 / N)   ... "구간 평균 x 구간 길이"
__global__ void UniformIntegrateKernel(
	unsigned long long seed, int which, double a, double b,
	int numThreads, int samplesPerThread, double* partialSums)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numThreads) return;

	DemoRng rng;
	curand_init(seed, id, 0, &rng);

	double sum = 0.0;
	for (int k = 0; k < samplesPerThread; k++)
	{
		// curand의 (0,1]을 그대로 써서 x 는 (a, b] 범위.
		// x = 0 에서 ln(sin 0) = -inf 가 나오는 일을 원천적으로 막는다.
		double x = a + (b - a) * curand_uniform_double(&rng);
		sum += EvalIntegrand(which, x);
	}
	partialSums[id] = sum;
}

static double IntegrateUniform(int which, double a, double b, long long n, unsigned long long seed)
{
	/* ... n을 (스레드 수) x (스레드당 샘플 수)로 나눈다 ... */
	double* dPartial;
	checkDemoErrors(cudaMalloc((void**)&dPartial, numThreads * sizeof(double)));

	UniformIntegrateKernel<<<(numThreads + 255) / 256, 256>>>(
		seed, which, a, b, numThreads, samplesPerThread, dPartial);
	checkDemoErrors(cudaGetLastError());

	double total = DeviceSum(dPartial, numThreads);   // thrust::reduce
	checkDemoErrors(cudaFree(dPartial));

	return (b - a) * total / (double(numThreads) * double(samplesPerThread));
}
```

> 🔧 **왜 `atomicAdd`가 아니라 부분합 배열인가?** 2장의 개수 세기는 정수라 `atomicAdd(unsigned long long*)`를 썼다. 실수 합에 필요한 `atomicAdd(double*)`는 **sm_60 이상**에서만 제공되는데, 이 프로젝트는 CUDA 기본 설정(compute_52)으로 빌드한다. 그래서 스레드마다 자기 칸에 부분합을 쓰고, 마지막에 `thrust::reduce`(GPU 병렬 리덕션)로 한 번에 합친다. thrust 호출은 `DemoThrust.cu`에 모아 두었다.

같은 커널에 함수만 바꿔서 세 가지를 적분했다(`--demo integrate`, N = 1,000,000). 기준값은 호스트에서 계산했다 — $\sin^5$는 부정적분 $-\cos x + \tfrac{2}{3}\cos^3 x - \tfrac{1}{5}\cos^5 x$, $\ln(\sin x)$는 초등함수 부정적분이 없어서 중점 규칙(1천만 칸)으로 구했다.

```text
[integrate] uniform Monte Carlo on [0, 2], N = 1000000
f(x)               Monte Carlo         reference           |error|
x^2             2.666781136105    2.666666666667    0.000114469438
sin^5(x)        0.903325618913    0.903931238481    0.000605619568
ln(sin(x))     -1.102914787695   -1.102222319590    0.000692468105
```

몬테카를로 쪽 코드는 세 함수 모두 **똑같이 쉽다**. 해석적 적분이 골치 아프거나 아예 없는 함수일수록 몬테카를로가 유리하다. 그래픽스에는 이런 함수가 흔하고, 더 나아가 **확률적으로만 값을 얻을 수 있는** 함수도 많다. 우리 `RayColor`가 바로 그렇다 — 한 점에서 모든 방향으로 보이는 색을 정확히 알 수는 없고, 한 방향에 대한 통계적 추정만 할 수 있다.

---

## 밀도 함수 (Density Functions)

1·2권의 `RayColor`는 단순하지만 큰 약점이 있다. **작은 광원은 노이즈가 심하다.** 균일하게 흩뜨리는 산란은 우연히 광원 쪽으로 튀어야만 빛을 모으는데, 작거나 멀리 있는 광원은 그 확률이 낮다. 그래서 바로 옆 픽셀인데 하나는 마침 광원을 맞춰 매우 밝고, 하나는 못 맞춰 매우 어둡다. 실제로는 둘 다 그 사이의 값이어야 한다(3권 2장 코넬 박스의 자글자글한 노이즈가 바로 이것).

레이를 일부러 광원 쪽으로 더 많이 보내면 노이즈는 줄지만, 그대로 두면 그림 전체가 **실제보다 밝게(편향되게)** 나온다. 더 자주 뽑은 방향은 그만큼 **가중치를 낮춰** 보정해야 하는데, 그러려면 먼저 **확률 밀도 함수(PDF)** 를 알아야 한다.

밀도 함수는 **히스토그램의 연속 버전**이다. 히스토그램의 각 칸 값을 "개수"가 아니라 "전체 대비 비율"로 바꾸고(이산 밀도), 칸 폭으로 나누어 칸을 무한히 잘게 만들면 연속인 밀도 함수가 된다. 이 함수를 어떤 구간에 대해 적분하면 "값이 그 구간에 들어갈 확률"이 나온다. 이것이 PDF다.

![그림 3: 히스토그램 예](https://raytracing.github.io/images/fig-3.03-histogram.jpg)

---

## PDF 만들기 (Constructing a PDF)

`[0, 2]`에서 직선으로 증가하는 PDF $p(r) = C \cdot r$를 생각하자. PDF는 전체 구간에서 적분하면 1(= 100%)이어야 하므로

$$ 1 = \int_0^2 C\,r\,dr = C\cdot\frac{r^2}{2}\Big|_0^2 = 2C \;\Rightarrow\; C = \frac{1}{2},\quad p(r) = \frac{r}{2} $$

![그림 4: 선형 PDF](https://raytracing.github.io/images/fig-3.04-linear-pdf.jpg)

어떤 구간 `[x0, x1]`에 들어갈 확률은 그 구간에서의 넓이 $\int_{x_0}^{x_1} p(r)\,dr$이다.

> ⚠️ **PDF 값 자체는 확률이 아니다.** 연속 분포에서 변수가 특정한 값 하나일 확률은 항상 0이다(폭이 0인 구간의 넓이). 의미 있는 것은 "구간에 들어갈 확률"뿐이다.

---

## 샘플 고르기 (Choosing our Samples)

PDF가 있으면 "어디를 더 많이 뽑을지"를 정할 수 있다. 문제는 반대 방향이다 — 균일 난수만 주는 생성기로 **원하는 PDF를 따르는 난수**를 어떻게 만들까? 균일 PDF는 쉽다(`[0,10]`이면 `10 * 난수`). 비균일 PDF가 문제다.

직관적으로 풀어 보자. $p(r) = r/2$에서 확률을 **반반으로 나누는 지점**은 어디일까? $\int_0^x r/2\,dr = x^2/4 = 0.5$ 에서 $x = \sqrt{2}$ 다. 그러면 "반 확률로 `[0, √2]`, 반 확률로 `[√2, 2]`에서 균일하게" 뽑는 거친 근사 생성기를 만들 수 있다(아래 중요도 샘플링의 **half-split**).

![그림 5: 균일 분포](https://raytracing.github.io/images/fig-3.05-uniform-dist.jpg)

![그림 6: r/2에 대한 비균일(거친) 분포](https://raytracing.github.io/images/fig-3.06-nonuniform-dist.jpg)

### 해석해가 없는 PDF의 "절반 지점" — GPU 버전

적분을 손으로 풀기 싫은 함수 $p(x) = e^{-x/2\pi}\sin^2 x$ (`[0, 2π]`)에서도 같은 지점을 **실험으로** 찾을 수 있다. 샘플을 뽑아 x 순으로 정렬한 뒤, 앞에서부터 p(x)를 더해 가다가 전체 합의 절반을 넘는 순간의 x가 답이다.

![그림 7: 해석적으로 풀고 싶지 않은 함수](https://raytracing.github.io/images/fig-3.07-exp-sin2.jpg)

원서는 샘플 배열 + `std::sort` + 순차 누적으로 푸는데, 이것을 GPU 병렬 기본 연산 세 개로 그대로 옮길 수 있다.

1. **`thrust::sort_by_key`** — x를 키로 정렬하면서 p(x)를 값으로 같이 끌고 간다.
2. **`thrust::inclusive_scan`** — 누적합(prefix sum). 순차 루프처럼 보이지만 병렬로 돈다.
3. **`thrust::lower_bound`** — 누적합은 정렬돼 있으므로 "절반 이상이 되는 첫 위치"를 이진 탐색으로 찾는다.

> 📄 **파일: `MonteCarloDemo.cu`**(샘플 생성) + **`DemoThrust.cu`**(thrust 래퍼) — 원서 Listing `est-halfway`에 대응

```cpp
// MonteCarloDemo.cu : 스레드 하나가 샘플 하나를 만든다.
__global__ void ExpSin2SampleKernel(unsigned long long seed, int n, double* xs, double* pxs)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= n) return;

	DemoRng rng;
	curand_init(seed, id, 0, &rng);

	double x = RandomDouble(&rng, 0.0, 2.0 * kPi);
	xs[id] = x;
	pxs[id] = ExpSin2(x);   // exp(-x/2pi) * sin^2(x)
}

// DemoThrust.cu : 정렬 → 누적합 → 이진 탐색
double DeviceHalfwayPoint(double* dXs, double* dValues, int n, double halfSum)
{
	thrust::device_ptr<double> xs(dXs);
	thrust::device_ptr<double> values(dValues);

	thrust::sort_by_key(xs, xs + n, values);                 // 1) x 순 정렬
	thrust::inclusive_scan(values, values + n, values);      // 2) 누적합
	int idx = int(thrust::lower_bound(values, values + n, halfSum) - values);  // 3)
	if (idx >= n)
		idx = n - 1;

	return xs[idx];   // 원소 하나 읽기 = 작은 디바이스→호스트 복사 1번
}
```

호스트에서 중점 규칙(200만 칸)으로 구한 기준값과 비교했다(`--demo halfway`).

```text
reference (midpoint rule, 2000000 cells)
  Area under curve = 1.973368799888
  Halfway          = 2.038434959652
N = 10000
  Area under curve = 1.970862073374
  Halfway          = 2.045577607524
N = 10000000
  Area under curve = 1.972701255232
  Halfway          = 2.038041469888
```

원서와 같은 1만 개에서는 절반 지점이 0.01 정도 흔들리고(시드에 따라 원서처럼 2.01대가 나오기도 한다), 천 배인 1천만 개에서는 기준값 2.0384에 소수점 셋째 자리까지 맞는다. 이 방식을 절반의 절반…으로 재귀적으로 반복하면 임의의 PDF를 따르는 난수 생성기를 대충 만들 수 있다. 다만 비용이 크고(정렬이 병목), 실무에서는 더 효율적인 방법(예: **Metropolis-Hastings**)을 쓴다.

---

## 분포 근사하기: CDF와 그 역함수

위의 "절반 지점 찾기"는 사실 **누적 확률**을 다룬 것이다. 이를 일반화한 것이 **누적 분포 함수(CDF)** 다.

$$ P(x) = \int_{-\infty}^{x} p(x')\,dx', \qquad p(x) = \frac{d}{dx}P(x) $$

PDF는 "어떤 구간이 뽑힐 확률의 밀도", CDF는 "입력보다 작은 값이 뽑힐 확률"이다. $p(r) = r/2$의 CDF는 `[0, 2]`에서 $P(r) = r^2/4$이다. 예컨대 $P(1) = 1/4$은 "뽑은 값이 1 이하일 확률이 25%"라는 뜻이다.

우리가 원하는 것은 균일 난수 d를 넣으면 이 분포를 따르는 x를 돌려주는 함수 f다. 값의 25%가 1 이하여야 하므로 f(0.25) = 1, 50%가 √2 이하여야 하므로 f(0.5) = √2 … 즉 **f(P(x)) = x**, f는 CDF의 **역함수**다. 이를 ICD(inverse cumulative distribution)라 부른다.

$$ y = \frac{r^2}{4} \;\Rightarrow\; r = \sqrt{4y}, \qquad \operatorname{ICD}(d) = \sqrt{4d} $$

검산: d = 1/4이면 1, d = 1/2이면 √2, 범위는 `[0, 2]` — 기대한 그대로다.

![그림 8: 비균일 f() 근사](https://raytracing.github.io/images/fig-3.08-approx-f.jpg)

PDF를 해석적으로 적분할 수 없으면 CDF도 없다. 그럴 때는 앞에서처럼 샘플을 정렬하거나 히스토그램을 만들어 근사한다.

---

## 중요도 샘플링 (Importance Sampling)

이제 처음의 $\int_0^2 x^2\,dx$로 돌아가 PDF를 바꿔 가며 풀어 본다. 핵심 규칙은 하나다.

$$ \int_a^b f(x)\,dx \;\approx\; \frac{1}{N}\sum_{i=1}^{N}\frac{f(r_i)}{p(r_i)}, \qquad r_i \sim p $$

더 자주 뽑히는 곳(p가 큰 곳)은 그만큼 나눠서 **덜 쳐주고**, 드물게 뽑히는 곳은 **더 쳐준다**. 가중치가 정확히 $1/p$이면 편향이 사라진다. 균일 샘플링도 사실 PDF = 1/2인 특수한 경우였고, 그래서 앞에서 곱한 "구간 길이 2"가 이 식에서는 "1/p = 2"로 자연스럽게 들어간다.

네 가지 PDF를 비교했다.

| PDF | p(x) | 샘플 생성 (d는 균일 난수) |
|---|---|---|
| uniform | 1/2 | ICD(d) = 2d |
| half-split (원서의 거친 근사) | `[0,√2]`에서 0.5/√2, `[√2,2]`에서 0.5/(2-√2) | 반 확률로 아래/위 구간에서 균일 |
| linear | x/2 | ICD(d) = √(4d) |
| quadratic | (3/8) x² | ICD(d) = (8d)^(1/3) = 2 d^(1/3) |

> ⚠️ **원서의 2차 PDF 역함수 표기 오류**: $P(x) = x^3/8$의 역함수는 $(8d)^{1/3} = 2\,d^{1/3}$인데, 원서 본문과 코드는 $8\,d^{1/3}$로 적혀 있다(이러면 x가 `[0, 8]`로 튄다). 그런데도 답이 맞게 나오는 이유는, 이 PDF에서는 $f/p = x^2 / (\tfrac{3}{8}x^2) = 8/3$이 **x와 상관없는 상수**라 어떤 x를 뽑아도 같은 값이 나오기 때문이다. 우리는 바른 식으로 구현했다.

> 📄 **파일: `MonteCarloDemo.cu`** — 원서 Listing `integ-xsq-2`, `integ-xsq-3`, `integ-xsq-5`에 대응

```cpp
// 균일 난수 → 원하는 분포를 따르는 x.
// curand_uniform_double은 (0,1]을 돌려주므로 d = 0 이 절대 나오지 않는다.
// 그래서 원서의 "if (z == 0.0) continue;"가 필요 없다.
__device__ inline double SampleX(int pdf, DemoRng* rng)
{
	double d = curand_uniform_double(rng);
	switch (pdf)
	{
	case kPdfUniform:
		return 2.0 * d;
	case kPdfHalfSplit:
	{
		double d2 = curand_uniform_double(rng);
		return (d <= 0.5) ? sqrt(2.0) * d2 : sqrt(2.0) + (2.0 - sqrt(2.0)) * d2;
	}
	case kPdfLinear:
		return sqrt(4.0 * d);
	default:
		return 2.0 * cbrt(d);          // (8d)^(1/3)
	}
}

__device__ inline double PdfValue(int pdf, double x)
{
	switch (pdf)
	{
	case kPdfUniform:   return 0.5;
	case kPdfHalfSplit: return (x < sqrt(2.0)) ? 0.5 / sqrt(2.0) : 0.5 / (2.0 - sqrt(2.0));
	case kPdfLinear:    return x / 2.0;
	default:            return (3.0 / 8.0) * x * x;
	}
}

// 샘플 하나의 추정값 g = f(x)/p(x). 합과 제곱합을 같이 모아
// 샘플 하나당 표준편차(= 노이즈의 크기)도 잰다.
__global__ void ImportanceIntegrateKernel(
	unsigned long long seed, int pdf, int numThreads, int samplesPerThread,
	double* partialSums, double* partialSumSquares)
{
	/* ... id, rng 초기화 ... */
	for (int k = 0; k < samplesPerThread; k++)
	{
		double x = SampleX(pdf, &rng);
		double g = (x * x) / PdfValue(pdf, x);
		sum += g;
		sumSq += g * g;
	}
	partialSums[id] = sum;
	partialSumSquares[id] = sumSq;
}
```

> 🔧 **half-split은 우리가 덧붙인 실험**: 원서는 이 분포를 "뽑는" 코드만 보여 주고 "균일보다 빠르고 선형보다 느리게 수렴할 것"이라고만 말한다. 여기서는 밀도(각 구간에 확률 1/2이 균일하게 퍼짐 → 0.5 / 구간 길이)까지 적어 직접 확인했다.

결과(`--demo importance`, 괄호 안은 |오차|, 마지막 열은 N = 1,048,576에서 잰 샘플 하나당 표준편차):

```text
pdf                     N=1                 N=100               N=10000             N=1048576   stddev/sample
uniform     1.27390469 (1.3927)  2.34748618 (0.3192)  2.65869372 (0.0080)  2.66530865 (0.0014)   2.3864619434
half-split  0.37566730 (2.2910)  2.80836213 (0.1417)  2.67464254 (0.0080)  2.66796445 (0.0013)   1.5026860787
linear      1.15660541 (1.5101)  2.65791027 (0.0088)  2.64772709 (0.0189)  2.66779381 (0.0011)   0.9423348548
quadratic   2.66666667 (0.0000)  2.66666667 (0.0000)  2.66666667 (0.0000)  2.66666667 (0.0000)   0.0000000000
```

- **모든 PDF가 8/3으로 수렴한다.** PDF는 정답을 바꾸지 않고 수렴 속도만 바꾼다.
- 수렴 속도는 샘플 하나당 표준편차에 그대로 드러난다: **uniform 2.39 > half-split 1.50 > linear 0.94 > quadratic 0**. 해석값과도 맞는다 — uniform은 $\sqrt{64/5 - 64/9} \approx 2.385$, linear는 $\sqrt{8 - 64/9} \approx 0.943$.
- 피적분 함수와 모양이 완벽히 같은 PDF(quadratic)는 **샘플 하나로 정답**을 낸다. 하지만 이런 PDF를 만들려면 이미 적분을 풀 수 있어야 하므로, 코드가 맞는지 확인하는 연습용이다.

비균일 PDF는 p가 큰 곳으로 샘플을 몰아준다. 노이즈가 큰 곳(= 값이 큰 곳)에서 p를 크게 잡으면 같은 샘플 수로 더 빨리 수렴한다. "중요한" 곳을 더 뽑는다고 해서 **중요도 샘플링**이라 부른다.

### 정리: 몬테카를로 레이트레이서의 기본 절차

1. 어떤 영역에서 f(x)의 적분을 구하고 싶다.
2. 그 영역에서 0이 아니고 음수도 아닌 PDF p를 고른다.
3. p를 따르는 난수 r로 f(r)/p(r)을 아주 많이 평균낸다.

어떤 p를 골라도 결국 정답에 수렴하고, p가 f와 비슷할수록 빨리 수렴한다. 다음 장부터는 이 규칙을 1차원에서 **방향(구면)** 으로 넓힌다.

---

## 결과 & 검증

- **빌드/실행 확인**: VS2022 + CUDA 12.9 Release로 컴파일·링크·실행 성공.
- **`--demo integrate`**: x², sin⁵, ln(sin) 모두 기준값과 소수점 셋째 자리 안쪽으로 일치(N = 10⁶).
- **`--demo halfway`**: 1천만 샘플에서 절반 지점 2.03804 (기준 2.03843).
- **`--demo importance`**: 네 PDF 모두 8/3으로 수렴, 샘플당 표준편차가 해석값과 일치.

> ⚠️ **실제로 부딪힌 빌드 문제: LF 줄끝 + 한글 주석**
> 새로 만든 `MonteCarloDemo.cu`에서 "expected a ;", "parsing restarts here", "identifier ... is undefined"가 엉뚱한 줄에서 났다. 주석을 고치면 에러 줄 번호만 같이 밀렸다.
> 원인: nvcc의 EDG 프런트엔드는 UTF-8 한글 주석을 cp949로 읽는다. 주석이 특정 한글(예: "배")로 끝나면 마지막 바이트가 다음 바이트를 삼키는데, **LF 파일에서는 그게 줄바꿈**이라 다음 줄(함수 끝 `}`)이 주석에 먹힌다. 기존 소스는 모두 **CRLF**라 `\r`만 먹히고 무사했던 것.
> 해결: 새 소스를 **CRLF로 저장**(BOM은 여전히 금지). 자세한 내용은 `Docs/빌드환경_및_트러블슈팅.md` ④에 정리했다.

### 변경 파일 요약

| 원서 Listing | 우리 파일 | 메모 |
|---|---|---|
| `estunitcircle` | — | 출력 문구만 다름 (2장 데모와 동일) |
| `integ-xsq-1`, `integ-sin5`, `integ-ln-sin` | `MonteCarloDemo.cu` | `UniformIntegrateKernel`, 호스트 기준값 비교 |
| `crude-approx` | `MonteCarloDemo.cu` | half-split 샘플러와 그 밀도 |
| `est-halfway` | `MonteCarloDemo.cu`, `DemoThrust.cu` | 정렬, 누적합, 이진 탐색 (thrust) |
| `integ-xsq-2/3/5` | `MonteCarloDemo.cu` | `SampleX`, `PdfValue`, `ImportanceIntegrateKernel` |
| — | `DemoThrust.cu`, `DemoThrust.h` | `DeviceSum`, `DeviceHalfwayPoint` 래퍼 |
| — | `.vcxproj` | `DemoThrust.cu` 등록 |

### CUDA 적용에서 꼭 기억할 3가지

1. **실수 합은 부분합 배열 → 병렬 리덕션**: `atomicAdd(double*)`는 sm_60 이상 전용. 스레드별 부분합을 쓰고 `thrust::reduce`로 합치면 아키텍처와 상관없이 동작하고 결과도 결정적이다.
2. **"정렬 후 누적"은 sort, scan, binary search**: 순차 알고리즘처럼 보이는 누적 합도 병렬 기본 연산 조합으로 그대로 GPU에 올라간다.
3. **curand의 `(0,1]`을 역이용**: ICD 샘플링에서 d = 0이 나오지 않아 0으로 나누기·`ln 0`을 따로 걸러낼 필요가 없다. (반대로 `[0,1)`이 필요하면 `1 - u`.)
