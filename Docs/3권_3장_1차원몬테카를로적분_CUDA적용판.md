# One Dimensional Monte Carlo Integration (1차원 몬테카를로 적분) — CUDA 적용판

> *Ray Tracing: The Rest of Your Life* 3장을 우리 **CUDA + 레이트레이싱 프로젝트** 기준으로 정리한 문서.
> 원서의 논지를 빠짐없이 따라가되 설명은 우리 말로 다시 썼고, 코드는 전부 우리 GPU 코드다. 실제로 빌드·실행해 수치를 확인했다.
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

2장에서 π를 추정한 방식을 다시 보자. 정사각형 안에 점을 뿌리고 원 안에 떨어진 비율을 셌다. 그 비율은 **두 넓이의 비**로 수렴한다.

$$ \frac{\operatorname{area}(circle)}{\operatorname{area}(square)} = \frac{\pi}{4} $$

여기서 관점을 하나 뒤집어 보자. 우리는 "π를 몰라서" 이 실험을 했지만, 반대로 **원의 넓이 공식을 모른다고 가정**하면 같은 실험이 원의 넓이를 재 주는 도구가 된다. 외접 정사각형의 넓이는 한 변이 $2r$이므로 $4r^2$이고, 위 비율을 알고 있으니

$$ \frac{\operatorname{area}(circle)}{(2r)^2} = \frac{\pi}{4} \;\Rightarrow\; \operatorname{area}(circle) = \frac{\pi}{4}\cdot 4r^2 = \pi r^2 $$

가 나온다. 반지름 1을 넣으면 원의 넓이가 그대로 π다. 그래서 2장의 프로그램은 **출력 문구만 바꾸면** 그대로 "단위원의 넓이를 추정하는 프로그램"이 된다 — 계산은 한 줄도 바뀌지 않는다.

이 관점이 이 장 전체의 출발점이다. **몬테카를로는 넓이를 재는 도구이고, 넓이를 재는 일이 곧 적분이다.**

---

## 기댓값 (Expected Value)

지금까지는 "점이 원 안에 들어갔나"라는 예/아니오만 셌다. 이제 일반적인 함수로 넓혀 보자. 다음이 있다고 하자.

1. 값들의 목록 $X = (x_0, x_1, \ldots, x_{N-1})$
2. 그 값을 받는 연속 함수 $f(x)$
3. 목록의 각 원소에 $f$를 적용해 얻은 출력 목록 $Y = (f(x_0), \ldots, f(x_{N-1}))$

이때 $Y$의 산술 평균은 다음과 같이 쓸 수 있고, 이것을 $Y$의 **기댓값** $E[Y]$라고 부른다.

$$ E[Y] = \frac{1}{N}\sum_{i=0}^{N-1} y_i = \frac{1}{N}\sum_{i=0}^{N-1} f(x_i) $$

여기서 **평균(average)** 과 **기댓값(expected value)** 의 차이를 한 번 짚고 가는 게 좋다. 미묘하지만 이 장 내내 쓰이는 구분이다.

- 어떤 집합에서 원소를 골라 만든 **부분집합은 여러 개** 있을 수 있고, 각 부분집합마다 **평균**이 하나씩 나온다. 같은 원소가 여러 번 뽑힐 수도, 한 번도 안 뽑힐 수도 있다.
- 반면 집합의 **기댓값은 하나뿐**이다. 집합 전체 원소의 합을 전체 개수로 나눈 값, 즉 "모든 원소의 평균"이다.
- 그리고 **표본 수가 늘어날수록 부분집합의 평균은 집합의 기댓값으로 수렴한다.** 몬테카를로가 작동하는 이유가 정확히 이것이다.

이제 $x_i$를 연속 구간 $[a, b]$에서 무작위로 뽑는다고 하자. 그러면 위의 평균은 그 구간에서 함수 $f$가 갖는 평균값으로 다가간다.

$$ E[f(x) \mid a \le x \le b] = \lim_{N\to\infty}\frac{1}{N}\sum_{i=0}^{N-1} f(x_i) $$

**무작위로 뽑는 것만이 유일한 방법은 아니다.** 구간을 $N$등분해 고르게 찍어도 된다.

$$ x_i = a + i\,\Delta x, \qquad \Delta x = \frac{b-a}{N} $$

이 경우 위 식을 정리하면 $\Delta x$를 끼워 넣을 수 있고,

$$ E[f(x)] \approx \frac{1}{N}\sum f(x_i) = \frac{\Delta x}{b-a}\sum f(x_i) = \frac{1}{b-a}\sum f(x_i)\,\Delta x $$

$N \to \infty$의 극한을 취하면 오른쪽 합은 우리가 아는 **리만 적분** 그 자체가 된다.

$$ E[f(x) \mid a \le x \le b] = \frac{1}{b-a}\int_a^b f(x)\,dx $$

그리고 적분은 곡선 아래 넓이이므로,

$$ \operatorname{area}(f, a, b) = \int_a^b f(x)\,dx, \qquad E[f(x)] = \frac{1}{b-a}\cdot\operatorname{area}(f, a, b) $$

**구간의 평균과 곡선 아래 넓이는 본질적으로 같은 것**이라는 결론이 나온다. 적분은 무한히 얇은 조각을 더해서 그 평균을 구하고, 몬테카를로는 무작위 점을 점점 더 많이 더해서 같은 평균에 다가간다. 둘은 같은 답을 향하는 서로 다른 길이다. 닫힌 형태(해석해)가 있으면 적분이 가장 깔끔하고, 없으면 몬테카를로가 남는다.

여기서 실전에서 계속 쓰게 될 형태를 얻는다.

$$ \boxed{\int_a^b f(x)\,dx = (b-a)\cdot E[f(x)] \approx (b-a)\cdot\frac{1}{N}\sum_{i=0}^{N-1} f(x_i)} $$

---

## x² 적분하기 (Integrating x²)

고전적인 예로 확인해 보자.

$$ I = \int_0^2 x^2\,dx $$

적분으로 풀면 $\frac{1}{3}x^3\big|_0^2 = \frac{8}{3}$ 이다. 몬테카를로로 풀려면 앞의 상자 친 식을 그대로 쓰면 된다. 구간 길이가 2이므로

$$ I = 2\cdot\operatorname{average}(x^2, 0, 2) $$

즉 $[0, 2]$에서 무작위로 뽑은 $x$들의 $x^2$ 평균에 2를 곱하면 된다.

여기서 자연스럽게 드는 반론이 있다. "$x^2$ 정도는 적분이 훨씬 쉬운데 왜 몬테카를로를 쓰나?" 맞는 말이다. 하지만 함수를 조금만 바꿔 보면 상황이 달라진다.

- $f(x) = \sin^5(x)$ — 적분이 가능하지만 부분적분을 거쳐야 해서 손이 많이 간다. 몬테카를로 쪽은 **함수 한 줄만 바꾸면 끝**이다.
- $f(x) = \ln(\sin(x))$ — 초등함수로 된 부정적분이 **아예 없다**. 그래도 몬테카를로는 똑같이 동작한다.

그래픽스에는 이런 함수가 흔하다. 더 나아가 **식으로 적을 수조차 없고 확률적으로만 값을 얻을 수 있는** 함수도 많다. 우리 `RayColor`가 정확히 그렇다 — 어떤 점에서 모든 방향으로 보이는 색이 얼마인지 우리는 알 수 없고, 한 방향을 무작위로 골라 그 방향의 색을 통계적으로 추정할 수 있을 뿐이다. **레이트레이싱 자체가 이미 몬테카를로 적분이었다**는 뜻이다.

### GPU로 옮기기

원서는 CPU에서 `for` 루프를 N번 돈다. 우리는 스레드 하나가 샘플 몇 개씩 맡고, 스레드별 부분합을 배열에 쓴 뒤, 그 배열을 병렬 리덕션으로 합친다.

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
```

> 🔧 **왜 `atomicAdd`가 아니라 부분합 배열인가?**
> 2장의 π 추정은 "원 안에 든 개수"를 세는 **정수** 합이라 `atomicAdd(unsigned long long*)`를 쓸 수 있었다. 그런데 적분은 **실수** 합이고, `atomicAdd(double*)`는 **Compute Capability 6.0(sm_60) 이상에서만** 제공된다. 이 프로젝트는 CUDA 기본 설정인 `compute_52`로 빌드하므로 그 함수 자체가 없다.
> 그래서 스레드마다 자기 칸(`partialSums[id]`)에 부분합을 쓰고, 마지막에 `thrust::reduce`로 한 번에 합친다. 원자 연산 경합이 아예 없고, 덧셈 순서가 실행마다 고정되므로 **결과가 결정적(deterministic)** 이라는 이점도 있다(부동소수점 덧셈은 순서에 따라 결과가 미세하게 달라진다).

작업을 스레드와 샘플로 쪼개는 부분은 다음 한 함수가 담당한다. 샘플 수 N이 스레드 수보다 많으면 스레드당 여러 샘플을 돌린다.

```cpp
static void SplitWork(long long n, int& numThreads, int& samplesPerThread)
{
	numThreads = int(n < (1LL << 20) ? n : (1LL << 20));   // 최대 100만 스레드
	samplesPerThread = int(n / numThreads);
}
```

### 결과

같은 커널에 함수만 바꿔서 세 가지를 적분했다(`--demo integrate`, N = 1,000,000). 기준값은 호스트에서 따로 계산했다 — $\sin^5$는 부정적분 $-\cos x + \tfrac{2}{3}\cos^3 x - \tfrac{1}{5}\cos^5 x$ 로, $\ln(\sin x)$는 초등함수 부정적분이 없어서 **중점 규칙(1천만 칸)** 으로 구했다.

```text
[integrate] uniform Monte Carlo on [0, 2], N = 1000000
f(x)               Monte Carlo         reference           |error|
x^2             2.666781136105    2.666666666667    0.000114469438
sin^5(x)        0.903325618913    0.903931238481    0.000605619568
ln(sin(x))     -1.102914787695   -1.102222319590    0.000692468105
```

세 함수 모두 몬테카를로 쪽 코드는 **난이도가 똑같다**. 해석적 적분이 어려울수록, 또는 아예 불가능할수록 몬테카를로의 상대적 가치가 올라간다.

---

## 밀도 함수 (Density Functions)

여기서 잠시 방향을 틀어, 앞으로 쓸 확률 도구를 만든다. 왜 필요한지부터 보자.

1·2권의 `RayColor`는 단순하고 우아하지만 **큰 약점**이 하나 있다. **작은 광원에서 노이즈가 심하다.** 우리는 방향을 균일하게 뽑아 산란시키므로, 광원은 "레이가 우연히 그쪽으로 튀었을 때만" 샘플링된다. 광원이 작거나 멀면 그 확률이 아주 낮다. 배경이 검정이면 장면의 유일한 빛은 광원뿐인데, 그 광원을 좀처럼 못 맞히는 것이다.

그 결과 바로 옆에 있는 두 레이가 극단적으로 갈린다. 우연히 광원 쪽으로 반사된 레이는 **아주 밝은** 값을 가져오고, 다른 데로 간 레이는 **아주 어두운** 값을 가져온다. 실제 정답은 그 중간 어딘가인데, 픽셀마다 운에 따라 밝고 어두운 점이 튀는 것이다.

그럼 "두 레이를 모두 광원 쪽으로 몰아 주면" 어떨까? 노이즈는 줄지만 이번엔 **그림 전체가 실제보다 밝아진다.** 왜 그런지는 레이를 거꾸로 생각하면 분명하다. 보통 우리는 카메라에서 출발해 장면을 거쳐 광원에서 끝나는 레이를 추적한다. 반대로 광원에서 출발해 카메라에 도달하는 레이를 상상해 보자. 이 레이는 밝게 출발해 튕길 때마다 표면 색에 물들고 에너지를 잃다가 카메라에 닿는다. 그런데 이 레이를 **가능한 한 빨리 카메라 쪽으로 꺾어** 버리면, 충분히 어두워지기 전에 도착하므로 실제보다 밝아진다. "광원 쪽으로 샘플을 더 보내는 것"이 바로 이 상황이다.

해결책은 **더 자주 뽑은 만큼 가중치를 낮춰 주는 것**이다. 그러려면 "얼마나 자주 뽑았는지"를 수로 표현해야 하고, 그것이 **확률 밀도 함수(PDF)** 다. 그리고 PDF를 이해하려면 먼저 **밀도 함수**가 무엇인지 알아야 한다.

**밀도 함수는 히스토그램의 연속 버전이다.** 단계적으로 보자.

- **히스토그램**: 데이터를 칸(bin)으로 나누고 각 칸에 든 **개수**를 세운 것. 데이터가 많아지면 칸 수는 그대로여도 각 칸의 개수가 커진다. 칸을 잘게 쪼개면 칸 수는 늘고 각 칸의 개수는 줄어든다. 칸 수를 무한히 늘리면 결국 **모든 칸의 개수가 0**이 되어 버린다 — 개수로는 연속으로 갈 수 없다.
- **이산 밀도 함수**: 그래서 y축을 "개수"가 아니라 **전체 대비 비율**로 바꾼다.

$$ \text{칸 } i \text{의 밀도} = \frac{\text{칸 } i \text{의 개수}}{\text{전체 개수}} $$

- **(연속) 밀도 함수**: 여기에 칸 폭으로 한 번 더 나누면 칸을 무한히 잘게 만들어도 값이 0으로 무너지지 않는다.

$$ \text{칸 밀도} = \frac{(\text{높이 } H \text{와 } H' \text{ 사이 나무의 비율})}{(H - H')} $$

- **확률 밀도 함수(PDF)**: 이 연속 밀도에 칸 폭을 곱하면 "값이 그 구간에 들어갈 확률"이 된다.

$$ \text{임의의 나무 높이가 } H \text{와 } H' \text{ 사이일 확률} = \text{칸 밀도}\cdot(H - H') $$

즉 **PDF는 적분해서 "어떤 구간이 나올 확률"을 얻을 수 있는 연속 함수**다.

![그림 3: 히스토그램 예](https://raytracing.github.io/images/fig-3.03-histogram.jpg)

---

## PDF 만들기 (Constructing a PDF)

감을 잡기 위해 직접 하나 만들어 보자. `[0, 2]` 구간에서 0부터 시작해 선형으로 증가하는 함수를 쓴다.

![그림 4: 선형 PDF](https://raytracing.github.io/images/fig-3.04-linear-pdf.jpg)

이 함수를 PDF로 삼아 난수를 만들면, **0 근처가 나올 확률보다 2 근처가 나올 확률이 크다.** 그런데 $p(2)$의 값은 정확히 얼마여야 할까? 선형이니 2쯤 되지 않을까 싶지만, 그렇게 어림해서는 안 된다. PDF에는 지켜야 할 제약이 있기 때문이다.

값이 `[0, 2]` 안에 있다는 것이 확실하다면, 그 구간에서 값이 나올 확률은 100%다. 확률은 곡선 아래 넓이이므로 **전체 넓이가 정확히 1**이어야 한다.

$$ \operatorname{area}(p(r), 0, 2) = 1 $$

모든 선형 함수는 $p(r) = C\cdot r$ 꼴로 쓸 수 있으니, 적분해서 거꾸로 $C$를 구한다.

$$ 1 = \int_0^2 C\,r\,dr = C\cdot\frac{r^2}{2}\Big|_0^2 = C\Big(\frac{4}{2} - 0\Big) = 2C \;\Rightarrow\; C = \frac{1}{2} $$

$$ \boxed{p(r) = \frac{r}{2}} $$

이제 히스토그램에서 하던 것처럼, 구간을 적분하면 그 구간이 나올 확률이 된다.

$$ \operatorname{Prob}(x_0 \le r \le x_1) = \int_{x_0}^{x_1}\frac{r}{2}\,dr $$

확인 삼아 $r = 0$부터 $2$까지 적분하면 정확히 1이 나온다.

> ⚠️ **PDF 값 자체는 확률이 아니다.** PDF에 익숙해지면 $p(r = x)$를 "변수 $r$이 값 $x$일 확률"처럼 부르고 싶어지는데, 그러면 안 된다. 연속 함수에서 **변수가 특정한 값 하나일 확률은 언제나 0**이다. 칸의 폭이 0이기 때문이다. 증명도 한 줄이다.
>
> $$ \operatorname{Prob}(r = x) = \int_x^x p(r)\,dr = P(r)\Big|_x^x = P(x) - P(x) = 0 $$
>
> 반면 $x$ 주변의 **구간**에 대한 확률은 0이 아니다.
>
> $$ \operatorname{Prob}(x - \Delta x < r < x + \Delta x) = P(x + \Delta x) - P(x - \Delta x) $$

---

## 샘플 고르기 (Choosing our Samples)

PDF가 있으면 "어느 구간이 얼마나 자주 나올지"를 안다. 이제 필요한 것은 그 반대다 — **균일 난수만 주는 생성기로 원하는 PDF를 따르는 난수를 만드는 것**이다.

균일 PDF는 쉽다. `[0, 10]`에서 균일하게 뽑고 싶으면 `10.0 * random_double()` 이면 끝이다. 문제는 우리가 실제로 다룰 대부분의 경우가 **비균일**이라는 점이다. 그래서 "균일 난수를 받아 PDF에 맞게 휘어진 분포를 내놓는 함수 $f(d)$가 있다"고 **가정**하고, 그 $f$를 찾아 나선다.

### 먼저 직관: 확률을 반으로 가르는 지점

$p(r) = r/2$에서는 2 근처가 0 근처보다 잘 나온다. 그렇다면 **확률을 정확히 반으로 가르는 값**은 어디일까? "이 값보다 클 확률 50%, 작을 확률 50%"인 지점 말이다.

$$ 50\% = \int_0^x \frac{r}{2}\,dr = \int_x^2 \frac{r}{2}\,dr $$

왼쪽을 풀면

$$ 0.5 = \frac{r^2}{4}\Big|_0^x = \frac{x^2}{4} \;\Rightarrow\; x^2 = 2 \;\Rightarrow\; x = \sqrt{2} $$

이 사실 하나만으로도 **거친 근사 생성기**를 만들 수 있다. 균일 난수 `d`가 0.5 이하이면 `[0, √2]`에서 균일하게, 초과면 `[√2, 2]`에서 균일하게 뽑는 것이다.

```cpp
double f(double d)
{
    if (d <= 0.5)
        return std::sqrt(2.0) * random_double();
    else
        return std::sqrt(2.0) + (2 - std::sqrt(2.0)) * random_double();
}
```

원래 생성기가 `[0,1]`에서 완전히 평평했다면, 이 함수를 거친 결과는 **계단 두 개짜리로 아주 조금 기울어진** 분포가 된다. $r/2$에 완벽하지는 않지만 방향은 맞다.

![그림 5: 균일 분포](https://raytracing.github.io/images/fig-3.05-uniform-dist.jpg)

![그림 6: r/2에 대한 비균일(거친) 분포](https://raytracing.github.io/images/fig-3.06-nonuniform-dist.jpg)

### 해석적으로 못 푸는 함수라면: 실험으로 절반 지점 찾기

위에서는 적분의 해석해가 있어서 √2를 바로 구했다. 하지만 적분을 풀 수 없거나 풀기 싫은 함수도 있다. 예를 들어

$$ p(x) = e^{-\frac{x}{2\pi}}\sin^2(x) $$

![그림 7: 해석적으로 풀고 싶지 않은 함수](https://raytracing.github.io/images/fig-3.07-exp-sin2.jpg)

이럴 때는 **실험으로** 절반 지점을 찾으면 된다. 방법은 이렇다.

1. 구간(여기서는 `[0, 2π]`)에서 샘플을 많이 뽑고, 각 샘플의 `x`와 `p(x)`를 **함께 저장**한다.
2. 전체 합을 구해 둔다(이것이 넓이에 비례한다).
3. 샘플을 **x 기준으로 정렬**한다.
4. 앞에서부터 `p(x)`를 더해 가다가 **전체 합의 절반을 넘는 순간**의 x가 곧 절반 지점이다.

### GPU로 옮기기: 정렬 → 누적합 → 이진 탐색

원서는 3~4단계를 `std::sort` 후 for 루프로 처리한다. 순차적으로 보이지만, 이 셋은 각각 **GPU 병렬 기본 연산(parallel primitive)** 에 정확히 대응한다.

| 단계 | 순차 코드 | 병렬 기본 연산 |
|---|---|---|
| x 기준 정렬(p를 같이 끌고) | `std::sort` | `thrust::sort_by_key` |
| 앞에서부터 누적해 더하기 | 누적 for 루프 | `thrust::inclusive_scan` (prefix sum) |
| 절반을 넘는 첫 위치 찾기 | 선형 탐색 | `thrust::lower_bound` (이진 탐색) |

특히 2번이 중요하다. 누적합은 "앞의 결과가 있어야 뒤를 계산할 수 있는" 전형적인 순차 작업처럼 보이지만, **실제로는 O(log n) 깊이로 병렬화되는 대표적인 연산**이다. 누적합이 정렬된 배열이 되므로 3번은 자연스럽게 이진 탐색이 된다.

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

> 🔧 **왜 thrust를 별도 파일로 뺐나**: thrust/CUB 헤더는 매우 무거워서, 데모 코드가 들어 있는 `MonteCarloDemo.cu`에 직접 include하면 그 파일을 고칠 때마다 컴파일이 느려진다. 그래서 thrust 호출은 `DemoThrust.cu` 한 곳에 모으고, 바깥에는 **원시 디바이스 포인터만 받는 평범한 함수**(`DeviceSum`, `DeviceHalfwayPoint`)를 노출했다.

### 결과

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

원서와 같은 1만 개에서는 절반 지점이 0.01 정도 흔들리고(시드에 따라 원서 값인 2.01대가 나오기도 한다), 천 배인 1천만 개에서는 기준값 2.0384에 소수점 셋째 자리까지 맞는다.

### 이 방법을 끝까지 밀면?

절반 지점을 찾았으면 **아래쪽 절반과 위쪽 절반에 대해 같은 일을 반복**할 수 있다.

1. PDF의 절반 지점을 구한다.
2. 아래쪽 절반에 대해 1번을 반복한다.
3. 위쪽 절반에 대해 1번을 반복한다.

적당한 깊이(예: 6~10단계)에서 멈추면 임의의 PDF를 흉내 내는 생성기가 만들어진다. 다만 비용이 만만치 않다 — 병목은 정렬이고, 소박한 정렬은 $O(n^2)$, 표준 라이브러리라도 $O(n\log n)$이다. 수백만~수십억 샘플에서는 부담이 크다. 이런 분할 정복은 실제로도 쓰이지만, 임의의 함수를 PDF로 삼고 싶다면 **Metropolis-Hastings** 같은 더 효율적인 방법을 찾아보는 편이 낫다.

---

## 분포 근사하기: 누적 분포 함수(CDF)와 그 역함수

앞의 "절반 지점 찾기"를 다시 보면, 사실 우리는 **누적된 확률**을 다루고 있었다. 이걸 일반화한 것이 **누적 분포 함수(cumulative distribution function, CDF)** 다.

$$ P(x) = \int_{-\infty}^{x} p(x')\,dx' $$

PDF와 CDF는 미분/적분 관계다.

$$ p(x) = \frac{d}{dx}P(x) $$

두 함수의 역할은 서로 다르다.

- **PDF $p(x)$**: "이 근처가 얼마나 조밀한가" — 값 자체는 확률이 아니고, **구간에 대해 적분해야** 확률이 된다.
- **CDF $P(x)$**: "뽑은 값이 $x$ **이하**일 확률" — 값 자체가 곧 확률이다. 항상 0에서 1로 단조 증가한다.

우리가 쓰는 $p(r)$은 `[0, 2]` 바깥에서 0이므로, 구간별로 나눠 쓰면 이렇다.

$$
p(r) =
\begin{cases}
0 & r < 0 \\
\dfrac{r}{2} & 0 \le r \le 2 \\
0 & r > 2
\end{cases}
\qquad
P(r) =
\begin{cases}
0 & r < 0 \\
\dfrac{r^2}{4} & 0 \le r \le 2 \\
1 & r > 2
\end{cases}
$$

감을 잡기 위해 값을 하나 넣어 보자. $P(1) = 1/4$ 이다. 이것은 "이 분포에서 뽑은 값이 1 이하일 확률이 25%"라는 뜻이고, 바꿔 말하면 **뽑은 값의 25%가 구간 `[0,1]`에 떨어진다**는 뜻이다. 25%밖에 안 되는 이유는 $p$가 오른쪽으로 갈수록 커지기 때문이다(구간 길이는 절반인데 확률은 1/4).

### 우리가 진짜 원하는 건 CDF의 역함수

목표를 다시 적자. 우리에게는 균일 난수 $d$가 있고, 이 분포를 따르는 $x$를 내놓는 함수 $f(d)$가 필요하다.

방금 "값의 25%가 1 이하"임을 알았다. 그렇다면 균일 난수의 **아래쪽 25%** 가 `[0,1]`로 가야 한다. 즉 $f(0.25) = 1$ 이다. 마찬가지로 절반 지점이 $\sqrt2$ 였으니 $f(0.5) = \sqrt2$. 이것을 일반화하면

$$ f(P(x)) = x $$

즉 **$f$는 CDF의 역함수**다. 이 역함수에는 이름이 있다 — **역누적분포함수(inverse cumulative distribution function, ICD)** 라고 부른다. "CDF를 뒤집는다"는 이 한 줄이 앞으로 이 책 전체에서 방향 벡터를 만드는 도구가 된다(7장의 코사인 방향, 12장의 구를 향한 원뿔 방향 모두 같은 방식이다).

직접 뒤집어 보자.

$$ y = \frac{r^2}{4} \;\Longrightarrow\; r^2 = 4y \;\Longrightarrow\; r = \sqrt{4y} $$

$$ \boxed{\;\operatorname{ICD}(d) = \sqrt{4d}\;} $$

검산해 보면 기대한 값이 그대로 나온다.

| $d$ | $\operatorname{ICD}(d) = \sqrt{4d}$ | 의미 |
|---|---|---|
| 0 | 0 | 구간의 왼쪽 끝 |
| 0.25 | 1 | 25%가 1 이하 ✔ |
| 0.5 | $\sqrt2 \approx 1.414$ | 절반 지점 ✔ |
| 1 | 2 | 구간의 오른쪽 끝 |

범위도 정확히 `[0, 2]`다. 다시 말해 `sqrt(4.0 * random_double())` 한 줄이면 $p(r) = r/2$ 분포를 따르는 난수가 나온다. 거절법도, 루프도, 정렬도 없다.

![그림 8: 비균일 f() 근사](https://raytracing.github.io/images/fig-3.08-approx-f.jpg)

> 🔧 **CUDA 관점에서 왜 이게 그렇게 좋은가**: ICD 샘플링은 **분기도 루프도 없는 고정 비용** 코드다. 워프 안 32개 레인이 전부 같은 명령을 같은 횟수로 실행하고 동시에 끝난다. 반대로 거절법(rejection)은 레인마다 시도 횟수가 다르므로 워프 전체가 **가장 운 나쁜 레인을 기다린다**. 4장에서 이 비용 차이를 실제로 측정하고, 7장에서는 거절법으로 만들던 코사인 방향을 ICD로 바꿔 둘을 나란히 비교한다.

물론 만능은 아니다. PDF를 해석적으로 적분할 수 없으면 CDF가 없고, CDF를 대수적으로 뒤집을 수 없으면 ICD도 없다. 그럴 때는 앞에서처럼 샘플을 정렬하거나 히스토그램으로 CDF를 근사하고, 이진 탐색으로 역함수를 대신한다 — 우리가 `thrust::inclusive_scan` + `thrust::lower_bound`로 한 일이 정확히 그 "수치적 ICD"다.

---

## 중요도 샘플링 (Importance Sampling)

이제 도구가 다 모였다. 처음의 $\int_0^2 x^2\,dx$로 돌아가, **PDF를 바꿔 가며** 같은 적분을 풀어 보자.

### 균일 샘플링을 다시 쓰기

앞에서 균일 샘플링 코드는 이렇게 생겼었다.

```cpp
sum += x * x;
...
return 2.0 * sum / N;     // (b - a) * 평균
```

맨 앞의 `2.0`이 어디서 왔는지 다시 보자. 우리는 `[0,2]`에서 **균일하게** 뽑았고, 균일 분포의 PDF는 구간 길이의 역수, 즉 $p(x) = 1/2$ 이다. 그러니 저 `2.0`은 사실 $1/p(x)$ 다. 이걸 밖으로 빼지 않고 샘플마다 명시적으로 나눠 쓰면 이렇게 된다.

```cpp
double pdf(double x) { return 0.5; }      // [0,2] 균일
...
sum += x * x / pdf(x);
...
return sum / N;                            // (b-a)가 사라졌다
```

값은 똑같은데, 식의 모양이 훨씬 일반적이 되었다. **구간 길이라는 특수한 상수가 사라지고, PDF로 나누는 일반 규칙만 남았다.**

### 몬테카를로의 일반 형태

$$ \boxed{\;\int_a^b f(x)\,dx \;\approx\; \frac{1}{N}\sum_{i=1}^{N}\frac{f(r_i)}{p(r_i)}, \qquad r_i \sim p\;} $$

왜 $p$로 나누는 것이 편향을 없애는가? 이렇게 생각하면 된다. $p$가 큰 곳은 샘플이 **자주** 뽑히므로 합에 여러 번 들어간다. 그러니 그만큼 **적게 쳐줘야** 한다. 반대로 $p$가 작은 곳은 어쩌다 한 번 뽑히므로 **크게 쳐줘야** 한다. 그 "적게/크게"의 정확한 비율이 $1/p$다. 실제로 기댓값을 계산해 보면

$$ E\!\left[\frac{f(X)}{p(X)}\right] = \int \frac{f(x)}{p(x)}\,p(x)\,dx = \int f(x)\,dx $$

$p$가 깨끗하게 약분된다. **어떤 $p$를 골라도 (0이 아닌 곳에서만 고르면) 추정값의 기댓값은 항상 참값이다.** PDF는 정답을 바꾸지 않는다. 바꾸는 것은 **분산** — 즉 수렴 속도뿐이다.

### 네 가지 PDF 비교

원서가 차례로 보여 주는 PDF들에, 원서가 코드로는 보여 주지 않는 half-split까지 더해 네 가지를 나란히 구현했다.

| PDF | $p(x)$ | CDF $P(x)$ | 샘플 생성 (d는 균일 난수) |
|---|---|---|---|
| uniform | 1/2 | x/2 | ICD(d) = 2d |
| half-split (원서의 거친 근사) | `[0,√2]`에서 0.5/√2, `[√2,2]`에서 0.5/(2-√2) | 구간별 직선 | 반 확률로 아래/위 구간에서 균일 |
| linear | x/2 | x²/4 | ICD(d) = √(4d) |
| quadratic | (3/8) x² | x³/8 | ICD(d) = (8d)^(1/3) = 2·d^(1/3) |

linear는 `[0,2]`에서 $\int x/2 = 1$ 이라 정규화가 맞고, quadratic은 $\int_0^2 C x^2 = C\cdot 8/3 = 1$ 에서 $C = 3/8$ 로 정한 것이다. quadratic은 **피적분 함수 $x^2$와 모양이 똑같은** PDF라는 점이 핵심이다.

> ⚠️ **원서의 2차 PDF 역함수 표기 오류**: $P(x) = x^3/8$의 역함수는 $(8d)^{1/3} = 2\,d^{1/3}$인데, 원서 본문과 코드는 $8\,d^{1/3}$로 적혀 있다(이러면 x가 `[0,8]`까지 튄다). 그런데도 원서의 출력이 정확히 2.666667로 나오는 이유가 재미있다. 이 PDF에서는
> $$ \frac{f(x)}{p(x)} = \frac{x^2}{\tfrac{3}{8}x^2} = \frac{8}{3} $$
> 로 **x가 통째로 약분되어** 어떤 x를 뽑든 같은 값이 나오기 때문이다. 즉 잘못된 범위의 x를 뽑아도 결과가 같다. 우리는 바른 식 `2.0 * cbrt(d)`로 구현했다.

### 구현

> 📄 **파일: `MonteCarloDemo.cu`** — 원서 Listing `integ-xsq-2`, `integ-xsq-3`, `integ-xsq-5`에 대응

```cpp
// 균일 난수 → 원하는 분포를 따르는 x (ICD 샘플링).
// curand_uniform_double은 (0,1]을 돌려주므로 d = 0 이 절대 나오지 않는다.
// 그래서 원서의 "if (z == 0.0) continue;" 가드가 필요 없다.
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
// 샘플 하나당 표준편차(= 노이즈의 크기)까지 잰다.
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

> 🔧 **왜 제곱합까지 모으나**: 위 논의의 결론이 "PDF는 답이 아니라 **분산**을 바꾼다"이므로, 답만 찍어 봐서는 아무것도 비교할 수 없다(넷 다 2.666…이 나온다). 그래서 $\sum g$ 와 $\sum g^2$ 를 함께 누적해 샘플 하나당 표준편차 $\sigma = \sqrt{E[g^2] - E[g]^2}$ 를 계산한다. 이 값이 곧 **"같은 정확도를 얻는 데 필요한 샘플 수"** 를 알려 준다 — 오차는 $\sigma/\sqrt{N}$ 으로 줄기 때문에, σ가 절반이면 샘플 수는 1/4로 충분하다. 배열 두 개(`partialSums`, `partialSumSquares`)를 쓰는 이유는 3장 앞부분에서 설명한 것과 같다(`atomicAdd(double*)` 회피 + 결정적 결과).

> 🔧 **half-split은 우리가 덧붙인 실험**: 원서는 이 거친 분포를 "뽑는" 코드만 보여 주고 밀도는 적지 않은 채 "균일보다 빠르고 선형보다 느릴 것"이라고만 말한다. $f/p$를 계산하려면 밀도가 반드시 필요하므로, 각 구간에 확률 1/2이 균일하게 퍼진다는 사실에서 `0.5 / 구간 길이`로 밀도를 직접 유도해 넣고 실제로 확인했다.

### 결과

`--demo importance` 실행 결과다(괄호 안은 |오차|, 마지막 열은 N = 1,048,576에서 잰 샘플 하나당 표준편차).

```text
pdf                     N=1                 N=100               N=10000             N=1048576   stddev/sample
uniform     1.27390469 (1.3927)  2.34748618 (0.3192)  2.65869372 (0.0080)  2.66530865 (0.0014)   2.3864619434
half-split  0.37566730 (2.2910)  2.80836213 (0.1417)  2.67464254 (0.0080)  2.66796445 (0.0013)   1.5026860787
linear      1.15660541 (1.5101)  2.65791027 (0.0088)  2.64772709 (0.0189)  2.66779381 (0.0011)   0.9423348548
quadratic   2.66666667 (0.0000)  2.66666667 (0.0000)  2.66666667 (0.0000)  2.66666667 (0.0000)   0.0000000000
```

읽는 법:

- **네 PDF 모두 8/3으로 수렴한다.** 위에서 증명한 대로 PDF는 정답을 바꾸지 않는다.
- **수렴 속도는 표준편차에 그대로 드러난다**: uniform 2.39 → half-split 1.50 → linear 0.94 → quadratic 0. 즉 linear PDF는 uniform과 같은 정확도를 **(2.386/0.942)² ≈ 6.4배 적은 샘플**로 얻는다. 원서가 "빨리 수렴한다"고 말한 것을 숫자로 확인한 셈이다.
- 해석값과도 맞다. uniform은 $\sqrt{\tfrac{64}{5} - \tfrac{64}{9}} \approx 2.385$, linear는 $\sqrt{8 - \tfrac{64}{9}} \approx 0.943$ 이다.
- 피적분 함수와 모양이 완전히 같은 PDF(quadratic)는 $f/p$가 상수라 **샘플 하나로 정답**을 낸다. 분산이 0이다. 물론 이런 PDF를 만들려면 $\int f$를 이미 알아야 하므로 실전에서는 쓸 수 없다 — 코드가 맞는지 확인하는 **검산용**이다.
- N이 작을 때 half-split의 오차가 uniform보다 커 보이는 것은 단순히 운이다. N=1은 샘플 한 개이므로 어떤 PDF든 오차가 요동친다. 비교해야 할 것은 마지막 열이다.

정리하면, 비균일 PDF는 샘플을 $p$가 큰 쪽으로 몰아준다. 그러니 **$f$가 큰 곳(= 결과에 많이 기여하는 곳)에 $p$를 크게 잡으면** 같은 샘플 수로 훨씬 빨리 수렴한다. "중요한 곳을 더 많이 뽑는다"고 해서 **중요도 샘플링(importance sampling)** 이다.

### 정리: 몬테카를로 레이트레이서의 기본 절차

1. 어떤 영역에서 $f(x)$의 적분을 구하고 싶다.
2. 그 영역에서 0이 아니고 음수도 아닌 PDF $p$를 고른다.
3. $p$를 따르는 난수 $r$로 $f(r)/p(r)$ 을 아주 많이 평균낸다.

어떤 $p$를 골라도 결국 정답에 수렴하고, $p$가 $f$와 닮을수록 빨리 수렴한다. 이 세 줄이 남은 장 전체의 뼈대다. 다음 장부터는 이 규칙을 1차원 $x$에서 **방향(구면 위의 입체각)** 으로 넓히고, 최종적으로 $f$ 자리에 "빛의 세기 × 산란 확률"을, $p$ 자리에 "광원 쪽으로 향하는 분포"를 넣는다.

---

## 결과 & 검증

- **빌드/실행 확인**: VS2022 + CUDA 12.9 Release로 컴파일·링크·실행 성공.
- **`--demo integrate`**: x², sin⁵, ln(sin) 모두 기준값과 소수점 셋째 자리 안쪽으로 일치(N = 10⁶).
- **`--demo halfway`**: 1천만 샘플에서 절반 지점 2.03804 (중점 규칙 기준 2.03843).
- **`--demo importance`**: 네 PDF 모두 8/3으로 수렴, 샘플당 표준편차가 해석값과 일치(uniform 2.386, linear 0.943).

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

1. **실수 합은 부분합 배열 → 병렬 리덕션**: `atomicAdd(double*)`는 sm_60 이상 전용이고, 우리 프로젝트는 `compute_52`까지 내려가는 설정으로 빌드된다. 스레드별 부분합을 쓰고 `thrust::reduce`로 합치면 아키텍처와 무관하게 동작하고, 덧셈 순서가 고정돼 **결과도 결정적**이다.
2. **"정렬 후 누적"은 sort + scan + binary search**: 순차 알고리즘처럼 보이는 누적합도 병렬 기본 연산의 조합으로 그대로 GPU에 올라간다. 해석적 CDF가 없는 PDF에서는 이 세 연산이 곧 "수치적 ICD"가 된다.
3. **curand의 `(0,1]`을 역이용**: `curand_uniform_double`은 0을 절대 돌려주지 않으므로, ICD 샘플링에서 0으로 나누기나 `ln 0`을 따로 걸러낼 필요가 없다(원서의 `if (z == 0.0) continue;`가 불필요). 반대로 `[0,1)`이 필요하면 `1.0 - u`로 뒤집어 쓴다.
