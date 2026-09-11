// === Ray Tracing: The Rest of Your Life — 몬테카를로 데모 (GPU 버전) ===
//
// 원서 3권 앞부분은 레이트레이서가 아니라 "작은 수치 실험 프로그램"으로 몬테카를로의
// 감을 잡는다(pi.cc, integrate_x_sq.cc, ...). 원서는 CPU for 루프로 샘플을 하나씩
// 뽑지만, 여기서는 같은 실험을 GPU 커널로 옮긴다:
//   - 스레드 하나가 샘플 여러 개(또는 격자 한 칸)를 맡는다.
//   - 스레드별 결과는 원자 연산(atomicAdd)이나 블록 단위 집계로 합친다.
//   - 난수는 cuRAND Philox 생성기를 쓴다(스레드가 수백만 개여도 초기화가 빠르다).
//
// 렌더러(kernel.cu)와는 독립된 번역 단위다. main()에서 --demo <이름>으로 호출한다.
//
// ※ 이 파일은 반드시 CRLF 줄끝으로 저장해야 한다. nvcc의 EDG 프런트엔드는 UTF-8
//   한글 주석을 cp949로 읽는데, 주석이 특정 한글 글자로 끝나면 마지막 바이트가
//   다음 바이트(LF 파일에서는 줄바꿈 '\n')를 삼켜 버린다. 그러면 다음 줄이 주석에
//   먹혀 엉뚱한 곳에서 "expected a ;"가 난다. CRLF면 '\r'만 먹히고 '\n'은 남는다.
//   (Docs/빌드환경_및_트러블슈팅.md ④ 참고)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "MonteCarloDemo.h"
#include "DemoThrust.h"   // thrust 병렬 기본 연산(리덕션/정렬/누적합)은 별도 파일로 격리
#include "Vec3.h"         // 4장부터: 방향(단위 벡터)을 다룬다

#define checkDemoErrors(val) CheckDemoCuda((val), #val, __FILE__, __LINE__)

static void CheckDemoCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		fprintf(stderr, "CUDA error = %u at %s:%d '%s'\n",
			static_cast<unsigned int>(result), file, line, func);
		cudaDeviceReset();
		exit(99);
	}
}

// constexpr이어야 디바이스 코드에서도 쓸 수 있다(일반 const double은 호스트 전용).
static constexpr double kPi = 3.1415926535897932385;

// 데모 전용 난수 상태.
// 렌더러는 curandState(XORWOW)를 쓰지만, XORWOW는 서브시퀀스마다 건너뛰기(skip-ahead)
// 비용이 커서 스레드를 수백만 개 띄우는 데모에는 초기화가 부담이 된다.
// Philox는 카운터 기반 생성기라 curand_init이 사실상 공짜다.
typedef curandStatePhilox4_32_10_t DemoRng;

// [0,1) 균일 난수. curand_uniform_double은 (0,1]을 돌려주므로 1에서 빼서
// 원서 random_double()과 같은 반열린 구간 [0,1)로 맞춘다.
__device__ inline double RandomDouble(DemoRng* rng)
{
	return 1.0 - curand_uniform_double(rng);
}

// [minimum, maximum) 균일 난수
__device__ inline double RandomDouble(DemoRng* rng, double minimum, double maximum)
{
	return minimum + (maximum - minimum) * RandomDouble(rng);
}

// ─────────────────────────────────────────────────────────────────────
// 2장: 간단한 몬테카를로 프로그램 — π 추정
// ─────────────────────────────────────────────────────────────────────

// [-1,1]^2 정사각형에 무작위 점을 뿌려 단위원 안에 떨어진 개수를 센다.
// 원과 정사각형의 넓이 비가 π/4 이므로 (안쪽 개수 / 전체 개수) * 4 ≈ π.
// 스레드마다 samplesPerThread개를 뽑아 지역 카운트를 만든 뒤, 전역 카운터에는
// atomicAdd를 한 번만 한다(원자 연산 경합을 줄이는 기본 패턴).
__global__ void PiCountKernel(
	unsigned long long seed, int numThreads, int samplesPerThread,
	unsigned long long* insideCount)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numThreads) return;

	DemoRng rng;
	curand_init(seed, id, 0, &rng);

	unsigned long long inside = 0;
	for (int k = 0; k < samplesPerThread; k++)
	{
		double x = RandomDouble(&rng, -1.0, 1.0);
		double y = RandomDouble(&rng, -1.0, 1.0);
		if (x * x + y * y < 1.0)
			inside++;
	}

	atomicAdd(insideCount, inside);
}

// 격자 한 칸 = 스레드 하나. 같은 칸에서
//   regular    : 정사각형 전체에서 뽑은 순수 무작위 점
//   stratified : 자기 칸 안에서만 뽑은 지터링 점
// 을 하나씩 만들어 비교한다. 블록 안 합계는 __syncthreads_count 한 번으로 구한다.
// (주의: __syncthreads_count는 블록의 모든 스레드가 호출해야 한다. 그래서 격자
//  밖 스레드도 return 하지 않고 0을 든 채 끝까지 따라온다.)
__global__ void PiStratifiedKernel(
	unsigned long long seed, int sqrtN,
	unsigned long long* insideRegular, unsigned long long* insideStratified)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	bool bValid = (i < sqrtN) && (j < sqrtN);

	int regular = 0;
	int stratified = 0;

	if (bValid)
	{
		DemoRng rng;
		curand_init(seed, (unsigned long long)j * sqrtN + i, 0, &rng);

		double x = RandomDouble(&rng, -1.0, 1.0);
		double y = RandomDouble(&rng, -1.0, 1.0);
		regular = (x * x + y * y < 1.0) ? 1 : 0;

		x = 2.0 * ((i + RandomDouble(&rng)) / sqrtN) - 1.0;
		y = 2.0 * ((j + RandomDouble(&rng)) / sqrtN) - 1.0;
		stratified = (x * x + y * y < 1.0) ? 1 : 0;
	}

	int blockRegular = __syncthreads_count(regular);
	int blockStratified = __syncthreads_count(stratified);

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		atomicAdd(insideRegular, (unsigned long long)blockRegular);
		atomicAdd(insideStratified, (unsigned long long)blockStratified);
	}
}

// numThreads x samplesPerThread 개의 점 중 원 안에 든 개수를 GPU로 센다.
static unsigned long long CountInsideCircle(
	unsigned long long seed, int numThreads, int samplesPerThread, unsigned long long* dCount)
{
	checkDemoErrors(cudaMemset(dCount, 0, sizeof(unsigned long long)));

	int threadsPerBlock = 256;
	int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
	PiCountKernel<<<blocks, threadsPerBlock>>>(seed, numThreads, samplesPerThread, dCount);
	checkDemoErrors(cudaGetLastError());

	unsigned long long inside = 0;
	checkDemoErrors(cudaMemcpy(&inside, dCount, sizeof(inside), cudaMemcpyDeviceToHost));
	return inside;
}

static void DemoPi()
{
	unsigned long long* dCounts;
	checkDemoErrors(cudaMalloc((void**)&dCounts, 2 * sizeof(unsigned long long)));

	// --- 원서 Listing estpi-1: N = 100,000개로 한 번 추정 ---
	{
		const int N = 100000;
		unsigned long long inside = CountInsideCircle(2026, N, 1, dCounts);
		printf("[pi v1] N = %d\n", N);
		printf("Estimate of Pi = %.12f\n\n", 4.0 * double(inside) / N);
	}

	// --- 원서 Listing estpi-2: 수렴 과정 보기 ---
	// 원서는 무한 루프를 돌며 누적 추정값을 계속 덮어 찍는다. GPU에서는 라운드마다
	// 샘플 수를 두 배로 늘려 가며 "누적" 추정값과 오차를 표로 남긴다.
	// N이 4배가 될 때 오차가 대략 절반으로 줄어드는(1/sqrt(N)) 모습이 보인다.
	{
		printf("[pi v2] running estimate (batch size doubles every round)\n");
		printf("%16s  %16s  %16s\n", "N", "Estimate of Pi", "|error|");

		unsigned long long totalInside = 0;
		unsigned long long totalRuns = 0;
		for (int round = 0; round <= 20; round++)
		{
			unsigned long long batch = 1024ULL << round;
			int numThreads = int(batch < (1ULL << 20) ? batch : (1ULL << 20));
			int samplesPerThread = int(batch / numThreads);

			// 라운드마다 시드를 바꿔 이전 라운드와 겹치지 않는 난수열을 쓴다.
			totalInside += CountInsideCircle(1000 + round, numThreads, samplesPerThread, dCounts);
			totalRuns += batch;

			double estimate = 4.0 * double(totalInside) / double(totalRuns);
			printf("%16llu  %16.12f  %16.12f\n", totalRuns, estimate, fabs(estimate - kPi));
		}
		printf("\n");
	}

	// --- 원서 Listing estpi-3: 일반 샘플 vs 층화(지터링) 샘플 ---
	// 원서는 sqrt_N = 1000(백만 샘플) 한 가지만 본다. GPU라 격자를 1억 칸까지
	// 키워서, 층화 쪽 오차가 훨씬 빠르게 줄어드는 것을 함께 확인한다.
	{
		printf("[pi v3] regular vs stratified (one sample per grid cell)\n");
		printf("%8s  %14s  %16s  %16s  %16s  %16s\n",
			"sqrt_N", "N", "Regular", "|error|", "Stratified", "|error|");

		const int sizes[] = { 100, 1000, 10000 };
		for (int s = 0; s < 3; s++)
		{
			int sqrtN = sizes[s];
			checkDemoErrors(cudaMemset(dCounts, 0, 2 * sizeof(unsigned long long)));

			dim3 threads(16, 16);
			dim3 blocks((sqrtN + threads.x - 1) / threads.x, (sqrtN + threads.y - 1) / threads.y);
			PiStratifiedKernel<<<blocks, threads>>>(7, sqrtN, &dCounts[0], &dCounts[1]);
			checkDemoErrors(cudaGetLastError());

			unsigned long long h[2];
			checkDemoErrors(cudaMemcpy(h, dCounts, sizeof(h), cudaMemcpyDeviceToHost));

			double n = double(sqrtN) * double(sqrtN);
			double regular = 4.0 * double(h[0]) / n;
			double stratified = 4.0 * double(h[1]) / n;
			printf("%8d  %14.0f  %16.12f  %16.12f  %16.12f  %16.12f\n",
				sqrtN, n, regular, fabs(regular - kPi), stratified, fabs(stratified - kPi));
		}
		printf("\n(first 12 decimal places of pi: 3.141592653589)\n");
	}

	checkDemoErrors(cudaFree(dCounts));
}

// ─────────────────────────────────────────────────────────────────────
// 3장: 1차원 몬테카를로 적분
// ─────────────────────────────────────────────────────────────────────

// double용 atomicAdd는 sm_60 이상에서만 쓸 수 있다. 이 프로젝트는 기본 설정
// (compute_52)으로 빌드하므로, 스레드별 부분합을 배열에 써 두고 DeviceSum
// (thrust::reduce, GPU 병렬 리덕션)으로 합친다. thrust 호출은 DemoThrust.cu에
// 모아 두었다(무거운 thrust 헤더를 그 파일에서만 컴파일).

// 요청한 샘플 수 n을 (스레드 수) x (스레드당 샘플 수)로 나눈다. n은 2의 거듭제곱이거나
// 2^20 이하라고 가정한다(데모에서 쓰는 값만 넣는다).
static void SplitWork(long long n, int& numThreads, int& samplesPerThread)
{
	numThreads = int(n < (1LL << 20) ? n : (1LL << 20));
	samplesPerThread = int(n / numThreads);
}

// 적분할 함수 목록 (원서 Listing integ-xsq-1 / integ-sin5 / integ-ln-sin)
enum IntegrandId
{
	kIntegrandXSquared = 0,   // x^2
	kIntegrandSin5 = 1,       // sin^5(x)
	kIntegrandLnSin = 2       // ln(sin(x))
};

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
		// curand의 (0,1]을 그대로 써서 x 는 (a, b] 범위. x = 0 에서 ln(sin 0) = -inf 가
		// 나오는 일을 원천적으로 막는다(끝점 하나는 적분값에 영향이 없다).
		double x = a + (b - a) * curand_uniform_double(&rng);
		sum += EvalIntegrand(which, x);
	}
	partialSums[id] = sum;
}

static double IntegrateUniform(int which, double a, double b, long long n, unsigned long long seed)
{
	int numThreads, samplesPerThread;
	SplitWork(n, numThreads, samplesPerThread);

	double* dPartial;
	checkDemoErrors(cudaMalloc((void**)&dPartial, numThreads * sizeof(double)));

	int threadsPerBlock = 256;
	UniformIntegrateKernel<<<(numThreads + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
		seed, which, a, b, numThreads, samplesPerThread, dPartial);
	checkDemoErrors(cudaGetLastError());

	double total = DeviceSum(dPartial, numThreads);
	checkDemoErrors(cudaFree(dPartial));

	return (b - a) * total / (double(numThreads) * double(samplesPerThread));
}

static void DemoIntegrate()
{
	const long long N = 1000000;
	const double a = 0.0;
	const double b = 2.0;

	// 기준값(호스트)
	//   x^2      : 8/3
	//   sin^5(x) : 부정적분 -cos x + (2/3)cos^3 x - (1/5)cos^5 x
	//   ln(sin x): 초등함수 부정적분이 없다 → 중점 규칙(1천만 칸)으로 계산
	double reference[3];
	reference[0] = 8.0 / 3.0;
	{
		auto F = [](double x)
		{
			double c = cos(x);
			return -c + (2.0 / 3.0) * c * c * c - (1.0 / 5.0) * c * c * c * c * c;
		};
		reference[1] = F(b) - F(a);
	}
	{
		const int M = 10000000;
		double h = (b - a) / M;
		double s = 0.0;
		for (int k = 0; k < M; k++)
			s += log(sin(a + (k + 0.5) * h));
		reference[2] = s * h;
	}

	const char* names[3] = { "x^2", "sin^5(x)", "ln(sin(x))" };
	printf("[integrate] uniform Monte Carlo on [%g, %g], N = %lld\n", a, b, N);
	printf("%-12s  %16s  %16s  %16s\n", "f(x)", "Monte Carlo", "reference", "|error|");
	for (int w = 0; w < 3; w++)
	{
		double mc = IntegrateUniform(w, a, b, N, 42 + w);
		printf("%-12s  %16.12f  %16.12f  %16.12f\n", names[w], mc, reference[w], fabs(mc - reference[w]));
	}
}

// --- 원서 Listing est-halfway: p(x) = exp(-x/2π) sin²(x)의 "넓이 절반" 지점 ---
__host__ __device__ inline double ExpSin2(double x)
{
	double s = sin(x);
	return exp(-x / (2.0 * kPi)) * s * s;
}

__global__ void ExpSin2SampleKernel(unsigned long long seed, int n, double* xs, double* pxs)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= n) return;

	DemoRng rng;
	curand_init(seed, id, 0, &rng);

	double x = RandomDouble(&rng, 0.0, 2.0 * kPi);
	xs[id] = x;
	pxs[id] = ExpSin2(x);
}

// 원서는 샘플을 배열에 모아 std::sort 후 앞에서부터 더해 가며 절반을 찾는다.
// GPU에서는 같은 일을 병렬 기본 연산 세 개로 나눈다:
//   1) thrust::sort_by_key   : x 기준 정렬(p(x)를 값으로 같이 끌고 간다)
//   2) thrust::inclusive_scan: 누적합(prefix sum)  px[i] = p(x_0) + ... + p(x_i)
//   3) thrust::lower_bound   : 누적합이 처음으로 절반 이상이 되는 위치(이진 탐색)
static void EstimateHalfway(int n, unsigned long long seed)
{
	double* dXs;
	double* dPx;
	checkDemoErrors(cudaMalloc((void**)&dXs, n * sizeof(double)));
	checkDemoErrors(cudaMalloc((void**)&dPx, n * sizeof(double)));

	int threadsPerBlock = 256;
	ExpSin2SampleKernel<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
		seed, n, dXs, dPx);
	checkDemoErrors(cudaGetLastError());

	double sum = DeviceSum(dPx, n);
	double halfway = DeviceHalfwayPoint(dXs, dPx, n, sum / 2.0);   // 정렬 → 누적합 → 이진 탐색

	checkDemoErrors(cudaFree(dXs));
	checkDemoErrors(cudaFree(dPx));

	printf("N = %d\n", n);
	printf("  Average          = %.12f\n", sum / n);
	printf("  Area under curve = %.12f\n", 2.0 * kPi * sum / n);
	printf("  Halfway          = %.12f\n", halfway);
}

static void DemoHalfway()
{
	printf("[halfway] 50%% point of p(x) = exp(-x/2pi) sin^2(x) on [0, 2pi]\n");

	// 호스트 기준값: 중점 규칙으로 넓이를 구하고, 누적 넓이가 절반을 넘는 x를 찾는다.
	{
		const int M = 2000000;
		double h = 2.0 * kPi / M;
		double total = 0.0;
		for (int k = 0; k < M; k++)
			total += ExpSin2((k + 0.5) * h) * h;

		double accum = 0.0;
		double halfway = 0.0;
		for (int k = 0; k < M; k++)
		{
			accum += ExpSin2((k + 0.5) * h) * h;
			if (accum >= total / 2.0)
			{
				halfway = (k + 1) * h;
				break;
			}
		}
		printf("reference (midpoint rule, %d cells)\n", M);
		printf("  Area under curve = %.12f\n", total);
		printf("  Halfway          = %.12f\n", halfway);
	}

	EstimateHalfway(10000, 11);        // 원서와 같은 샘플 수
	EstimateHalfway(10000000, 12);     // GPU라서 1000배
}

// --- 원서 Importance Sampling 절: integral(0..2) x^2 dx 를 여러 PDF로 ---
enum PdfId
{
	kPdfUniform = 0,     // p(x) = 1/2            ICD(d) = 2d
	kPdfHalfSplit = 1,   // 원서 Listing crude-approx: [0,sqrt2]와 [sqrt2,2]에 절반씩
	kPdfLinear = 2,      // p(x) = x/2            ICD(d) = sqrt(4d)
	kPdfQuadratic = 3    // p(x) = (3/8) x^2      ICD(d) = (8d)^(1/3) = 2 d^(1/3)
};

// 균일 난수 → 원하는 분포를 따르는 x.
// curand_uniform_double은 (0,1]을 돌려주므로 d = 0 이 절대 나오지 않는다. 그래서
// 원서의 "if (z == 0.0) continue;"(x = 0 에서 pdf = 0 으로 나누는 것 방지)가 필요 없다.
__device__ inline double SampleX(int pdf, DemoRng* rng)
{
	double d = curand_uniform_double(rng);
	switch (pdf)
	{
	case kPdfUniform:
		return 2.0 * d;
	case kPdfHalfSplit:
	{
		// 절반 확률로 아래쪽 [0,sqrt2], 절반 확률로 위쪽 [sqrt2,2]에서 균일하게.
		double d2 = curand_uniform_double(rng);
		return (d <= 0.5) ? sqrt(2.0) * d2 : sqrt(2.0) + (2.0 - sqrt(2.0)) * d2;
	}
	case kPdfLinear:
		return sqrt(4.0 * d);
	default:
		// 원서 본문은 ICD(d) = 8 d^(1/3)로 적었지만, P(x) = x^3/8 의 역함수는
		// (8d)^(1/3) = 2 d^(1/3) 이다. (8 d^(1/3)는 x가 [0,8]로 튄다. 다만 이 경우
		//  f/p = 8/3 이 상수라서 틀린 식으로도 답은 똑같이 나온다.)
		return 2.0 * cbrt(d);
	}
}

__device__ inline double PdfValue(int pdf, double x)
{
	switch (pdf)
	{
	case kPdfUniform:
		return 0.5;
	case kPdfHalfSplit:
		// 각 구간에 확률 1/2이 균일하게 퍼져 있다 → 밀도 = 0.5 / 구간 길이.
		// (원서는 이 분포를 "뽑는" 코드만 보이고 적분에는 쓰지 않는다. 여기서는
		//  밀도까지 적어 균일/선형 PDF 사이의 수렴 속도를 직접 비교해 본다.)
		return (x < sqrt(2.0)) ? 0.5 / sqrt(2.0) : 0.5 / (2.0 - sqrt(2.0));
	case kPdfLinear:
		return x / 2.0;
	default:
		return (3.0 / 8.0) * x * x;
	}
}

// 샘플 하나의 추정값 g = f(x)/p(x). 그 합과 제곱합을 스레드별로 모은다
// (제곱합으로 샘플 하나당 표준편차 = 노이즈의 크기를 잰다).
__global__ void ImportanceIntegrateKernel(
	unsigned long long seed, int pdf, int numThreads, int samplesPerThread,
	double* partialSums, double* partialSumSquares)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numThreads) return;

	DemoRng rng;
	curand_init(seed, id, 0, &rng);

	double sum = 0.0;
	double sumSq = 0.0;
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

static void IntegrateImportance(int pdf, long long n, unsigned long long seed, double& estimate, double& stddev)
{
	int numThreads, samplesPerThread;
	SplitWork(n, numThreads, samplesPerThread);

	double* dSums;
	double* dSumSquares;
	checkDemoErrors(cudaMalloc((void**)&dSums, numThreads * sizeof(double)));
	checkDemoErrors(cudaMalloc((void**)&dSumSquares, numThreads * sizeof(double)));

	int threadsPerBlock = 256;
	ImportanceIntegrateKernel<<<(numThreads + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
		seed, pdf, numThreads, samplesPerThread, dSums, dSumSquares);
	checkDemoErrors(cudaGetLastError());

	double count = double(numThreads) * double(samplesPerThread);
	double mean = DeviceSum(dSums, numThreads) / count;
	double meanSq = DeviceSum(dSumSquares, numThreads) / count;

	checkDemoErrors(cudaFree(dSums));
	checkDemoErrors(cudaFree(dSumSquares));

	estimate = mean;                                   // 원서: sum / N (구간 길이 곱 없음)
	stddev = sqrt(fmax(0.0, meanSq - mean * mean));    // 샘플 하나당 표준편차
}

static void DemoImportance()
{
	const double exact = 8.0 / 3.0;
	const char* names[4] = { "uniform", "half-split", "linear", "quadratic" };
	const long long counts[4] = { 1, 100, 10000, 1048576 };

	printf("[importance] I = integral of x^2 on [0,2] = 8/3 = %.12f\n", exact);
	printf("estimate (|error|) by sample count N\n");
	printf("%-11s", "pdf");
	for (int c = 0; c < 4; c++)
		printf("  %24s", (c == 0) ? "N=1" : (c == 1) ? "N=100" : (c == 2) ? "N=10000" : "N=1048576");
	printf("  %14s\n", "stddev/sample");

	for (int p = 0; p < 4; p++)
	{
		printf("%-11s", names[p]);
		double stddev = 0.0;
		for (int c = 0; c < 4; c++)
		{
			double estimate;
			IntegrateImportance(p, counts[c], 100 * p + c, estimate, stddev);
			printf("  %11.8f (%10.8f)", estimate, fabs(estimate - exact));
		}
		printf("  %14.10f\n", stddev);   // 가장 큰 N에서 잰 값
	}
}

// ─────────────────────────────────────────────────────────────────────
// 4장: 방향(단위 구) 위의 몬테카를로 적분
// ─────────────────────────────────────────────────────────────────────

// 거절법(rejection method)으로 균일한 무작위 방향을 만든다(1·2권 방식).
// [-1,1]^3 정육면체에서 점을 뽑아 단위 공 안에 들어오면 정규화해 단위 구 위의 점,
// 즉 방향으로 쓴다. 공 밖이면 다시 뽑는다. 받아들여질 확률은 (공 부피)/(정육면체 부피)
// = (4/3 pi)/8 = pi/6 (약 52%)이라, 평균 6/pi (약 1.91)번 시도한다.
// tries에 이번 샘플의 시도 횟수를 돌려준다(워프 발산 측정용).
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

// 단위 구 전체에서 cos^2(theta) 적분: f(d) = d_z^2, p(d) = 1/(4 pi).
// 거절법의 while 루프는 스레드마다 반복 횟수가 다르다. 워프(32스레드)는 가장 오래
// 도는 스레드를 기다려야 하므로, 실제 비용은 "평균 시도 횟수"가 아니라 "워프 안 최대
// 시도 횟수"다. __shfl_xor_sync로 워프 최대값을 구해 그 비용도 같이 잰다.
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
			double cosineSquared = d.Z() * d.Z();   // theta는 z축과의 각 → cos(theta) = d_z
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

static void DemoSphere()
{
	const int numThreads = 1 << 20;
	const int samplesPerThread = 16;

	double* dSums;
	double* dTries;
	double* dWarpTries;
	checkDemoErrors(cudaMalloc((void**)&dSums, numThreads * sizeof(double)));
	checkDemoErrors(cudaMalloc((void**)&dTries, numThreads * sizeof(double)));
	checkDemoErrors(cudaMalloc((void**)&dWarpTries, numThreads * sizeof(double)));

	int threadsPerBlock = 256;
	SphereIntegrateKernel<<<(numThreads + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
		2024, numThreads, samplesPerThread, dSums, dTries, dWarpTries);
	checkDemoErrors(cudaGetLastError());

	double n = double(numThreads) * double(samplesPerThread);
	double estimate = DeviceSum(dSums, numThreads) / n;
	double meanTries = DeviceSum(dTries, numThreads) / n;
	double meanWarpTries = DeviceSum(dWarpTries, numThreads) / n;

	checkDemoErrors(cudaFree(dSums));
	checkDemoErrors(cudaFree(dTries));
	checkDemoErrors(cudaFree(dWarpTries));

	double exact = 4.0 * kPi / 3.0;
	printf("[sphere] integral of cos^2(theta) over the unit sphere, N = %.0f\n", n);
	printf("  Estimate          = %.12f\n", estimate);
	printf("  Exact (4/3 pi)    = %.12f\n", exact);
	printf("  |error|           = %.12f\n", fabs(estimate - exact));
	printf("rejection-method cost per sample\n");
	printf("  mean tries per thread      = %.4f   (theory 6/pi = %.4f)\n", meanTries, 6.0 / kPi);
	printf("  mean tries the warp waited = %.4f   (max over the 32 lanes)\n", meanWarpTries);
	printf("  lane utilization           = %.1f%%\n", 100.0 * meanTries / meanWarpTries);
}

// ─────────────────────────────────────────────────────────────────────
// 5장: 빛의 산란 — Lambertian 산란 PDF 확인
// ─────────────────────────────────────────────────────────────────────
// 원서 5장은 코드 없이 수식만 세운다. 여기서는 그 수식을 숫자로 확인한다.
//   (1) 정규화: 반구에서 pScatter = cos(theta)/pi 를 적분하면 1 이어야 한다.
//   (2) 산란 방향 생성기가 정말 cos(theta)/pi 분포를 따르는지 cos(theta) 히스토그램으로 본다.
//       - inBall  : normal + (단위 공 "안"의 무작위 점)  <- 지금 우리 렌더러의 Lambertian
//       - onSphere: normal + (단위 구 "위"의 무작위 점) <- 원서 v4의 Lambertian
//       - hemi    : 반구 균일 방향
//   법선은 +z로 고정한다(그러면 cos(theta) = 방향의 z성분).

constexpr int kCosBins = 10;

enum ScatterGenId
{
	kGenInBall = 0,
	kGenOnSphere = 1,
	kGenHemisphere = 2
};

// 단위 공 안의 무작위 점(거절법). 렌더러의 RandomInUnitSphere와 같은 방식.
__device__ inline Vector3 RandomInUnitBall(DemoRng* rng)
{
	while (true)
	{
		Vector3 p(RandomDouble(rng, -1.0, 1.0), RandomDouble(rng, -1.0, 1.0), RandomDouble(rng, -1.0, 1.0));
		if (p.LengthSquared() < 1.0)
			return p;
	}
}

__device__ inline Vector3 ScatterDirection(int generator, DemoRng* rng)
{
	const Vector3 normal(0.0, 0.0, 1.0);
	int tries;
	Vector3 d;

	if (generator == kGenInBall)
	{
		d = normal + RandomInUnitBall(rng);
	}
	else if (generator == kGenOnSphere)
	{
		d = normal + RandomUnitVectorRejection(rng, tries);
	}
	else
	{
		d = RandomUnitVectorRejection(rng, tries);
		if (d.Z() < 0.0)
			d = -d;    // 아래 반구면 뒤집어 위 반구로
	}

	// 무작위 점이 정확히 -normal이면 영벡터가 된다 → 법선으로 대체(렌더러와 동일)
	if (d.NearZero())
		d = normal;

	return UnitVector(d);
}

// (1) 반구 정규화: 구 전체에서 균일하게 뽑아 f = max(0, cos)/pi 를 p = 1/(4 pi)로 나눈 평균 → 1
__global__ void LambertNormalizationKernel(
	unsigned long long seed, int numThreads, int samplesPerThread, double* partialSums)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numThreads) return;

	DemoRng rng;
	curand_init(seed, id, 0, &rng);

	double sum = 0.0;
	for (int k = 0; k < samplesPerThread; k++)
	{
		int tries;
		double c = RandomUnitVectorRejection(&rng, tries).Z();
		double pScatter = (c > 0.0) ? c / kPi : 0.0;   // 수평선 아래로는 산란하지 않는다
		sum += pScatter * (4.0 * kPi);                 // f / p,  p = 1/(4 pi)
	}
	partialSums[id] = sum;
}

// (2) cos(theta) 히스토그램.
// 블록 전용 히스토그램(공유 메모리)에 먼저 모은 뒤, 블록마다 한 번씩 전역에 더한다.
// 모든 스레드가 전역 bins[10]에 직접 atomicAdd하면 주소 10개에 경합이 몰린다
// (공유 메모리 "사유화(privatization)" — CUDA 히스토그램의 기본 패턴).
__global__ void CosineHistogramKernel(
	unsigned long long seed, int generator, int numThreads, int samplesPerThread,
	unsigned long long* bins, double* partialCosSums)
{
	__shared__ unsigned int localBins[kCosBins];
	for (int b = threadIdx.x; b < kCosBins; b += blockDim.x)
		localBins[b] = 0;
	__syncthreads();

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numThreads)
	{
		DemoRng rng;
		curand_init(seed, id, 0, &rng);

		double cosSum = 0.0;
		for (int k = 0; k < samplesPerThread; k++)
		{
			double c = ScatterDirection(generator, &rng).Z();   // 법선이 +z라 cos(theta) = z
			cosSum += c;

			int bin = int(c * kCosBins);
			if (bin < 0) bin = 0;
			if (bin >= kCosBins) bin = kCosBins - 1;
			atomicAdd(&localBins[bin], 1u);
		}
		partialCosSums[id] = cosSum;
	}
	__syncthreads();

	for (int b = threadIdx.x; b < kCosBins; b += blockDim.x)
		atomicAdd(&bins[b], (unsigned long long)localBins[b]);
}

static void DemoLambert()
{
	const int numThreads = 1 << 20;
	const int samplesPerThread = 8;
	const double n = double(numThreads) * double(samplesPerThread);
	const int threadsPerBlock = 256;
	const int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

	double* dPartial;
	checkDemoErrors(cudaMalloc((void**)&dPartial, numThreads * sizeof(double)));

	// (1) 정규화
	LambertNormalizationKernel<<<blocks, threadsPerBlock>>>(77, numThreads, samplesPerThread, dPartial);
	checkDemoErrors(cudaGetLastError());
	double integral = DeviceSum(dPartial, numThreads) / n;

	printf("[lambert] (1) integral of pScatter = cos(theta)/pi over the hemisphere, N = %.0f\n", n);
	printf("  Estimate = %.9f   (should be 1)\n\n", integral);

	// (2) 생성기별 cos(theta) 분포
	unsigned long long* dBins;
	checkDemoErrors(cudaMalloc((void**)&dBins, kCosBins * sizeof(unsigned long long)));

	unsigned long long hist[3][kCosBins];
	double meanCos[3];
	for (int g = 0; g < 3; g++)
	{
		checkDemoErrors(cudaMemset(dBins, 0, kCosBins * sizeof(unsigned long long)));
		CosineHistogramKernel<<<blocks, threadsPerBlock>>>(500 + g, g, numThreads, samplesPerThread, dBins, dPartial);
		checkDemoErrors(cudaGetLastError());
		checkDemoErrors(cudaMemcpy(hist[g], dBins, sizeof(hist[g]), cudaMemcpyDeviceToHost));
		meanCos[g] = DeviceSum(dPartial, numThreads) / n;
	}

	checkDemoErrors(cudaFree(dBins));
	checkDemoErrors(cudaFree(dPartial));

	// 이상적인 분포의 칸별 확률 (mu = cos(theta), 방위각은 적분해 없앰)
	//   cos/pi   : p(mu) = 2 mu     → 칸 확률 mu1^2 - mu0^2,  평균 cos = 2/3
	//   cos^3    : p(mu) = 4 mu^3   → 칸 확률 mu1^4 - mu0^4,  평균 cos = 4/5
	//   uniform  : p(mu) = 1        → 칸 확률 0.1,             평균 cos = 1/2
	printf("[lambert] (2) distribution of cos(theta) for three scatter-direction generators (normal = +z)\n");
	printf("%-11s  %9s %9s %9s  | %9s %9s %9s\n",
		"cos(theta)", "inBall", "onSphere", "hemi", "cos/pi", "cos^3", "uniform");
	for (int b = 0; b < kCosBins; b++)
	{
		double mu0 = double(b) / kCosBins;
		double mu1 = double(b + 1) / kCosBins;
		printf("[%.1f, %.1f)  %9.4f %9.4f %9.4f  | %9.4f %9.4f %9.4f\n",
			mu0, mu1,
			hist[0][b] / n, hist[1][b] / n, hist[2][b] / n,
			mu1 * mu1 - mu0 * mu0,
			mu1 * mu1 * mu1 * mu1 - mu0 * mu0 * mu0 * mu0,
			1.0 / kCosBins);
	}
	printf("%-11s  %9.4f %9.4f %9.4f  | %9.4f %9.4f %9.4f\n",
		"mean cos", meanCos[0], meanCos[1], meanCos[2], 2.0 / 3.0, 4.0 / 5.0, 0.5);
}

// ─────────────────────────────────────────────────────────────────────
// 6장: 중요도 샘플링 가지고 놀기 — 화로 테스트(white furnace test)
// ─────────────────────────────────────────────────────────────────────
// 사방에서 밝기 1의 빛이 들어오는 "화로" 안에 알베도 A인 표면을 두면, 에너지 보존상
// 반사되어 나오는 빛은 정확히 A여야 한다. 산란 한 번의 추정값
//   g = A * pScatter(w) * L / pdf(w),   w ~ (방향을 실제로 뽑은 분포)
// 의 평균이 A인지 본다. pdf에는 반드시 "방향을 실제로 뽑은 분포의 밀도"를 넣어야 한다.
// 뽑는 분포는 그대로 두고 pdf 값만 바꾸면(원서 6장 균일 PDF 절의 코드 그대로) 편향된다.
enum FurnaceCaseId
{
	kFurnaceCosCos = 0,        // cos 분포로 뽑고 pdf = cos/pi            → A
	kFurnaceUniformUniform,    // 반구 균일로 뽑고 pdf = 1/2pi            → A (노이즈 증가)
	kFurnaceCosUniform,        // cos 분포로 뽑고 pdf = 1/2pi (어긋남)    → 4A/3 (편향)
	kFurnaceHemiMaterial,      // pScatter = pdf = 1/2pi (다른 재질)      → A
	kFurnaceCaseCount
};

__global__ void FurnaceKernel(
	unsigned long long seed, int caseId, double albedo, int numThreads, int samplesPerThread,
	double* partialSums, double* partialSumSquares)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numThreads) return;

	DemoRng rng;
	curand_init(seed, id, 0, &rng);

	const double uniformPdf = 1.0 / (2.0 * kPi);
	bool bCosSampled = (caseId == kFurnaceCosCos || caseId == kFurnaceCosUniform);

	double sum = 0.0;
	double sumSq = 0.0;
	for (int k = 0; k < samplesPerThread; k++)
	{
		Vector3 w = ScatterDirection(bCosSampled ? kGenOnSphere : kGenHemisphere, &rng);
		double cosTheta = w.Z();

		double pScatter = (caseId == kFurnaceHemiMaterial) ? uniformPdf : cosTheta / kPi;
		double pdf = (caseId == kFurnaceCosCos) ? cosTheta / kPi : uniformPdf;

		double g = (pdf > 0.0) ? albedo * pScatter * 1.0 / pdf : 0.0;   // 들어오는 빛 L = 1
		sum += g;
		sumSq += g * g;
	}
	partialSums[id] = sum;
	partialSumSquares[id] = sumSq;
}

static void DemoFurnace()
{
	const double albedo = 0.73;          // 코넬 박스 흰 벽과 같은 값
	const int numThreads = 1 << 20;
	const int samplesPerThread = 8;
	const double n = double(numThreads) * double(samplesPerThread);
	const int threadsPerBlock = 256;

	const char* names[kFurnaceCaseCount] =
	{
		"cos-sampled,     pdf = cos/pi",
		"uniform-sampled, pdf = 1/2pi",
		"cos-sampled,     pdf = 1/2pi  (book listing)",
		"hemispherical material, pScatter = pdf = 1/2pi"
	};
	const double expected[kFurnaceCaseCount] = { albedo, albedo, albedo * 4.0 / 3.0, albedo };

	double* dSums;
	double* dSumSquares;
	checkDemoErrors(cudaMalloc((void**)&dSums, numThreads * sizeof(double)));
	checkDemoErrors(cudaMalloc((void**)&dSumSquares, numThreads * sizeof(double)));

	printf("[furnace] white furnace test: albedo A = %.2f, incoming radiance L = 1 from every direction\n", albedo);
	printf("%-48s  %10s  %10s  %14s\n", "case", "estimate", "expected", "stddev/sample");
	for (int c = 0; c < kFurnaceCaseCount; c++)
	{
		FurnaceKernel<<<(numThreads + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
			900 + c, c, albedo, numThreads, samplesPerThread, dSums, dSumSquares);
		checkDemoErrors(cudaGetLastError());

		double mean = DeviceSum(dSums, numThreads) / n;
		double meanSq = DeviceSum(dSumSquares, numThreads) / n;
		double stddev = sqrt(fmax(0.0, meanSq - mean * mean));
		printf("%-48s  %10.6f  %10.6f  %14.6f\n", names[c], mean, expected[c], stddev);
	}

	checkDemoErrors(cudaFree(dSums));
	checkDemoErrors(cudaFree(dSumSquares));
}

// ─────────────────────────────────────────────────────────────────────
// 진입점
// ─────────────────────────────────────────────────────────────────────

bool RunMonteCarloDemo(const char* name)
{
	if (strcmp(name, "pi") == 0)
	{
		DemoPi();
		return true;
	}
	if (strcmp(name, "integrate") == 0)
	{
		DemoIntegrate();
		return true;
	}
	if (strcmp(name, "halfway") == 0)
	{
		DemoHalfway();
		return true;
	}
	if (strcmp(name, "importance") == 0)
	{
		DemoImportance();
		return true;
	}
	if (strcmp(name, "sphere") == 0)
	{
		DemoSphere();
		return true;
	}
	if (strcmp(name, "lambert") == 0)
	{
		DemoLambert();
		return true;
	}
	if (strcmp(name, "furnace") == 0)
	{
		DemoFurnace();
		return true;
	}

	return false;
}

void PrintMonteCarloDemoList()
{
	fprintf(stderr, "Available demos (--demo <name>):\n");
	fprintf(stderr, "  pi         ch.2  estimate pi: one-shot / convergence / stratified\n");
	fprintf(stderr, "  integrate  ch.3  uniform MC integration of x^2, sin^5, ln(sin)\n");
	fprintf(stderr, "  halfway    ch.3  50%% point of a PDF via GPU sort + prefix sum\n");
	fprintf(stderr, "  importance ch.3  integrate x^2 with uniform/half-split/linear/quadratic PDFs\n");
	fprintf(stderr, "  sphere     ch.4  integrate cos^2 over the unit sphere (rejection-sampled directions)\n");
	fprintf(stderr, "  lambert    ch.5  Lambertian scattering PDF: normalization + cos(theta) histograms\n");
	fprintf(stderr, "  furnace    ch.6  white furnace test of the f/p estimator for several sampling choices\n");
}
