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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "MonteCarloDemo.h"

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

static const double kPi = 3.1415926535897932385;

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
// 진입점
// ─────────────────────────────────────────────────────────────────────

bool RunMonteCarloDemo(const char* name)
{
	if (strcmp(name, "pi") == 0)
	{
		DemoPi();
		return true;
	}

	return false;
}

void PrintMonteCarloDemoList()
{
	fprintf(stderr, "Available demos (--demo <name>):\n");
	fprintf(stderr, "  pi         ch.2  estimate pi: one-shot / convergence / stratified\n");
}
