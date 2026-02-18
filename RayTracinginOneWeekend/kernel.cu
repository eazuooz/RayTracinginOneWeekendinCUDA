
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <ctime>
#include <cfloat>
#include <fstream>
#include <curand_kernel.h>

#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"
#include "HittableList.h"
#include "Sphere.h"
#include "Material.h"
#include "Metal.h"
#include "Dielectric.h"
#include "Camera.h"

// CUDA 에러 체크 매크로
#define checkCudaErrors(val) CheckCuda((val), #val, __FILE__, __LINE__)

void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
			<< file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

// === Chapter 12: 최종 렌더 (Final Render) ===
//
// "Ray Tracing in One Weekend"의 표지 장면을 GPU에서 렌더링한다.
// 22×22 격자에 랜덤 소형 구체(Lambertian/Metal/Dielectric)를 배치하고,
// 중앙에 유리/난반사/금속 대형 구체 3개를 놓는다.
// 총 구체 수: 22*22 + 1(바닥) + 3(대형) = 최대 488개
//
// GPU에서 재귀 대신 루프(최대 50회)로 레이를 추적한다.
// 매 반복마다 재질의 Scatter 함수로 감쇠 색상과 새 레이를 얻는다.
__device__ Color RayColor(const Ray& r, Hittable** world, curandState* randState)
{
	Ray currentRay = r;
	Color currentAttenuation(1.0, 1.0, 1.0);

	for (int i = 0; i < 50; i++)
	{
		HitRecord rec;
		if ((*world)->Hit(currentRay, 0.001, DBL_MAX, rec))
		{
			Ray scattered;
			Color attenuation;
			if (rec.MaterialPtr->Scatter(currentRay, rec, attenuation, scattered, randState))
			{
				currentAttenuation = currentAttenuation * attenuation;
				currentRay = scattered;
			}
			else
			{
				return Color(0.0, 0.0, 0.0);
			}
		}
		else
		{
			Vector3 unitDirection = UnitVector(currentRay.Direction());
			double t = 0.5 * (unitDirection.Y() + 1.0);
			Color c = (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
			return currentAttenuation * c;
		}
	}

	return Color(0.0, 0.0, 0.0);
}

// 월드 생성용 cuRAND 초기화 (단일 스레드)
__global__ void RandInit(curandState* randState)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curand_init(1984, 0, 0, randState);
	}
}

// 렌더링용 cuRAND 초기화 (픽셀당 1개)
__global__ void RenderInit(int maxX, int maxY, curandState* randState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX + i;
	curand_init(1984, pixelIndex, 0, &randState[pixelIndex]);
}

// 렌더 커널: 안티앨리어싱 + 재질 기반 산란 + 피사계 심도
__global__ void Render(
	Vector3* frameBuffer, int maxX, int maxY, int numSamples,
	Camera** camera, Hittable** world, curandState* randState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX + i;
	curandState localRandState = randState[pixelIndex];
	Color col(0.0, 0.0, 0.0);

	for (int s = 0; s < numSamples; s++)
	{
		double u = double(i + curand_uniform(&localRandState)) / double(maxX);
		double v = double(j + curand_uniform(&localRandState)) / double(maxY);
		Ray r = (*camera)->GetRay(u, v, &localRandState);
		col += RayColor(r, world, &localRandState);
	}

	randState[pixelIndex] = localRandState;
	col = col / double(numSamples);

	// 감마 보정 (gamma = 2.0)
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	frameBuffer[pixelIndex] = col;
}

// curand_uniform 단축 매크로 (CreateWorld 내부에서 사용)
#define RND (curand_uniform(&localRandState))

// GPU에서 랜덤 장면을 생성하는 커널
// 22×22 격자에 소형 구체를 랜덤 배치하고, 대형 구체 3개와 바닥을 추가한다.
// 구체 배치 시 (4, 0.2, 0) 근처는 대형 구체와 겹치지 않도록 건너뛴다.
__global__ void CreateWorld(
	Hittable** list, Hittable** world, Camera** camera,
	int imageWidth, int imageHeight, curandState* randState)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState localRandState = *randState;

		// [0] 바닥: 거대한 회색 Lambertian 구체
		list[0] = new Sphere(
			Vector3(0.0, -1000.0, -1.0), 1000.0,
			new Lambertian(Color(0.5, 0.5, 0.5)));

		// [1~] 22×22 격자에 소형 구체를 랜덤 배치
		int i = 1;
		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
				double chooseMat = RND;
				Vector3 center(a + RND, 0.2, b + RND);

				if (chooseMat < 0.8)
				{
					// 80%: Lambertian (랜덤 색상의 난반사)
					list[i++] = new Sphere(
						center, 0.2,
						new Lambertian(Color(RND * RND, RND * RND, RND * RND)));
				}
				else if (chooseMat < 0.95)
				{
					// 15%: Metal (밝은 랜덤 색상, 랜덤 fuzz)
					list[i++] = new Sphere(
						center, 0.2,
						new Metal(
							Color(0.5 * (1.0 + RND), 0.5 * (1.0 + RND), 0.5 * (1.0 + RND)),
							0.5 * RND));
				}
				else
				{
					// 5%: Dielectric (유리)
					list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
				}
			}
		}

		// 대형 구체 3개: 유리, Lambertian, Metal
		list[i++] = new Sphere(Vector3(0.0, 1.0, 0.0), 1.0, new Dielectric(1.5));
		list[i++] = new Sphere(Vector3(-4.0, 1.0, 0.0), 1.0, new Lambertian(Color(0.4, 0.2, 0.1)));
		list[i++] = new Sphere(Vector3(4.0, 1.0, 0.0), 1.0, new Metal(Color(0.7, 0.6, 0.5), 0.0));

		*randState = localRandState;
		*world = new HittableList(list, 22 * 22 + 1 + 3);

		// 카메라: (13,2,3)에서 원점을 바라봄, 얕은 피사계 심도
		Vector3 lookfrom(13.0, 2.0, 3.0);
		Vector3 lookat(0.0, 0.0, 0.0);
		double distToFocus = 10.0;
		double aperture = 0.1;

		*camera = new Camera(
			lookfrom,
			lookat,
			Vector3(0.0, 1.0, 0.0),
			30.0,
			double(imageWidth) / double(imageHeight),
			aperture,
			distToFocus);
	}
}

#undef RND

// GPU 오브젝트 해제 커널
__global__ void FreeWorld(Hittable** list, int numHittables, Hittable** world, Camera** camera)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < numHittables; i++)
		{
			delete list[i];
		}
		delete *world;
		delete *camera;
	}
}

int main()
{
	int imageWidth = 1440;
	int imageHeight = 720;
	int numSamples = 10;

	int blockWidth = 8;
	int blockHeight = 8;

	// GPU 스택 크기 증가 (가상 함수 + 루프에서 필요)
	checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

	std::cerr << "Rendering a " << imageWidth << "x" << imageHeight
		<< " image with " << numSamples << " samples per pixel "
		<< "in " << blockWidth << "x" << blockHeight << " blocks.\n";

	int numPixels = imageWidth * imageHeight;
	size_t frameBufferSize = numPixels * sizeof(Vector3);

	Vector3* frameBuffer;
	checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));

	// 렌더링용 cuRAND 상태 (픽셀당 1개)
	curandState* randState;
	checkCudaErrors(cudaMalloc((void**)&randState, numPixels * sizeof(curandState)));

	// 월드 생성용 cuRAND 상태 (단일)
	curandState* randState2;
	checkCudaErrors(cudaMalloc((void**)&randState2, sizeof(curandState)));

	RandInit<<<1, 1>>>(randState2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// 월드 + 카메라를 GPU 메모리에 생성
	// 최대 구체 수: 22*22(소형) + 1(바닥) + 3(대형) = 488
	int numHittables = 22 * 22 + 1 + 3;
	Hittable** list;
	checkCudaErrors(cudaMalloc((void**)&list, numHittables * sizeof(Hittable*)));
	Hittable** world;
	checkCudaErrors(cudaMalloc((void**)&world, sizeof(Hittable*)));
	Camera** camera;
	checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Camera*)));

	CreateWorld<<<1, 1>>>(list, world, camera, imageWidth, imageHeight, randState2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();

	dim3 blocks(imageWidth / blockWidth + 1, imageHeight / blockHeight + 1);
	dim3 threads(blockWidth, blockHeight);

	RenderInit<<<blocks, threads>>>(imageWidth, imageHeight, randState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	Render<<<blocks, threads>>>(
		frameBuffer, imageWidth, imageHeight,
		numSamples, camera, world, randState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timerSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timerSeconds << " seconds.\n";

	// PPM 이미지 파일 저장
	std::ofstream outFile("output.ppm");
	outFile << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

	for (int j = imageHeight - 1; j >= 0; j--)
	{
		std::cerr << "\rWriting scanline " << (imageHeight - 1 - j)
			<< " / " << imageHeight << std::flush;

		for (int i = 0; i < imageWidth; i++)
		{
			size_t pixelIndex = j * imageWidth + i;
			Color col = frameBuffer[pixelIndex];

			int ir = int(255.99 * col.X());
			int ig = int(255.99 * col.Y());
			int ib = int(255.99 * col.Z());

			outFile << ir << " " << ig << " " << ib << "\n";
		}
	}
	outFile.close();
	std::cerr << "\nDone. Saved to output.ppm\n";

	// GPU 메모리 해제
	FreeWorld<<<1, 1>>>(list, numHittables, world, camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(randState));
	checkCudaErrors(cudaFree(randState2));
	checkCudaErrors(cudaFree(list));
	checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(camera));
	checkCudaErrors(cudaFree(frameBuffer));

	return 0;
}
