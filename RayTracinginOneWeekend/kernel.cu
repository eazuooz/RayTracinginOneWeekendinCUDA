
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

// === Chapter 10: 자유 시점 카메라 (Positionable Camera) ===
//
// lookfrom/lookat/vup으로 카메라 위치와 방향을 자유롭게 설정한다.
// vfov(수직 시야각)와 aspect(종횡비)로 뷰포트 크기를 결정한다.
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

// cuRAND 초기화 커널
__global__ void RenderInit(int maxX, int maxY, curandState* randState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX + i;
	curand_init(1984, pixelIndex, 0, &randState[pixelIndex]);
}

// 렌더 커널: 안티앨리어싱 + 재질 기반 산란
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
		Ray r = (*camera)->GetRay(u, v);
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

// GPU에서 월드 오브젝트, 재질, 카메라를 생성
__global__ void CreateWorld(Hittable** list, Hittable** world, Camera** camera, int imageWidth, int imageHeight)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		// 바닥: 연한 노란색 Lambertian
		list[0] = new Sphere(
			Vector3(0.0, -100.5, -1.0), 100.0,
			new Lambertian(Color(0.8, 0.8, 0.0)));

		// 중앙: 파란 계열 난반사 구체
		list[1] = new Sphere(
			Vector3(0.0, 0.0, -1.0), 0.5,
			new Lambertian(Color(0.1, 0.2, 0.5)));

		// 오른쪽: 금속 구체 (선명한 반사)
		list[2] = new Sphere(
			Vector3(1.0, 0.0, -1.0), 0.5,
			new Metal(Color(0.8, 0.6, 0.2), 0.0));

		// 왼쪽: 유리 구체 (굴절률 1.5)
		list[3] = new Sphere(
			Vector3(-1.0, 0.0, -1.0), 0.5,
			new Dielectric(1.5));

		// 왼쪽 내부: 음수 반지름으로 속이 빈 유리 구체 (버블 효과)
		list[4] = new Sphere(
			Vector3(-1.0, 0.0, -1.0), -0.45,
			new Dielectric(1.5));

		*world = new HittableList(list, 5);

		// 자유 시점 카메라: (-2,2,1)에서 (0,0,-1)을 바라봄, vfov=20도
		*camera = new Camera(
			Vector3(-2.0, 2.0, 1.0),   // lookfrom
			Vector3(0.0, 0.0, -1.0),   // lookat
			Vector3(0.0, 1.0, 0.0),    // vup
			20.0,                        // vfov (degrees)
			double(imageWidth) / double(imageHeight));  // aspect ratio
	}
}

// GPU 오브젝트 해제 커널
__global__ void FreeWorld(Hittable** list, Hittable** world, Camera** camera)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < 5; i++)
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
	int numSamples = 30;

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

	curandState* randState;
	checkCudaErrors(cudaMalloc((void**)&randState, numPixels * sizeof(curandState)));

	// 월드 + 카메라를 GPU 메모리에 생성 (구체 5개)
	Hittable** list;
	checkCudaErrors(cudaMalloc((void**)&list, 5 * sizeof(Hittable*)));
	Hittable** world;
	checkCudaErrors(cudaMalloc((void**)&world, sizeof(Hittable*)));
	Camera** camera;
	checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Camera*)));

	CreateWorld<<<1, 1>>>(list, world, camera, imageWidth, imageHeight);
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
	FreeWorld<<<1, 1>>>(list, world, camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(randState));
	checkCudaErrors(cudaFree(list));
	checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(camera));
	checkCudaErrors(cudaFree(frameBuffer));

	return 0;
}
