
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
#include "Camera.h"

// CUDA 에러 체크 매크로
#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
			<< file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

// === Chapter 6: 안티앨리어싱 ===
// 법선 벡터를 색상으로 변환 (이전과 동일)
__device__ Color RayColor(const Ray& r, Hittable** world)
{
	HitRecord rec;
	if ((*world)->Hit(r, 0.001, DBL_MAX, rec))
	{
		return 0.5 * Color(rec.Normal.X() + 1.0,
		                    rec.Normal.Y() + 1.0,
		                    rec.Normal.Z() + 1.0);
	}

	Vector3 unitDirection = UnitVector(r.Direction());
	double t = 0.5 * (unitDirection.Y() + 1.0);
	return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}

// 각 픽셀마다 cuRAND 난수 생성기를 초기화하는 커널
// 모든 스레드가 동일한 시드(1984)를 사용하되, 픽셀 인덱스로 시퀀스를 구분
__global__ void renderInit(int maxX, int maxY, curandState* randState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX + i;
	// curand_init(시드, 시퀀스 번호, 오프셋, 상태)
	curand_init(1984, pixelIndex, 0, &randState[pixelIndex]);
}

// 안티앨리어싱 렌더 커널
// 각 픽셀에서 numSamples 만큼 랜덤 오프셋을 적용하여 레이를 생성하고 평균
__global__ void render(Vector3* frameBuffer, int maxX, int maxY, int numSamples,
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
		// curand_uniform: (0, 1] 범위의 랜덤 값으로 픽셀 내 위치를 흔든다
		double u = double(i + curand_uniform(&localRandState)) / double(maxX);
		double v = double(j + curand_uniform(&localRandState)) / double(maxY);
		Ray r = (*camera)->GetRay(u, v);
		col += RayColor(r, world);
	}

	frameBuffer[pixelIndex] = col / double(numSamples);
}

// GPU에서 월드 오브젝트와 카메라를 생성하는 커널
__global__ void createWorld(Hittable** list, Hittable** world, Camera** camera)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		list[0] = new Sphere(Vector3(0.0, 0.0, -1.0), 0.5);
		list[1] = new Sphere(Vector3(0.0, -100.5, -1.0), 100.0);
		*world = new HittableList(list, 2);
		*camera = new Camera();
	}
}

// GPU 오브젝트 해제 커널
__global__ void freeWorld(Hittable** list, Hittable** world, Camera** camera)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		delete list[0];
		delete list[1];
		delete *world;
		delete *camera;
	}
}

int main()
{
	int imageWidth = 1440;
	int imageHeight = 720;
	int numSamples = 100;

	int blockWidth = 8;
	int blockHeight = 8;

	std::cerr << "Rendering a " << imageWidth << "x" << imageHeight
		<< " image with " << numSamples << " samples per pixel "
		<< "in " << blockWidth << "x" << blockHeight << " blocks.\n";

	int numPixels = imageWidth * imageHeight;
	size_t frameBufferSize = numPixels * sizeof(Vector3);

	Vector3* frameBuffer;
	checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));

	// cuRAND 상태 배열 할당 (픽셀당 1개)
	curandState* randState;
	checkCudaErrors(cudaMalloc((void**)&randState, numPixels * sizeof(curandState)));

	// 월드 + 카메라를 GPU 메모리에 생성
	Hittable** list;
	checkCudaErrors(cudaMalloc((void**)&list, 2 * sizeof(Hittable*)));
	Hittable** world;
	checkCudaErrors(cudaMalloc((void**)&world, sizeof(Hittable*)));
	Camera** camera;
	checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Camera*)));

	createWorld<<<1, 1>>>(list, world, camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();

	dim3 blocks(imageWidth / blockWidth + 1, imageHeight / blockHeight + 1);
	dim3 threads(blockWidth, blockHeight);

	// 1단계: cuRAND 초기화
	renderInit<<<blocks, threads>>>(imageWidth, imageHeight, randState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// 2단계: 안티앨리어싱 렌더링
	render<<<blocks, threads>>>(frameBuffer, imageWidth, imageHeight,
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
		std::cerr << "\rWriting scanline " << (imageHeight - 1 - j) << " / " << imageHeight << std::flush;
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
	freeWorld<<<1, 1>>>(list, world, camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(randState));
	checkCudaErrors(cudaFree(list));
	checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(camera));
	checkCudaErrors(cudaFree(frameBuffer));

	return 0;
}
