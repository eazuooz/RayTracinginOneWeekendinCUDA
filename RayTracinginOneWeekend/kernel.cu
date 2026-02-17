
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <ctime>
#include <fstream>

#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"
#include "HittableList.h"
#include "Sphere.h"

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

// === Chapter 5: 법선 벡터 시각화 ===
// 오브젝트에 맞으면 법선 벡터를 색상으로 변환하여 표시
// 법선 (-1~1) → 색상 (0~1) 매핑: color = 0.5 * (normal + 1)
__device__ Color RayColor(const Ray& r, Hittable** world)
{
	HitRecord rec;
	if ((*world)->Hit(r, 0.0, DBL_MAX, rec))
	{
		// 법선 벡터를 RGB 색상으로 변환
		// X(좌우) → R, Y(상하) → G, Z(앞뒤) → B
		return 0.5 * Color(rec.Normal.X() + 1.0,
		                    rec.Normal.Y() + 1.0,
		                    rec.Normal.Z() + 1.0);
	}

	// 배경: 하늘 그라디언트
	Vector3 unitDirection = UnitVector(r.Direction());
	double t = 0.5 * (unitDirection.Y() + 1.0);
	return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}

__global__ void render(Vector3* frameBuffer, int maxX, int maxY,
	Vector3 lowerLeftCorner, Vector3 horizontal, Vector3 vertical, Vector3 origin,
	Hittable** world)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX + i;
	double u = double(i) / double(maxX);
	double v = double(j) / double(maxY);
	Ray r(origin, lowerLeftCorner + u * horizontal + v * vertical);
	frameBuffer[pixelIndex] = RayColor(r, world);
}

// GPU 메모리에서 월드 오브젝트를 생성하는 커널
// CUDA에서 가상 함수(virtual)를 사용하려면 객체를 GPU에서 new로 생성해야 함
__global__ void createWorld(Hittable** list, Hittable** world)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		// 구체 2개: 작은 구체(중앙) + 큰 구체(바닥)
		list[0] = new Sphere(Vector3(0.0, 0.0, -1.0), 0.5);
		list[1] = new Sphere(Vector3(0.0, -100.5, -1.0), 100.0);
		*world = new HittableList(list, 2);
	}
}

// GPU에서 할당한 월드 오브젝트를 해제하는 커널
__global__ void freeWorld(Hittable** list, Hittable** world)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		delete list[0];
		delete list[1];
		delete *world;
	}
}

int main()
{
	int imageWidth = 1440;
	int imageHeight = 720;

	int blockWidth = 8;
	int blockHeight = 8;

	std::cerr << "Rendering a " << imageWidth << "x" << imageHeight << " image "
		<< "in " << blockWidth << "x" << blockHeight << " blocks.\n";

	int numPixels = imageWidth * imageHeight;
	size_t frameBufferSize = numPixels * sizeof(Vector3);

	Vector3* frameBuffer;
	checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));

	// 월드 오브젝트를 GPU 메모리에 할당
	Hittable** list;
	checkCudaErrors(cudaMalloc((void**)&list, 2 * sizeof(Hittable*)));
	Hittable** world;
	checkCudaErrors(cudaMalloc((void**)&world, sizeof(Hittable*)));

	createWorld<<<1, 1>>>(list, world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();

	dim3 blocks(imageWidth / blockWidth + 1, imageHeight / blockHeight + 1);
	dim3 threads(blockWidth, blockHeight);
	render<<<blocks, threads>>>(frameBuffer, imageWidth, imageHeight,
		Vector3(-2.0, -1.0, -1.0),   // lowerLeftCorner
		Vector3(4.0, 0.0, 0.0),      // horizontal
		Vector3(0.0, 2.0, 0.0),      // vertical
		Vector3(0.0, 0.0, 0.0),      // origin
		world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timerSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timerSeconds << " seconds.\n";

	// 프레임버퍼를 PPM 이미지 파일로 저장
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

	// GPU 메모리 해제 (역순)
	freeWorld<<<1, 1>>>(list, world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(list));
	checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(frameBuffer));

	return 0;
}
