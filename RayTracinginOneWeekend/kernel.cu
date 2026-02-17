
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <ctime>
#include <fstream>

#include "Vec3.h"
#include "Ray.h"

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

// === Chapter 3: Ray 클래스를 사용한 하늘 배경 렌더링 ===
// 레이 방향의 y 성분에 따라 흰색 → 하늘색 그라디언트를 생성
__device__ Color RayColor(const Ray& r)
{
	Vector3 unitDirection = UnitVector(r.Direction());
	double t = 0.5 * (unitDirection.Y() + 1.0);
	return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}

__global__ void render(Vector3* frameBuffer, int maxX, int maxY,
	Vector3 lowerLeftCorner, Vector3 horizontal, Vector3 vertical, Vector3 origin)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX + i;
	double u = double(i) / double(maxX);
	double v = double(j) / double(maxY);
	Ray r(origin, lowerLeftCorner + u * horizontal + v * vertical);
	frameBuffer[pixelIndex] = RayColor(r);
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

	clock_t start, stop;
	start = clock();

	dim3 blocks(imageWidth / blockWidth + 1, imageHeight / blockHeight + 1);
	dim3 threads(blockWidth, blockHeight);
	render<<<blocks, threads>>>(frameBuffer, imageWidth, imageHeight,
		Vector3(-2.0, -1.0, -1.0),   // lowerLeftCorner
		Vector3(4.0, 0.0, 0.0),      // horizontal
		Vector3(0.0, 2.0, 0.0),      // vertical
		Vector3(0.0, 0.0, 0.0));     // origin
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

	checkCudaErrors(cudaFree(frameBuffer));

	return 0;
}
