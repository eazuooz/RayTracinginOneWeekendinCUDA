
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <ctime>
#include <fstream>

#include "Vec3.h"

// CUDA 에러 체크 매크로 - 에러 발생 시 파일명, 라인 번호와 함께 에러 메시지 출력
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

// === Chapter 2: Vec3 클래스를 사용한 이미지 렌더링 CUDA 커널 ===
// 각 스레드가 하나의 픽셀을 담당하여 Vector3(Color)로 색상을 계산
__global__ void render(Vector3* frameBuffer, int maxX, int maxY)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX + i;
	frameBuffer[pixelIndex] = Color(
		double(i) / double(maxX),
		double(j) / double(maxY),
		0.2
	);
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

	// GPU Unified Memory로 프레임버퍼 할당 (Vector3 배열)
	Vector3* frameBuffer;
	checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));

	clock_t start, stop;
	start = clock();

	dim3 blocks(imageWidth / blockWidth + 1, imageHeight / blockHeight + 1);
	dim3 threads(blockWidth, blockHeight);
	render<<<blocks, threads>>>(frameBuffer, imageWidth, imageHeight);
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
