
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <ctime>
#include <fstream>

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

// === Chapter 1: 이미지 렌더링 CUDA 커널 ===
// 각 스레드가 하나의 픽셀을 담당하여 색상을 계산
// R = x좌표 비율, G = y좌표 비율, B = 0.2 고정값으로 그라디언트 이미지 생성
__global__ void render(float* frameBuffer, int maxX, int maxY)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// 이미지 범위를 벗어나는 스레드는 종료
	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX * 3 + i * 3;
	frameBuffer[pixelIndex + 0] = float(i) / float(maxX);
	frameBuffer[pixelIndex + 1] = float(j) / float(maxY);
	frameBuffer[pixelIndex + 2] = 0.2f;
}

int main()
{
	// 이미지 해상도 설정
	int imageWidth = 1440;
	int imageHeight = 720;

	// CUDA 스레드 블록 크기 (8x8 = 64 스레드/블록)
	int blockWidth = 8;
	int blockHeight = 8;

	std::cerr << "Rendering a " << imageWidth << "x" << imageHeight << " image "
		<< "in " << blockWidth << "x" << blockHeight << " blocks.\n";

	int numPixels = imageWidth * imageHeight;
	size_t frameBufferSize = 3 * numPixels * sizeof(float);

	// GPU Unified Memory로 프레임버퍼 할당
	// cudaMallocManaged: CPU와 GPU 모두에서 접근 가능한 메모리 할당
	float* frameBuffer;
	checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));

	// 렌더링 시간 측정 시작
	clock_t start, stop;
	start = clock();

	// 그리드/블록 구성 후 커널 실행
	// blocks: 이미지 전체를 덮도록 블록 수 계산 (올림 나눗셈)
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
			size_t pixelIndex = j * 3 * imageWidth + i * 3;
			float r = frameBuffer[pixelIndex + 0];
			float g = frameBuffer[pixelIndex + 1];
			float b = frameBuffer[pixelIndex + 2];

			int ir = int(255.99f * r);
			int ig = int(255.99f * g);
			int ib = int(255.99f * b);

			outFile << ir << " " << ig << " " << ib << "\n";
		}
	}
	outFile.close();
	std::cerr << "\nDone. Saved to output.ppm\n";

	// GPU 메모리 해제
	checkCudaErrors(cudaFree(frameBuffer));

	return 0;
}
