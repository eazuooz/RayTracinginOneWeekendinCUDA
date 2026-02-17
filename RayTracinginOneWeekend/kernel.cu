
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

// === Chapter 4: 구체 교차 판정 ===
//
// 레이-구체 교차 판정은 이차방정식의 판별식(discriminant)을 이용한다.
//
// 구체 방정식: (P - C) · (P - C) = r²
//   P: 구체 표면 위의 점, C: 구체 중심, r: 반지름
//
// 레이 방정식: P(t) = O + t·D
//   O: 레이 원점(Origin), D: 레이 방향(Direction), t: 매개변수
//
// 레이를 구체 방정식에 대입하면:
//   (O + t·D - C) · (O + t·D - C) = r²
//
// oc = O - C 로 치환하면:
//   (oc + t·D) · (oc + t·D) = r²
//   t²(D·D) + 2t(oc·D) + (oc·oc - r²) = 0
//
// 이것은 at² + bt + c = 0 형태의 이차방정식이다:
//   a = D · D           (레이 방향 벡터의 길이 제곱)
//   b = 2 * (oc · D)    (레이 원점~구체 중심 벡터와 방향의 내적 × 2)
//   c = oc · oc - r²    (원점~중심 거리 제곱 - 반지름 제곱)
//
// 판별식 = b² - 4ac
//   > 0: 레이가 구체와 두 점에서 교차 (관통)
//   = 0: 레이가 구체에 접함 (한 점에서 교차)
//   < 0: 교차하지 않음
//
__device__ bool HitSphere(const Vector3& center, double radius, const Ray& r)
{
	// oc: 레이 원점에서 구체 중심까지의 벡터
	Vector3 oc = r.Origin() - center;

	// 이차방정식 계수 계산
	double a = Dot(r.Direction(), r.Direction());
	double b = 2.0 * Dot(oc, r.Direction());
	double c = Dot(oc, oc) - radius * radius;

	// 판별식으로 교차 여부 판정
	double discriminant = b * b - 4.0 * a * c;
	return (discriminant > 0.0);
}

// 구체에 맞으면 빨간색, 아니면 하늘 배경 그라디언트
__device__ Color RayColor(const Ray& r)
{
	if (HitSphere(Vector3(0.0, 0.0, -1.0), 0.5, r))
		return Color(1.0, 0.0, 0.0);

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
