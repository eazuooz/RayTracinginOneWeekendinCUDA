
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RtWeekend.h"

#include "Hittable.h"
#include "HittableList.h"
#include "Sphere.h"
#include "Camera.h"
#include "Metal.h"


// GPU에서 벡터 덧셈을 수행하는 헬퍼 함수 선언
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

// GPU에서 실행되는 커널 함수 - 각 스레드가 배열의 한 원소씩 덧셈 수행
__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;  // 현재 스레드의 인덱스를 배열 인덱스로 사용
	c[i] = a[i] + b[i];
}

Color RayColor(const Ray& ray, const Hittable& world)
{
	HitRecord hitRecord;
	if (world.Hit(ray, Interval(0.0, Infinity), hitRecord))
	{
		return 0.5 * (hitRecord.Normal + Color(1.0, 1.0, 1.0));
	}

	Vector3 unitDirection = UnitVector(ray.Direction());
	auto a = 0.5 * (unitDirection.Y() + 1.0);

	return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color	(0.5, 0.7, 1.0);
}

int main()
{
#pragma region Ray Tracing in One Weekend - CUDA setup test
	// === CUDA 기본 벡터 덧셈 예제 ===
	// CUDA 환경이 정상적으로 동작하는지 확인하기 위한 테스트 코드

	std::cout << "CUDA Vector Addition Example" << std::endl;

	// 호스트(CPU) 측 입력 배열 a, b와 결과 배열 c 선언
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// GPU를 이용하여 두 벡터를 병렬로 덧셈 (a + b = c)
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// 결과 출력: {11, 22, 33, 44, 55}가 나오면 정상
	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// 프로그램 종료 전 GPU 디바이스 리셋
	// Nsight, Visual Profiler 등 프로파일링 도구가 완전한 트레이스를 표시하려면 필요
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
#pragma endregion


#pragma region Ray Tracing in One Weekend 1st Use Not CUDA
	// === Ray Tracing in One Weekend - 첫 번째 장 구현 테스트 코드 ===
	//HittableList world;

	//auto groundMaterial = std::make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
	//world.Add(std::make_shared<Sphere>(Point3(0.0, -1000.0, 0.0), 1000.0, groundMaterial));

	//for (int gridX = -11; gridX < 11; gridX++)
	//{
	//	for (int gridZ = -11; gridZ < 11; gridZ++)
	//	{
	//		const double materialSelector = RandomDouble();

	//		const Point3 center(
	//			gridX + 0.9 * RandomDouble(),
	//			0.2,
	//			gridZ + 0.9 * RandomDouble()
	//		);

	//		if ((center - Point3(4.0, 0.2, 0.0)).Length() > 0.9)
	//		{
	//			std::shared_ptr<Material> sphereMaterial;

	//			if (materialSelector < 0.8)
	//			{
	//				const Color albedo = Color::Random() * Color::Random();
	//				sphereMaterial = std::make_shared<Lambertian>(albedo);
	//			}
	//			else if (materialSelector < 0.95)
	//			{
	//				const Color albedo = Color::Random(0.5, 1.0);
	//				const double fuzz = RandomDouble(0.0, 0.5);
	//				sphereMaterial = std::make_shared<Metal>(albedo, fuzz);
	//			}
	//			else
	//			{
	//				sphereMaterial = std::make_shared<Dielectric>(1.5);
	//			}

	//			world.Add(std::make_shared<Sphere>(center, 0.2, sphereMaterial));
	//		}
	//	}
	//}

	//auto material1 = std::make_shared<Dielectric>(1.5);
	//world.Add(std::make_shared<Sphere>(Point3(0.0, 1.0, 0.0), 1.0, material1));

	//auto material2 = std::make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
	//world.Add(std::make_shared<Sphere>(Point3(-4.0, 1.0, 0.0), 1.0, material2));

	//auto material3 = std::make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
	//world.Add(std::make_shared<Sphere>(Point3(4.0, 1.0, 0.0), 1.0, material3));

	//Camera camera;

	//camera.aspectRatio = 16.0 / 9.0;
	//camera.imageWidth = 1200;
	//camera.samplesPerPixel = 500;
	//camera.maxDepth = 50;

	//camera.vfov = 20.0;
	//camera.lookfrom = Point3(13.0, 2.0, 3.0);
	//camera.lookat = Point3(0.0, 0.0, 0.0);
	//camera.vup = Vec3(0.0, 1.0, 0.0);

	//camera.defocusAngle = 0.6;
	//camera.focusDistance = 10.0;

	//camera.Render(world);
#pragma endregion
	return 0;
}

// CUDA를 사용하여 벡터 덧셈을 수행하는 헬퍼 함수
// CUDA 프로그래밍의 기본 흐름을 보여주는 예제:
//   1. GPU 디바이스 선택
//   2. GPU 메모리 할당 (cudaMalloc)
//   3. 호스트 → 디바이스 메모리 복사 (cudaMemcpy)
//   4. 커널 실행 (addKernel<<<...>>>)
//   5. 디바이스 → 호스트 메모리 복사 (cudaMemcpy)
//   6. GPU 메모리 해제 (cudaFree)
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	// 디바이스(GPU) 측 포인터 선언
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// 사용할 GPU 선택 (멀티 GPU 시스템에서는 번호 변경)
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// GPU 메모리 할당 - 입력 2개(dev_a, dev_b) + 출력 1개(dev_c)
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// 호스트(CPU) 메모리 → 디바이스(GPU) 메모리로 입력 데이터 복사
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// 커널 실행: <<<1, size>>> = 1개 블록, size개 스레드
	// 각 스레드가 배열의 한 원소씩 병렬로 덧셈 수행
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// 커널 실행 시 에러 확인
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// GPU 커널이 완료될 때까지 대기 (동기화)
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// 디바이스(GPU) → 호스트(CPU)로 결과 복사
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	// GPU 메모리 해제
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
