
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "Vec3.h"
#include "Color.h"
#include "Ray.h"

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

bool HitSphere(const Point3& center, double radius, const Ray& r)
{
	Vec3 oc = center - r.Origin();
	auto a = Dot(r.Direction(), r.Direction());
	auto b = -2.0 * Dot(r.Direction(), oc);
	auto c = Dot(oc, oc) - radius * radius;
	auto discriminant = b * b - 4 * a * c;

	return (discriminant >= 0);
}

Color RayColor(const Ray& r)
{
    if (HitSphere(Point3(0,0,-1), 0.5, r))
        return Color(1, 0, 0);

	Vec3 unitDirection = UnitVector(r.Direction());
	auto a = 0.5 * (unitDirection.Y() + 1.0);

	return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color	(0.5, 0.7, 1.0);
}

int main()
{
#pragma region Ray Tracing in One Weekend - CUDA setup test
//std::cout << "CUDA Vector Addition Example" << std::endl;
//const int arraySize = 5;
//const int a[arraySize] = { 1, 2, 3, 4, 5 };
//const int b[arraySize] = { 10, 20, 30, 40, 50 };
//int c[arraySize] = { 0 };

//// Add vectors in parallel.
//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//if (cudaStatus != cudaSuccess) 
//{
//	fprintf(stderr, "addWithCuda failed!");
//	return 1;
//}

//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//	c[0], c[1], c[2], c[3], c[4]);

//// cudaDeviceReset must be called before exiting in order for profiling and
//// tracing tools such as Nsight and Visual Profiler to show complete traces.
//cudaStatus = cudaDeviceReset();
//if (cudaStatus != cudaSuccess) 
//{
//	fprintf(stderr, "cudaDeviceReset failed!");
//	return 1;
//}  
#pragma endregion

	// Image
	auto aspectRatio = 16.0 / 9.0;
	int imageWidth = 400;

	// 이미지 높이를 계산하고 최소 1이 되도록 합니다.
	int imageHeight = int(imageWidth / aspectRatio);
	imageHeight = (imageHeight < 1) ? 1 : imageHeight;

	// Camera
	auto focalLength = 1.0;
	auto viewportHeight = 2.0;
	auto viewportWidth = viewportHeight * (double(imageWidth) / imageHeight);
	auto cameraCenter = Point3(0, 0, 0);

	// 뷰포트의 수평 및 수직 가장자리를 가로지르는 벡터를 계산합니다.
	auto viewportU = Vec3(viewportWidth, 0, 0);
	auto viewportV = Vec3(0, -viewportHeight, 0);

	// 픽셀 간 수평 및 수직 델타 벡터를 계산합니다.
	auto pixelDeltaU = viewportU / imageWidth;
	auto pixelDeltaV = viewportV / imageHeight;

	// 왼쪽 위 픽셀의 위치를 계산합니다.
	auto viewportUpperLeft = cameraCenter - Vec3(0, 0, focalLength) - viewportU / 2 - viewportV / 2;
	auto pixel00Loc = viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);

	// Render
	std::cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

	for (int j = 0; j < imageHeight; j++)
	{
		std::clog << "\rScanlines remaining: " << (imageHeight - j) << ' ' << std::flush;
		for (int i = 0; i < imageWidth; i++)
		{
			auto pixelCenter = pixel00Loc + (i * pixelDeltaU) + (j * pixelDeltaV);
			auto rayDirection = pixelCenter - cameraCenter;
			Ray r(cameraCenter, rayDirection);

			Color pixelColor = RayColor(r);
			WriteColor(std::cout, pixelColor);
		}
	}

	std::clog << "\rDone.                 \n";

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
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

	// Copy input vectors from host memory to GPU buffers.
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

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
