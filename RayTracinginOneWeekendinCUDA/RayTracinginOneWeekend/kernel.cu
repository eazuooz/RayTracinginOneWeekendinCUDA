
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RtWeekend.h"

#include "Hittable.h"
#include "HittableList.h"
#include "Sphere.h"
#include "Camera.h"
#include "Metal.h"


cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
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

	HittableList world;

	auto R = std::cos(Pi / 4);

	//auto materialLeft = std::make_shared<Lambertian>(Color(0, 0, 1));
	//auto materialRight = std::make_shared<Lambertian>(Color(1, 0, 0));

	//world.Add(std::make_shared<Sphere>(Point3(-R, 0, -1), R, materialLeft));
	//world.Add(std::make_shared<Sphere>(Point3(R, 0, -1), R, materialRight));

	auto materialGround = std::make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
	auto materialCenter = std::make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
	auto materialLeft = std::make_shared<Dielectric>(1.50);
	auto materialBubble = std::make_shared<Dielectric>(1.0 / 1.50);
	auto materialRight = std::make_shared<Metal>(Color(0.8, 0.6, 0.2), 1.0);

	world.Add(std::make_shared<Sphere>(Point3(0.0, -100.5, -1.0), 100.0, materialGround));
	world.Add(std::make_shared<Sphere>(Point3(0.0, 0.0, -1.2), 0.5, materialCenter));
	world.Add(std::make_shared<Sphere>(Point3(-1.0, 0.0, -1.0), 0.5, materialLeft));
	world.Add(std::make_shared<Sphere>(Point3(-1.0, 0.0, -1.0), 0.4, materialBubble));
	world.Add(std::make_shared<Sphere>(Point3(1.0, 0.0, -1.0), 0.5, materialRight));

	Camera camera;

	camera.aspectRatio = 16.0 / 9.0;
	camera.imageWidth = 400;
	camera.samplesPerPixel = 100;
	camera.maxDepth = 50;
	camera.vfov = 20;
	camera.lookfrom = Point3(-2, 2, 1);
	camera.lookat = Point3(0, 0, -1);
	camera.vup = Vec3(0, 1, 0);

	camera.defocusAngle = 10.0;
	camera.focusDistance = 3.4;

	camera.Render(world);

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
