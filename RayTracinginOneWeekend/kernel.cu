
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
#include "BvhNode.h"
#include "Sphere.h"
#include "MovingSphere.h"
#include "Quad.h"
#include "Texture.h"
#include "Material.h"
#include "Metal.h"
#include "Dielectric.h"
#include "Camera.h"
#include "RtwImage.h"

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

// === Chapter 12: 최종 렌더 (Final Render) + The Next Week: 모션 블러 ===
//
// "Ray Tracing in One Weekend"의 표지 장면을 GPU에서 렌더링한다.
// 22×22 격자에 랜덤 소형 구체(Lambertian/Metal/Dielectric)를 배치하고,
// 중앙에 유리/난반사/금속 대형 구체 3개를 놓는다.
// 총 구체 수: 22*22 + 1(바닥) + 3(대형) = 최대 488개
//
// 모션 블러: Lambertian 소형 구체들이 MovingSphere로 교체되어
// 셔터 개방 시간(time0=0, time1=1) 동안 위로 튀어오르는 운동을 한다.
//
// GPU에서 재귀 대신 루프(최대 50회)로 레이를 추적한다.
// 매 반복마다 재질의 Scatter 함수로 감쇠 색상과 새 레이를 얻는다.
// === The Next Week Chapter 7: 배경색 + 발광 재질 ===
// 원서의 재귀 ray_color를 반복(iterative)으로 푼 형태에 방출광을 더한다.
//
// 재귀식:  result = emit_0 + atten_0*(emit_1 + atten_1*(emit_2 + ...))
// 이를 누적변수 두 개로 펼친다:
//   throughput  : 지금까지 곱해진 감쇠(누적 반사율)
//   accumulated : 지금까지 모은 방출광의 합
//
// 더 이상 산란하지 않으면(빛이거나 흡수) 지금까지 모은 색을 반환하고,
// 아무것도 맞추지 못하면 배경색을 throughput에 실어 더한 뒤 반환한다.
// (background가 (0,0,0)이면 장면의 유일한 빛은 발광 재질뿐이다.)
__device__ Color RayColor(const Ray& r, const Color& background, Hittable** world, curandState* randState)
{
	Ray currentRay = r;
	Color throughput(1.0, 1.0, 1.0);
	Color accumulated(0.0, 0.0, 0.0);

	for (int i = 0; i < 50; i++)
	{
		HitRecord rec;
		if (!(*world)->Hit(currentRay, 0.001, DBL_MAX, rec))
		{
			// 아무것도 안 맞음 → 배경색
			accumulated += throughput * background;
			return accumulated;
		}

		// 방출광 누적(비발광 재질은 검정이라 영향 없음)
		Color emission = rec.MaterialPtr->Emitted(rec.U, rec.V, rec.P);
		accumulated += throughput * emission;

		Ray scattered;
		Color attenuation;
		if (!rec.MaterialPtr->Scatter(currentRay, rec, attenuation, scattered, randState))
		{
			// 산란 안 함(빛/흡수) → 지금까지 모은 색 반환
			return accumulated;
		}

		throughput = throughput * attenuation;
		currentRay = scattered;
	}

	return accumulated;
}

// 월드 생성용 cuRAND 초기화 (단일 스레드)
__global__ void RandInit(curandState* randState)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curand_init(1984, 0, 0, randState);
	}
}

// 렌더링용 cuRAND 초기화 (픽셀당 1개)
__global__ void RenderInit(int maxX, int maxY, curandState* randState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX + i;
	curand_init(1984, pixelIndex, 0, &randState[pixelIndex]);
}

// 렌더 커널: 안티앨리어싱 + 재질 기반 산란 + 피사계 심도
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

	// 장면 배경색(빛 장면은 검정). 카메라에 저장해 둔 값을 읽는다.
	Color background = (*camera)->Background();

	for (int s = 0; s < numSamples; s++)
	{
		double u = double(i + curand_uniform(&localRandState)) / double(maxX);
		double v = double(j + curand_uniform(&localRandState)) / double(maxY);
		Ray r = (*camera)->GetRay(u, v, &localRandState);
		col += RayColor(r, background, world, &localRandState);
	}

	randState[pixelIndex] = localRandState;
	col = col / double(numSamples);

	// 감마 보정 (gamma = 2.0)
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	frameBuffer[pixelIndex] = col;
}

// curand_uniform 단축 매크로 (CreateWorld 내부에서 사용)
#define RND (curand_uniform(&localRandState))

// GPU에서 장면을 생성하는 커널.
//
// === The Next Week Chapter 4: 텍스처 매핑 적용 ===
// 원서가 main()의 switch로 장면을 고르는 것을 본떠, sceneId로 분기한다.
//   0: bouncing_spheres  — 최종 랜덤 구 장면(바닥을 "체커 텍스처"로 교체)
//   1: checkered_spheres — 위아래로 놓인 체커 구 2개
//   2: earth             — 이미지 텍스처(지구 맵)를 입힌 구 1개
//   3: perlin_spheres    — 펄린 노이즈(대리석) 텍스처 구 2개
//   4: quads             — 5색 사각형(평행사변형) 장면
//   5: simple_light      — 펄린 구 + 사각형/구 광원 (배경 검정)
//   6: cornell_box       — 빈 코넬 박스 (천장 광원, 배경 검정)
//
// earthData/earthW/earthH: 호스트가 stb_image로 로드해 디바이스에 올린
// RGB 바이트 버퍼와 크기(scene 2에서만 사용). 로드 실패 시 nullptr → 청록색.
__global__ void CreateWorld(
	Hittable** list, Hittable** world, Camera** camera,
	int imageWidth, int imageHeight, curandState* randState, int* outCount,
	Hittable** bvhNodes, int* outNodeCount,
	int sceneId, const unsigned char* earthData, int earthW, int earthH)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState localRandState = *randState;

		int i = 0;

		// 장면별 카메라 파라미터 (아래 분기에서 채운다)
		Vector3 lookfrom(13.0, 2.0, 3.0);
		Vector3 lookat(0.0, 0.0, 0.0);
		double vfov = 20.0;
		double aperture = 0.0;
		double distToFocus = 10.0;
		double shutterOpen = 0.0;
		double shutterClose = 0.0;
		// 기본 배경은 푸르스름한 흰색(하늘). 빛 장면(5,6)은 검정으로 덮어쓴다.
		Color background(0.70, 0.80, 1.00);

		if (sceneId == 0)
		{
			// === bouncing_spheres: 바닥을 체커 텍스처로 (원서 Listing 26) ===
			// 단색 Lambertian 대신, 두 SolidColor를 번갈아 쓰는 CheckerTexture.
			Texture* checker = new CheckerTexture(
				0.32,
				new SolidColor(Color(0.2, 0.3, 0.1)),
				new SolidColor(Color(0.9, 0.9, 0.9)));
			list[i++] = new Sphere(
				Vector3(0.0, -1000.0, -1.0), 1000.0, new Lambertian(checker));

			// 22×22 격자에 소형 구체를 랜덤 배치
			for (int a = -11; a < 11; a++)
			{
				for (int b = -11; b < 11; b++)
				{
					double chooseMat = RND;
					Vector3 center(a + 0.9 * RND, 0.2, b + 0.9 * RND);

					// 대형 구체(4, 0.2, 0)와 겹치는 위치는 건너뜀
					Vector3 diff = center - Vector3(4.0, 0.2, 0.0);
					if (diff.Length() <= 0.9)
						continue;

					if (chooseMat < 0.8)
					{
						// 80%: Lambertian (랜덤 색상의 난반사) - 모션 블러 적용
						Vector3 center2 = center + Vector3(0.0, 0.5 * RND, 0.0);
						list[i++] = new MovingSphere(
							center, center2, 0.0, 1.0, 0.2,
							new Lambertian(Color(RND * RND, RND * RND, RND * RND)));
					}
					else if (chooseMat < 0.95)
					{
						// 15%: Metal (밝은 랜덤 색상, 랜덤 fuzz)
						list[i++] = new Sphere(
							center, 0.2,
							new Metal(
								Color(0.5 * (1.0 + RND), 0.5 * (1.0 + RND), 0.5 * (1.0 + RND)),
								0.5 * RND));
					}
					else
					{
						// 5%: Dielectric (유리)
						list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
					}
				}
			}

			// 대형 구체 3개: 유리, Lambertian, Metal
			list[i++] = new Sphere(Vector3(0.0, 1.0, 0.0), 1.0, new Dielectric(1.5));
			list[i++] = new Sphere(Vector3(-4.0, 1.0, 0.0), 1.0, new Lambertian(Color(0.4, 0.2, 0.1)));
			list[i++] = new Sphere(Vector3(4.0, 1.0, 0.0), 1.0, new Metal(Color(0.7, 0.6, 0.5), 0.0));

			lookfrom = Vector3(13.0, 2.0, 3.0);
			vfov = 30.0;
			aperture = 0.1;     // 얕은 피사계 심도
			shutterOpen = 0.0;  // 모션 블러 셔터 구간
			shutterClose = 1.0;
		}
		else if (sceneId == 1)
		{
			// === checkered_spheres: 체커 구 2개 (원서 Listing 28) ===
			// 두 Lambertian이 같은 CheckerTexture 포인터를 공유한다.
			Texture* checker = new CheckerTexture(
				0.32,
				new SolidColor(Color(0.2, 0.3, 0.1)),
				new SolidColor(Color(0.9, 0.9, 0.9)));

			list[i++] = new Sphere(Vector3(0.0, -10.0, 0.0), 10.0, new Lambertian(checker));
			list[i++] = new Sphere(Vector3(0.0, 10.0, 0.0), 10.0, new Lambertian(checker));

			lookfrom = Vector3(13.0, 2.0, 3.0);
			vfov = 20.0;
			aperture = 0.0;
		}
		else if (sceneId == 2)
		{
			// === earth: 이미지 텍스처를 입힌 구 (원서 Listing 33) ===
			// earthData는 호스트가 디바이스로 올린 RGB 버퍼. nullptr이면
			// ImageTexture::Value가 청록색을 반환한다(디버깅 표시).
			Texture* earthTex = new ImageTexture(earthData, earthW, earthH);
			list[i++] = new Sphere(Vector3(0.0, 0.0, 0.0), 2.0, new Lambertian(earthTex));

			lookfrom = Vector3(0.0, 0.0, 12.0);
			vfov = 20.0;
			aperture = 0.0;
		}
		else if (sceneId == 3)
		{
			// === perlin_spheres: 펄린 노이즈(대리석) 구 (원서 Listing 36/40/47) ===
			// 두 Lambertian이 같은 NoiseTexture를 공유한다. scale=4로 주파수를 올린다.
			// NoiseTexture 생성자가 localRandState로 격자 벡터/순열을 디바이스에서 만든다.
			Texture* pertext = new NoiseTexture(4.0, &localRandState);
			list[i++] = new Sphere(Vector3(0.0, -1000.0, 0.0), 1000.0, new Lambertian(pertext));
			list[i++] = new Sphere(Vector3(0.0, 2.0, 0.0), 2.0, new Lambertian(pertext));

			lookfrom = Vector3(13.0, 2.0, 3.0);
			vfov = 20.0;
			aperture = 0.0;
		}
		else if (sceneId == 4)
		{
			// === quads: 5색 사각형(평행사변형) 장면 (원서 Listing 54) ===
			// 두 번째 프리미티브 Quad를 시연. 각 면을 다른 색 Lambertian으로.
			list[i++] = new Quad(Vector3(-3, -2, 5), Vector3(0, 0, -4), Vector3(0, 4, 0),
				new Lambertian(Color(1.0, 0.2, 0.2)));   // left  (red)
			list[i++] = new Quad(Vector3(-2, -2, 0), Vector3(4, 0, 0), Vector3(0, 4, 0),
				new Lambertian(Color(0.2, 1.0, 0.2)));   // back  (green)
			list[i++] = new Quad(Vector3(3, -2, 1), Vector3(0, 0, 4), Vector3(0, 4, 0),
				new Lambertian(Color(0.2, 0.2, 1.0)));   // right (blue)
			list[i++] = new Quad(Vector3(-2, 3, 1), Vector3(4, 0, 0), Vector3(0, 0, 4),
				new Lambertian(Color(1.0, 0.5, 0.0)));   // upper (orange)
			list[i++] = new Quad(Vector3(-2, -3, 5), Vector3(4, 0, 0), Vector3(0, 0, -4),
				new Lambertian(Color(0.2, 0.8, 0.8)));   // lower (teal)

			// 원서는 정사각형(aspect 1.0)으로 렌더하지만, 우리 출력은 1440x720(2:1)
			// 이라 카메라 aspect도 그에 맞춰진다(가로로 조금 넓게 보임).
			lookfrom = Vector3(0.0, 0.0, 9.0);
			vfov = 80.0;
			aperture = 0.0;
		}
		else if (sceneId == 5)
		{
			// === simple_light: 펄린 구 + 사각형 광원 + 구 광원 (원서 Listing 59/60) ===
			// 배경이 검정이라, 장면의 유일한 빛은 발광 재질(DiffuseLight)뿐이다.
			Texture* pertext = new NoiseTexture(4.0, &localRandState);
			list[i++] = new Sphere(Vector3(0.0, -1000.0, 0.0), 1000.0, new Lambertian(pertext));
			list[i++] = new Sphere(Vector3(0.0, 2.0, 0.0), 2.0, new Lambertian(pertext));

			// (4,4,4): (1,1,1)보다 밝아야 주변을 비출 수 있다.
			Material* diffLight = new DiffuseLight(Color(4.0, 4.0, 4.0));
			list[i++] = new Sphere(Vector3(0.0, 7.0, 0.0), 2.0, diffLight);            // 구 광원
			list[i++] = new Quad(Vector3(3.0, 1.0, -2.0),
				Vector3(2.0, 0.0, 0.0), Vector3(0.0, 2.0, 0.0), diffLight);            // 사각형 광원

			background = Color(0.0, 0.0, 0.0);
			lookfrom = Vector3(26.0, 3.0, 6.0);
			lookat = Vector3(0.0, 2.0, 0.0);
			vfov = 20.0;
			aperture = 0.0;
		}
		else
		{
			// === cornell_box: 빈 코넬 박스 (원서 Listing 61) ===
			// 5개 벽 + 천장 광원. 확산 표면 간 빛 상호작용의 고전 장면.
			Material* red = new Lambertian(Color(0.65, 0.05, 0.05));
			Material* white = new Lambertian(Color(0.73, 0.73, 0.73));
			Material* green = new Lambertian(Color(0.12, 0.45, 0.15));
			Material* light = new DiffuseLight(Color(15.0, 15.0, 15.0));

			list[i++] = new Quad(Vector3(555, 0, 0), Vector3(0, 555, 0), Vector3(0, 0, 555), green);
			list[i++] = new Quad(Vector3(0, 0, 0), Vector3(0, 555, 0), Vector3(0, 0, 555), red);
			list[i++] = new Quad(Vector3(343, 554, 332), Vector3(-130, 0, 0), Vector3(0, 0, -105), light);
			list[i++] = new Quad(Vector3(0, 0, 0), Vector3(555, 0, 0), Vector3(0, 0, 555), white);
			list[i++] = new Quad(Vector3(555, 555, 555), Vector3(-555, 0, 0), Vector3(0, 0, -555), white);
			list[i++] = new Quad(Vector3(0, 0, 555), Vector3(555, 0, 0), Vector3(0, 555, 0), white);

			background = Color(0.0, 0.0, 0.0);
			lookfrom = Vector3(278.0, 278.0, -800.0);
			lookat = Vector3(278.0, 278.0, 0.0);
			vfov = 40.0;
			aperture = 0.0;
		}

		*randState = localRandState;
		*outCount = i;  // 실제 배치된 프리미티브 수 (FreeWorld에서 사용)

		// === BVH 빌드 (모든 장면 공통) ===
		// list[0..i)를 BVH로 묶어 레이-객체 교차를 로그 시간에 가깝게 만든다.
		int nodeCount = 0;
		BvhNode* root = new BvhNode(list, 0, i, bvhNodes, &nodeCount);
		bvhNodes[nodeCount++] = root;  // 루트도 해제 레지스트리에 등록
		*outNodeCount = nodeCount;
		*world = root;

		// === 카메라 (장면별 파라미터로 공통 생성) ===
		*camera = new Camera(
			lookfrom,
			lookat,
			Vector3(0.0, 1.0, 0.0),
			vfov,
			double(imageWidth) / double(imageHeight),
			aperture,
			distToFocus,
			shutterOpen,
			shutterClose,
			background);
	}
}

#undef RND

// GPU 오브젝트 해제 커널
__global__ void FreeWorld(
	Hittable** list, int numHittables,
	Hittable** bvhNodes, int numNodes,
	Hittable** world, Camera** camera)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		// 잎(primitive)들을 한 번씩 해제
		for (int i = 0; i < numHittables; i++)
		{
			delete list[i];
		}
		// 모든 BvhNode를 한 번씩 해제 (*world == 루트 노드도 이 배열에 포함되어
		// 있으므로 *world를 따로 delete하지 않는다 → 더블 프리 방지)
		for (int i = 0; i < numNodes; i++)
		{
			delete bvhNodes[i];
		}
		delete *camera;
	}
}

int main()
{
	int imageWidth = 1440;
	int imageHeight = 720;

	int blockWidth = 8;
	int blockHeight = 8;

	// === 렌더링할 장면 선택 (The Next Week Ch.4~7) ===
	//   0: bouncing_spheres  — 바닥이 체커 텍스처인 최종 랜덤 구 장면
	//   1: checkered_spheres — 체커 구 2개
	//   2: earth             — 지구 이미지 텍스처 구 (earthmap.jpg 필요)
	//   3: perlin_spheres    — 펄린 노이즈(대리석) 구 2개
	//   4: quads             — 5색 사각형(평행사변형) 장면
	//   5: simple_light      — 사각형/구 광원 (배경 검정)
	//   6: cornell_box       — 빈 코넬 박스 (천장 광원, 배경 검정)
	int sceneId = 6;

	// 픽셀당 샘플 수. 빛 장면(5,6)은 작은 광원 때문에 노이즈가 심하므로
	// 샘플을 크게 잡는다(원서도 100~200 사용).
	int numSamples = (sceneId == 5 || sceneId == 6) ? 200 : 10;

	// GPU 스택 크기 증가
	// MovingSphere 추가로 가상함수 깊이가 늘어 스택 소비 증가 → 32768로 확장.
	// BVH 순회는 재귀 대신 명시적 스택(BvhNode::Hit)을 쓰므로 추가 스택은
	// 필요 없다(재귀로 두면 이 한도로도 일부 스레드에서 스택이 넘쳤다).
	checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 32768));

	std::cerr << "Rendering a " << imageWidth << "x" << imageHeight
		<< " image with " << numSamples << " samples per pixel "
		<< "in " << blockWidth << "x" << blockHeight << " blocks.\n";

	int numPixels = imageWidth * imageHeight;
	size_t frameBufferSize = numPixels * sizeof(Vector3);

	Vector3* frameBuffer;
	checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));

	// 렌더링용 cuRAND 상태 (픽셀당 1개)
	curandState* randState;
	checkCudaErrors(cudaMalloc((void**)&randState, numPixels * sizeof(curandState)));

	// 월드 생성용 cuRAND 상태 (단일)
	curandState* randState2;
	checkCudaErrors(cudaMalloc((void**)&randState2, sizeof(curandState)));

	RandInit<<<1, 1>>>(randState2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// 월드 + 카메라를 GPU 메모리에 생성
	// 최대 구체 수: 22*22(소형) + 1(바닥) + 3(대형) = 488 (실제는 continue로 일부 제외)
	int maxHittables = 22 * 22 + 1 + 3;
	Hittable** list;
	checkCudaErrors(cudaMalloc((void**)&list, maxHittables * sizeof(Hittable*)));
	Hittable** world;
	checkCudaErrors(cudaMalloc((void**)&world, sizeof(Hittable*)));
	Camera** camera;
	checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Camera*)));

	// BVH 노드 레지스트리: n개 잎에 대한 BVH 내부 노드 수는 최대 2n 미만이므로
	// 넉넉하게 2*maxHittables 크기로 잡는다 (해제 시 이 배열을 순회).
	Hittable** bvhNodes;
	checkCudaErrors(cudaMalloc((void**)&bvhNodes, 2 * maxHittables * sizeof(Hittable*)));

	// 실제 배치된 구체 수를 GPU→CPU로 공유하기 위한 Managed 메모리
	int* d_numHittables;
	checkCudaErrors(cudaMallocManaged((void**)&d_numHittables, sizeof(int)));
	*d_numHittables = 0;

	// 실제 생성된 BVH 노드 수를 GPU→CPU로 공유 (FreeWorld에서 사용)
	int* d_numNodes;
	checkCudaErrors(cudaMallocManaged((void**)&d_numNodes, sizeof(int)));
	*d_numNodes = 0;

	// === 이미지 텍스처 업로드 (scene 2에서만) ===
	// 호스트에서 stb_image로 디코딩 → 디바이스 글로벌 메모리로 업로드.
	// RtwImage 소멸자가 디바이스 버퍼를 해제하므로, 렌더가 끝날 때까지
	// 살아 있도록 main 스코프에 둔다. 파일이 없으면 DeviceData()==nullptr →
	// 커널의 ImageTexture가 청록색을 표시한다.
	RtwImage earthImage;
	const unsigned char* earthData = nullptr;
	int earthW = 0, earthH = 0;
	if (sceneId == 2)
	{
		earthImage.Load("earthmap.jpg");
		earthData = earthImage.DeviceData();
		earthW = earthImage.Width();
		earthH = earthImage.Height();
	}

	CreateWorld<<<1, 1>>>(list, world, camera, imageWidth, imageHeight, randState2, d_numHittables, bvhNodes, d_numNodes,
		sceneId, earthData, earthW, earthH);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int numHittables = *d_numHittables;  // CreateWorld가 기록한 실제 배치 수
	int numNodes = *d_numNodes;          // CreateWorld가 생성한 BVH 노드 수

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

			// [0,1)로 클램프 후 [0,255]로 변환.
			// 발광 재질(빛)은 색이 1.0을 넘을 수 있어, 클램프하지 않으면
			// PPM에 255를 초과하는 값(예: 511)이 찍혀 파일이 깨진다.
			double r = col.X() < 0.0 ? 0.0 : (col.X() > 0.999 ? 0.999 : col.X());
			double g = col.Y() < 0.0 ? 0.0 : (col.Y() > 0.999 ? 0.999 : col.Y());
			double b = col.Z() < 0.0 ? 0.0 : (col.Z() > 0.999 ? 0.999 : col.Z());

			int ir = int(256.0 * r);
			int ig = int(256.0 * g);
			int ib = int(256.0 * b);

			outFile << ir << " " << ig << " " << ib << "\n";
		}
	}
	outFile.close();
	std::cerr << "\nDone. Saved to output.ppm\n";

	// GPU 메모리 해제
	FreeWorld<<<1, 1>>>(list, numHittables, bvhNodes, numNodes, world, camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(randState));
	checkCudaErrors(cudaFree(randState2));
	checkCudaErrors(cudaFree(list));
	checkCudaErrors(cudaFree(bvhNodes));
	checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(camera));
	checkCudaErrors(cudaFree(d_numHittables));
	checkCudaErrors(cudaFree(d_numNodes));
	checkCudaErrors(cudaFree(frameBuffer));

	return 0;
}
