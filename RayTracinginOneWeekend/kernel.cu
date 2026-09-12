
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <ctime>
#include <cfloat>
#include <cstdlib>
#include <cstring>
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
#include "Instance.h"
#include "ConstantMedium.h"
#include "Texture.h"
#include "Material.h"
#include "Metal.h"
#include "Dielectric.h"
#include "Camera.h"
#include "RtwImage.h"
#include "Pdf.h"
#include "MonteCarloDemo.h"

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
__device__ Color RayColor(
	const Ray& r, const Color& background, Hittable** world, Hittable** lights,
	bool bSampleLight, curandState* randState)
{
	Ray currentRay = r;
	Color throughput(1.0, 1.0, 1.0);
	Color accumulated(0.0, 0.0, 0.0);

	for (int i = 0; i < 50; i++)
	{
		HitRecord rec;
		if (!(*world)->Hit(currentRay, 0.001, DBL_MAX, rec, randState))
		{
			// 아무것도 안 맞음 → 배경색
			accumulated += throughput * background;
			return accumulated;
		}

		// 방출광 누적(비발광 재질은 검정이라 영향 없음)
		Color emission = rec.MaterialPtr->Emitted(currentRay, rec, rec.U, rec.V, rec.P);
		accumulated += throughput * emission;

		// === The Rest of Your Life Chapter 12: ScatterRecord로 정리 ===
		// 재질은 "감쇠 + (PDF 또는 정반사 레이)"만 돌려준다. 방향을 실제로 뽑고 밀도로
		// 나누는 일은 전부 여기서 한다.
		ScatterRecord srec;
		if (!rec.MaterialPtr->Scatter(currentRay, rec, srec, randState))
		{
			// 산란 안 함(빛/흡수) → 지금까지 모은 색 반환
			return accumulated;
		}

		// 정반사(거울/유리): 밀도로 나눌 수 없는 델타 분포 → 그대로 따라간다.
		if (srec.bSkipPdf)
		{
			throughput = throughput * srec.Attenuation;
			currentRay = srec.SkipPdfRay;
			continue;
		}

		// === The Rest of Your Life Chapter 6: 중요도 샘플링 계측 ===
		// 몬테카를로 기본식 (적분 ≈ f(r)/p(r)의 평균)을 산란에 적용한다:
		//   색_o = 방출 + 감쇠 * pScatter(방향) * 색_i / pdfValue(방향)
		// 반복형이므로 "감쇠 * pScatter / pdfValue"를 throughput에 곱해 둔다.
		// === The Rest of Your Life Chapter 10: 혼합 밀도 (Mixture Density) ===
		// 재질이 준 PDF(srec.PdfPtr)와 광원 PDF를 반반 섞어 방향을 뽑는다.
		//   pLight   : 광원 쪽으로 (HittablePdf) — 작은 광원을 확실히 맞힌다
		//   pSurface : 재질의 분포 (Lambertian이면 CosinePdf, 볼륨이면 SpherePdf)
		// 어느 쪽에서 뽑았든 그 방향의 밀도는 두 밀도의 평균이다.
		//
		// CUDA 메모: PDF 객체는 전부 스택에 있다(ScatterRecord가 값으로 품고 있고,
		// 여기서는 포인터만 엮는다). 광원이 없는 장면(1·2권)은 재질 PDF만 쓴다.
		Ray scattered;
		double pdfValue = 0.0;
		if (bSampleLight && lights != nullptr && *lights != nullptr)
		{
			HittablePdf lightPdf(*lights, rec.P, randState);
			MixturePdf mixedPdf(&lightPdf, srec.PdfPtr);

			Vector3 direction = mixedPdf.Generate(randState);
			scattered = Ray(rec.P, direction, currentRay.Time());
			pdfValue = mixedPdf.Value(direction);
		}
		else
		{
			Vector3 direction = srec.PdfPtr->Generate(randState);
			scattered = Ray(rec.P, direction, currentRay.Time());
			pdfValue = srec.PdfPtr->Value(direction);
		}

		// 밀도가 0인 방향(광원을 등지는 방향 등)은 기여를 계산할 수 없다.
		if (!(pdfValue > 0.0))
			return accumulated;

		// ※ 10장에서 실제로 부딪힌 버그: 혼합 PDF는 광원 쪽 방향을 뽑기 때문에 표면
		//   아래(수평선 뒤)를 향하는 방향도 나올 수 있다. 그때 pScatter는 0이고 그
		//   샘플의 기여도 0이어야 한다. 이 경우를 "정반사"처럼 취급해 감쇠만 곱하면
		//   레이가 계속 나아가 에너지가 부풀어 오른다(그림이 6배 밝아졌다).
		double scatteringPdf = rec.MaterialPtr->ScatteringPdf(currentRay, rec, scattered);
		if (!(scatteringPdf > 0.0))
			return accumulated;

		throughput = throughput * srec.Attenuation * (scatteringPdf / pdfValue);
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
//
// === The Rest of Your Life Chapter 2: 층화 샘플링 (Stratified Samples / Jittering) ===
// 픽셀을 sqrtSpp x sqrtSpp 격자로 나누고, 각 칸 안에서 무작위 위치를 하나씩 뽑는다.
// 샘플이 픽셀 안에 고르게 퍼져서, 같은 샘플 수라도 물체 경계처럼 변화가 급한 곳이
// 더 또렷해진다. bStratify=false면 예전처럼 픽셀 전체에서 순수 무작위로 뽑는다
// (비교용). 스레드 하나가 픽셀 하나를 맡는 구조는 그대로라, 층화는 이 커널 안의
// 이중 루프만 바꾸면 된다.
__global__ void Render(
	Vector3* frameBuffer, int maxX, int maxY, int sqrtSpp, bool bStratify,
	bool bSampleLight, Camera** camera, Hittable** world, Hittable** lights,
	curandState* randState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX + i;
	curandState localRandState = randState[pixelIndex];
	Color col(0.0, 0.0, 0.0);

	// 장면 배경색(빛 장면은 검정). 카메라에 저장해 둔 값을 읽는다.
	Color background = (*camera)->Background();

	// 격자 한 칸의 크기(픽셀 폭 = 1 기준)
	double recipSqrtSpp = 1.0 / double(sqrtSpp);

	for (int sj = 0; sj < sqrtSpp; sj++)
	{
		for (int si = 0; si < sqrtSpp; si++)
		{
			// 픽셀 안의 샘플 위치 (px, py) ∈ [0,1)^2
			double px, py;
			if (bStratify)
			{
				// (si, sj) 칸 안에서만 무작위 (원서 sample_square_stratified)
				px = (si + curand_uniform(&localRandState)) * recipSqrtSpp;
				py = (sj + curand_uniform(&localRandState)) * recipSqrtSpp;
			}
			else
			{
				// 픽셀 전체에서 무작위 (층화 없음)
				px = curand_uniform(&localRandState);
				py = curand_uniform(&localRandState);
			}

			double u = (double(i) + px) / double(maxX);
			double v = (double(j) + py) / double(maxY);
			Ray r = (*camera)->GetRay(u, v, &localRandState);
			col += RayColor(r, background, world, lights, bSampleLight, &localRandState);
		}
	}

	randState[pixelIndex] = localRandState;
	col = col / double(sqrtSpp * sqrtSpp);

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
//   7: cornell_box(상자2) — 회전·이동시킨 직육면체 2개 추가 (인스턴스 시연)
//   8: cornell_smoke      — 두 상자를 연기/안개 볼륨으로 (ConstantMedium)
//   9: final_scene        — 모든 기능을 모은 최종 장면 (원서 Listing 74)
//  10: cornell_box(3권)   — The Rest of Your Life의 기준 코넬 박스 (600x600)
//
// earthData/earthW/earthH: 호스트가 stb_image로 로드해 디바이스에 올린
// RGB 바이트 버퍼와 크기(scene 2에서만 사용). 로드 실패 시 nullptr → 청록색.
__global__ void CreateWorld(
	Hittable** list, Hittable** world, Camera** camera,
	int imageWidth, int imageHeight, curandState* randState, int* outCount,
	Hittable** bvhNodes, int* outNodeCount,
	int sceneId, const unsigned char* earthData, int earthW, int earthH,
	Hittable** lights)
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
		else if (sceneId == 6)
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
		else if (sceneId == 7)
		{
			// === cornell_box (두 회전 상자): 인스턴스 시연 (원서 Listing 62~70) ===
			// scene 6과 같은 5벽+광원에, 회전·이동시킨 직육면체 2개를 추가한다.
			//   키 큰 상자  : 15° 회전 후 (265,0,295)로 이동
			//   키 작은 상자: -18° 회전 후 (130,0,65)로 이동
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

			// 키 큰 상자 (원서 Listing 70). MakeBox → RotateY → Translate 로 감싼다.
			Hittable* box1 = MakeBox(Point3(0, 0, 0), Point3(165, 330, 165), white);
			box1 = new RotateY(box1, 15.0);
			box1 = new Translate(box1, Vector3(265, 0, 295));
			list[i++] = box1;

			// 키 작은 상자
			Hittable* box2 = MakeBox(Point3(0, 0, 0), Point3(165, 165, 165), white);
			box2 = new RotateY(box2, -18.0);
			box2 = new Translate(box2, Vector3(130, 0, 65));
			list[i++] = box2;

			background = Color(0.0, 0.0, 0.0);
			lookfrom = Vector3(278.0, 278.0, -800.0);
			lookat = Vector3(278.0, 278.0, 0.0);
			vfov = 40.0;
			aperture = 0.0;
		}
		else if (sceneId == 8)
		{
			// === cornell_smoke: 연기/안개 상자 코넬 박스 (원서 Listing 73) ===
			// scene 7의 두 회전 상자를 ConstantMedium(볼륨)으로 감싼다.
			//   상자1 → 밀도 0.01, 색 (0,0,0) = 어두운 연기
			//   상자2 → 밀도 0.01, 색 (1,1,1) = 밝은 안개
			// 빠른 수렴을 위해 광원을 더 크고(113,554,127 ~) 어둡게((7,7,7)) 잡는다.
			Material* red = new Lambertian(Color(0.65, 0.05, 0.05));
			Material* white = new Lambertian(Color(0.73, 0.73, 0.73));
			Material* green = new Lambertian(Color(0.12, 0.45, 0.15));
			Material* light = new DiffuseLight(Color(7.0, 7.0, 7.0));

			list[i++] = new Quad(Vector3(555, 0, 0), Vector3(0, 555, 0), Vector3(0, 0, 555), green);
			list[i++] = new Quad(Vector3(0, 0, 0), Vector3(0, 555, 0), Vector3(0, 0, 555), red);
			list[i++] = new Quad(Vector3(113, 554, 127), Vector3(330, 0, 0), Vector3(0, 0, 305), light);
			list[i++] = new Quad(Vector3(0, 555, 0), Vector3(555, 0, 0), Vector3(0, 0, 555), white);
			list[i++] = new Quad(Vector3(0, 0, 0), Vector3(555, 0, 0), Vector3(0, 0, 555), white);
			list[i++] = new Quad(Vector3(0, 0, 555), Vector3(555, 0, 0), Vector3(0, 555, 0), white);

			// 키 큰 상자 → 어두운 연기
			Hittable* box1 = MakeBox(Point3(0, 0, 0), Point3(165, 330, 165), white);
			box1 = new RotateY(box1, 15.0);
			box1 = new Translate(box1, Vector3(265, 0, 295));
			list[i++] = new ConstantMedium(box1, 0.01, Color(0.0, 0.0, 0.0));

			// 키 작은 상자 → 밝은 안개
			Hittable* box2 = MakeBox(Point3(0, 0, 0), Point3(165, 165, 165), white);
			box2 = new RotateY(box2, -18.0);
			box2 = new Translate(box2, Vector3(130, 0, 65));
			list[i++] = new ConstantMedium(box2, 0.01, Color(1.0, 1.0, 1.0));

			background = Color(0.0, 0.0, 0.0);
			lookfrom = Vector3(278.0, 278.0, -800.0);
			lookat = Vector3(278.0, 278.0, 0.0);
			vfov = 40.0;
			aperture = 0.0;
		}
		else if (sceneId == 9)
		{
			// === final_scene: 모든 새 기능을 테스트하는 최종 장면 (원서 Listing 74) ===
			// 전체를 덮는 크고 얇은 안개 + 청색 표면하부 산란 구(유전체 내부의 볼륨)를
			// 포함해 지금까지의 모든 요소를 한 장면에 모은다.

			// 바닥: 20x20 = 400개, 높이가 랜덤인 상자 (원서 boxes1)
			Material* ground = new Lambertian(Color(0.48, 0.83, 0.53));
			const int boxesPerSide = 20;
			for (int bi = 0; bi < boxesPerSide; bi++)
			{
				for (int bj = 0; bj < boxesPerSide; bj++)
				{
					double w = 100.0;
					double x0 = -1000.0 + bi * w;
					double z0 = -1000.0 + bj * w;
					double x1 = x0 + w;
					double y1 = 1.0 + 100.0 * RND;   // random_double(1,101)
					double z1 = z0 + w;
					list[i++] = MakeBox(Point3(x0, 0.0, z0), Point3(x1, y1, z1), ground);
				}
			}

			// 광원
			Material* light = new DiffuseLight(Color(7.0, 7.0, 7.0));
			list[i++] = new Quad(Vector3(123, 554, 147), Vector3(300, 0, 0), Vector3(0, 0, 265), light);

			// 모션 블러 구 (center1 → center2 로 이동)
			Material* sphereMaterial = new Lambertian(Color(0.7, 0.3, 0.1));
			list[i++] = new MovingSphere(
				Point3(400, 400, 200), Point3(430, 400, 200), 0.0, 1.0, 50.0, sphereMaterial);

			// 유리 구 + 금속 구
			list[i++] = new Sphere(Point3(260, 150, 45), 50.0, new Dielectric(1.5));
			list[i++] = new Sphere(Point3(0, 150, 145), 50.0, new Metal(Color(0.8, 0.8, 0.9), 1.0));

			// 청색 표면하부 산란 구: 유리 경계 + 내부 볼륨.
			// 원서는 같은 sphere 포인터를 유리와 볼륨이 공유하지만, 우리의 raw 포인터
			// 모델에서는 더블 프리가 되므로 동일한 구를 2개 만든다(하나는 보이는 유리,
			// 하나는 ConstantMedium이 소유하는 경계).
			list[i++] = new Sphere(Point3(360, 150, 145), 70.0, new Dielectric(1.5));
			Hittable* blueBoundary = new Sphere(Point3(360, 150, 145), 70.0, new Dielectric(1.5));
			list[i++] = new ConstantMedium(blueBoundary, 0.2, Color(0.2, 0.4, 0.9));

			// 전체를 덮는 얇은 안개: 거대한(반지름 5000) 유리 구 경계 + 매우 옅은 볼륨
			Hittable* mistBoundary = new Sphere(Point3(0, 0, 0), 5000.0, new Dielectric(1.5));
			list[i++] = new ConstantMedium(mistBoundary, 0.0001, Color(1.0, 1.0, 1.0));

			// 지구 이미지 텍스처 구 (earthmap.jpg; 없으면 청록색)
			Texture* earthTex = new ImageTexture(earthData, earthW, earthH);
			list[i++] = new Sphere(Point3(400, 200, 400), 100.0, new Lambertian(earthTex));

			// 펄린 노이즈 구
			Texture* pertext = new NoiseTexture(0.2, &localRandState);
			list[i++] = new Sphere(Point3(220, 280, 300), 80.0, new Lambertian(pertext));

			// 1000개의 작은 흰 구 클러스터 → 15° 회전 + 이동.
			// 원서는 bvh_node로 묶지만, Translate/RotateY가 자식을 delete하는 우리
			// 메모리 모델과 BVH 노드 레지스트리 해제가 충돌한다. 그래서 "소유하는
			// HittableList"로 묶는다(MakeBox와 동일 패턴). 메인 BVH가 이 그룹의 경계
			// 상자로 컬링하므로, 그룹 bbox에 들어온 레이만 1000개를 선형 검사한다.
			Material* white = new Lambertian(Color(0.73, 0.73, 0.73));
			const int ns = 1000;
			Hittable** boxes2 = new Hittable*[ns];
			for (int s = 0; s < ns; s++)
			{
				Point3 c(165.0 * RND, 165.0 * RND, 165.0 * RND);   // point3::random(0,165)
				boxes2[s] = new Sphere(c, 10.0, white);
			}
			Hittable* cluster = new HittableList(boxes2, ns, true);   // 1000개 구를 소유
			cluster = new RotateY(cluster, 15.0);
			cluster = new Translate(cluster, Vector3(-100, 270, 395));
			list[i++] = cluster;

			background = Color(0.0, 0.0, 0.0);
			lookfrom = Vector3(478.0, 278.0, -600.0);
			lookat = Vector3(278.0, 278.0, 0.0);
			vfov = 40.0;
			aperture = 0.0;
			shutterOpen = 0.0;     // 모션 블러 셔터 구간 (이동 구와 일치)
			shutterClose = 1.0;
		}
		else if (sceneId == 10)
		{
			// === The Rest of Your Life: 코넬 박스 다시 보기 (3권 2장 "Cornell box, revisited") ===
			// 2권 scene 7과 같은 방이지만, 3권은 벽 사각형의 시작 모서리/변 방향과 광원
			// 위치를 조금 다르게 잡는다(광원: 천장 가운데 x 213~343, z 227~332).
			// 3권 전체가 이 장면을 기준으로 노이즈를 비교하므로, 이미지도 원서처럼
			// 600x600 정사각형으로 렌더한다(main 참고).
			Material* red = new Lambertian(Color(0.65, 0.05, 0.05));
			Material* white = new Lambertian(Color(0.73, 0.73, 0.73));
			Material* green = new Lambertian(Color(0.12, 0.45, 0.15));
			Material* light = new DiffuseLight(Color(15.0, 15.0, 15.0));

			// 코넬 박스 벽 5개
			list[i++] = new Quad(Point3(555, 0, 0), Vector3(0, 0, 555), Vector3(0, 555, 0), green);
			list[i++] = new Quad(Point3(0, 0, 555), Vector3(0, 0, -555), Vector3(0, 555, 0), red);
			list[i++] = new Quad(Point3(0, 555, 0), Vector3(555, 0, 0), Vector3(0, 0, 555), white);
			list[i++] = new Quad(Point3(0, 0, 555), Vector3(555, 0, 0), Vector3(0, 0, -555), white);
			list[i++] = new Quad(Point3(555, 0, 555), Vector3(-555, 0, 0), Vector3(0, 555, 0), white);

			// 천장 광원
			list[i++] = new Quad(Point3(213, 554, 227), Vector3(130, 0, 0), Vector3(0, 0, 105), light);

			// 키 큰 상자
			Hittable* box1 = MakeBox(Point3(0, 0, 0), Point3(165, 330, 165), white);
			box1 = new RotateY(box1, 15.0);
			box1 = new Translate(box1, Vector3(265, 0, 295));
			list[i++] = box1;

			// 키 작은 상자
			Hittable* box2 = MakeBox(Point3(0, 0, 0), Point3(165, 165, 165), white);
			box2 = new RotateY(box2, -18.0);
			box2 = new Translate(box2, Vector3(130, 0, 65));
			list[i++] = box2;

			background = Color(0.0, 0.0, 0.0);
			lookfrom = Vector3(278.0, 278.0, -800.0);
			lookat = Vector3(278.0, 278.0, 0.0);
			vfov = 40.0;
			aperture = 0.0;
		}
		else if (sceneId == 11 || sceneId == 12)
		{
			// === The Rest of Your Life Chapter 12 ===
			//  11: 키 큰 상자를 알루미늄(금속)으로 — 정반사가 되살아났는지 확인 (원서 이미지 12)
			//  12: 키 작은 상자 대신 유리 구 — 구를 향한 샘플링 (원서 이미지 13·14)
			Material* red = new Lambertian(Color(0.65, 0.05, 0.05));
			Material* white = new Lambertian(Color(0.73, 0.73, 0.73));
			Material* green = new Lambertian(Color(0.12, 0.45, 0.15));
			Material* light = new DiffuseLight(Color(15.0, 15.0, 15.0));

			// 코넬 박스 벽 5개 (scene 10과 동일)
			list[i++] = new Quad(Point3(555, 0, 0), Vector3(0, 0, 555), Vector3(0, 555, 0), green);
			list[i++] = new Quad(Point3(0, 0, 555), Vector3(0, 0, -555), Vector3(0, 555, 0), red);
			list[i++] = new Quad(Point3(0, 555, 0), Vector3(555, 0, 0), Vector3(0, 0, 555), white);
			list[i++] = new Quad(Point3(0, 0, 555), Vector3(555, 0, 0), Vector3(0, 0, -555), white);
			list[i++] = new Quad(Point3(555, 0, 555), Vector3(-555, 0, 0), Vector3(0, 555, 0), white);

			// 천장 광원
			list[i++] = new Quad(Point3(213, 554, 227), Vector3(130, 0, 0), Vector3(0, 0, 105), light);

			// 키 큰 상자 (11: 알루미늄, 12: 흰색)
			Material* tallBoxMaterial = (sceneId == 11)
				? (Material*)new Metal(Color(0.8, 0.85, 0.88), 0.0)
				: white;
			Hittable* box1 = MakeBox(Point3(0, 0, 0), Point3(165, 330, 165), tallBoxMaterial);
			box1 = new RotateY(box1, 15.0);
			box1 = new Translate(box1, Vector3(265, 0, 295));
			list[i++] = box1;

			if (sceneId == 11)
			{
				// 키 작은 상자 (scene 10과 동일)
				Hittable* box2 = MakeBox(Point3(0, 0, 0), Point3(165, 165, 165), white);
				box2 = new RotateY(box2, -18.0);
				box2 = new Translate(box2, Vector3(130, 0, 65));
				list[i++] = box2;
			}
			else
			{
				// 유리 구 (원서 Listing sampling-sphere)
				list[i++] = new Sphere(Point3(190, 90, 190), 90.0, new Dielectric(1.5));
			}

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

		// === The Rest of Your Life Chapter 10: 광원 샘플링 대상 ===
		// 3권 코넬 박스의 천장 광원과 같은 사각형을 하나 더 만들어 "이쪽으로 레이를
		// 보내라"는 대상으로 쓴다. 밀도 계산과 점 뽑기에만 쓰이므로 재질은 필요 없다.
		// (월드에 넣지 않으므로 BVH나 화면에는 영향이 없다.)
		if (sceneId == 10 || sceneId == 11)
		{
			*lights = new Quad(
				Point3(213.0, 554.0, 227.0),
				Vector3(130.0, 0.0, 0.0),
				Vector3(0.0, 0.0, 105.0),
				nullptr);
		}
		else if (sceneId == 12)
		{
			// === 3권 12장 === 광원이 둘(천장 광원 + 유리 구)이면 리스트째로 샘플링한다.
			// HittableList가 소유(bOwns=true)하므로 FreeWorld에서 *lights 하나만 지우면
			// 내부 사각형·구와 배열까지 연쇄 해제된다.
			Hittable** lightList = new Hittable*[2];
			lightList[0] = new Quad(
				Point3(213.0, 554.0, 227.0),
				Vector3(130.0, 0.0, 0.0),
				Vector3(0.0, 0.0, 105.0),
				nullptr);
			lightList[1] = new Sphere(Point3(190.0, 90.0, 190.0), 90.0, nullptr);
			*lights = new HittableList(lightList, 2, true);
		}
		else
		{
			*lights = nullptr;
		}

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
	Hittable** world, Camera** camera, Hittable** lights)
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

		// 3권 10장: 광원 샘플링용 사각형(월드와 별개로 만든 것)
		if (*lights != nullptr)
			delete *lights;
	}
}

// CreateWorld가 아는 가장 큰 장면 번호
static const int kMaxSceneId = 12;

int main(int argc, char** argv)
{
	// === 렌더링할 장면 선택 ===
	//   0: bouncing_spheres  — 바닥이 체커 텍스처인 최종 랜덤 구 장면
	//   1: checkered_spheres — 체커 구 2개
	//   2: earth             — 지구 이미지 텍스처 구 (earthmap.jpg 필요)
	//   3: perlin_spheres    — 펄린 노이즈(대리석) 구 2개
	//   4: quads             — 5색 사각형(평행사변형) 장면
	//   5: simple_light      — 사각형/구 광원 (배경 검정)
	//   6: cornell_box       — 빈 코넬 박스 (천장 광원, 배경 검정)
	//   7: cornell_box(상자2) — 회전·이동시킨 직육면체 2개를 넣은 코넬 박스
	//   8: cornell_smoke      — 두 상자를 연기/안개 볼륨으로 (ConstantMedium)
	//   9: final_scene        — 모든 기능을 모은 최종 장면 (2권 Listing 74)
	//  10: cornell_box(3권)   — The Rest of Your Life의 기준 코넬 박스
	//  11: cornell_aluminum   — 키 큰 상자를 금속으로 (3권 12장)
	//  12: cornell_glass      — 키 작은 상자 대신 유리 구 (3권 12장)
	int sceneId = 12;
	int numSamples = -1;            // -1이면 아래에서 장면별 기본값을 쓴다
	bool bStratify = true;          // 3권 2장: 픽셀 안 층화 샘플링
	bool bSampleLight = true;       // 3권 9장: 광원 직접 샘플링(3권 코넬 박스에만 적용)
	const char* outPath = "output.ppm";

	// === 명령행 옵션 (3권부터 추가) ===
	// 인자 없이 실행하면(VS F5 / 빌드 후 이벤트) 위 기본값으로 output.ppm을 만든다.
	//   --demo <이름>   몬테카를로 데모 실행 (MonteCarloDemo.cu, 렌더는 하지 않음)
	//   --scene <번호>  렌더할 장면
	//   --spp <수>      픽셀당 샘플 수 (층화 격자 때문에 제곱수로 내림)
	//   --nostrat       층화 샘플링 끄기 (비교용)
	//   --out <파일>    출력 PPM 경로
	for (int a = 1; a < argc; a++)
	{
		if (strcmp(argv[a], "--demo") == 0 && a + 1 < argc)
		{
			if (!RunMonteCarloDemo(argv[a + 1]))
			{
				std::cerr << "Unknown demo: " << argv[a + 1] << "\n";
				PrintMonteCarloDemoList();
				return 1;
			}
			return 0;
		}
		else if (strcmp(argv[a], "--scene") == 0 && a + 1 < argc)
		{
			sceneId = atoi(argv[++a]);
		}
		else if (strcmp(argv[a], "--spp") == 0 && a + 1 < argc)
		{
			numSamples = atoi(argv[++a]);
		}
		else if (strcmp(argv[a], "--nostrat") == 0)
		{
			bStratify = false;
		}
		else if (strcmp(argv[a], "--nolightsample") == 0)
		{
			bSampleLight = false;
		}
		else if (strcmp(argv[a], "--out") == 0 && a + 1 < argc)
		{
			outPath = argv[++a];
		}
		else
		{
			std::cerr << "Unknown option: " << argv[a] << "\n";
			return 1;
		}
	}

	if (sceneId < 0 || sceneId > kMaxSceneId)
	{
		std::cerr << "Scene id must be 0.." << kMaxSceneId << "\n";
		return 1;
	}

	// 3권 장면(10~)은 원서처럼 600x600 정사각형, 1·2권 장면은 기존 1440x720.
	bool bBook3Scene = (sceneId >= 10);
	int imageWidth = bBook3Scene ? 600 : 1440;
	int imageHeight = bBook3Scene ? 600 : 720;

	// 광원 직접 샘플링은 광원 좌표를 하드코딩한 임시 구현이라 3권 코넬 박스에서만 켠다.
	bSampleLight = bSampleLight && bBook3Scene;

	int blockWidth = 8;
	int blockHeight = 8;

	// 픽셀당 샘플 수 기본값. 빛/볼륨 장면(5~9)은 작은 광원·산란 때문에 노이즈가
	// 심하므로 크게 잡는다. 최종 장면(9)은 무거워 100, 3권 코넬 박스는 원서 6장처럼
	// 1000(층화 격자 때문에 31x31 = 961).
	if (numSamples <= 0)
	{
		if (bBook3Scene) numSamples = 1000;
		else if (sceneId == 9) numSamples = 100;
		else if (sceneId >= 5) numSamples = 200;
		else numSamples = 10;
	}

	// 층화 샘플링은 픽셀을 sqrtSpp x sqrtSpp 격자로 나누므로, 실제 샘플 수는
	// 제곱수로 내림된다(원서 camera::initialize의 sqrt_spp와 같다).
	int sqrtSpp = int(sqrt(double(numSamples)));
	if (sqrtSpp < 1) sqrtSpp = 1;
	numSamples = sqrtSpp * sqrtSpp;

	// GPU 스택 크기 증가
	// MovingSphere 추가로 가상함수 깊이가 늘어 스택 소비 증가 → 32768로 확장.
	// BVH 순회는 재귀 대신 명시적 스택(BvhNode::Hit)을 쓰므로 추가 스택은
	// 필요 없다(재귀로 두면 이 한도로도 일부 스레드에서 스택이 넘쳤다).
	checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 32768));

	std::cerr << "Rendering scene " << sceneId << ": " << imageWidth << "x" << imageHeight
		<< " image with " << numSamples << " samples per pixel ("
		<< (bStratify ? "stratified " : "random ") << sqrtSpp << "x" << sqrtSpp << ") "
		<< "in " << blockWidth << "x" << blockHeight << " blocks"
		<< (bSampleLight ? ", sampling the light directly" : "") << ".\n";

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
	// 최대 잎(leaf) 수는 가장 무거운 두 장면을 모두 수용해야 한다:
	//   scene 0 (bouncing): 22*22(소형) + 1(바닥) + 3(대형) = 488 (continue로 일부 제외)
	//   scene 9 (final)    : 400(바닥 상자) + 10(기타 오브젝트) = 410
	// (scene 9의 1000개 구 클러스터는 "소유 HittableList" 안에 있어 list[]에 들어가지 않음)
	int maxHittables = 512;
	Hittable** list;
	checkCudaErrors(cudaMalloc((void**)&list, maxHittables * sizeof(Hittable*)));
	Hittable** world;
	checkCudaErrors(cudaMalloc((void**)&world, sizeof(Hittable*)));
	Camera** camera;
	checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Camera*)));

	// 3권 10장: 광원 샘플링 대상(없는 장면이면 nullptr)
	Hittable** lights;
	checkCudaErrors(cudaMalloc((void**)&lights, sizeof(Hittable*)));

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

	// === 이미지 텍스처 업로드 (scene 2 / scene 9에서 사용) ===
	// 호스트에서 stb_image로 디코딩 → 디바이스 글로벌 메모리로 업로드.
	// RtwImage 소멸자가 디바이스 버퍼를 해제하므로, 렌더가 끝날 때까지
	// 살아 있도록 main 스코프에 둔다. 파일이 없으면 DeviceData()==nullptr →
	// 커널의 ImageTexture가 청록색을 표시한다.
	RtwImage earthImage;
	const unsigned char* earthData = nullptr;
	int earthW = 0, earthH = 0;
	if (sceneId == 2 || sceneId == 9)
	{
		earthImage.Load("earthmap.jpg");
		earthData = earthImage.DeviceData();
		earthW = earthImage.Width();
		earthH = earthImage.Height();
	}

	CreateWorld<<<1, 1>>>(list, world, camera, imageWidth, imageHeight, randState2, d_numHittables, bvhNodes, d_numNodes,
		sceneId, earthData, earthW, earthH, lights);
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
		sqrtSpp, bStratify, bSampleLight, camera, world, lights, randState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timerSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timerSeconds << " seconds.\n";

	// PPM 이미지 파일 저장
	std::ofstream outFile(outPath);
	outFile << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

	for (int j = imageHeight - 1; j >= 0; j--)
	{
		std::cerr << "\rWriting scanline " << (imageHeight - 1 - j)
			<< " / " << imageHeight << std::flush;

		for (int i = 0; i < imageWidth; i++)
		{
			size_t pixelIndex = j * imageWidth + i;
			Color col = frameBuffer[pixelIndex];

			// === The Rest of Your Life Chapter 12: NaN 걸러내기 ===
			// 몬테카를로 렌더러에서는 수천만 레이에 한 번쯤 NaN이 나올 수 있다. 평균에
			// NaN이 하나 섞이면 그 픽셀 전체가 죽어 검은 점(acne)으로 남는다.
			// NaN은 자기 자신과 같지 않다는 성질을 이용해 걸러 0으로 바꾼다.
			if (col.X() != col.X()) col[0] = 0.0;
			if (col.Y() != col.Y()) col[1] = 0.0;
			if (col.Z() != col.Z()) col[2] = 0.0;

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
	std::cerr << "\nDone. Saved to " << outPath << "\n";

	// GPU 메모리 해제
	FreeWorld<<<1, 1>>>(list, numHittables, bvhNodes, numNodes, world, camera, lights);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(randState));
	checkCudaErrors(cudaFree(randState2));
	checkCudaErrors(cudaFree(list));
	checkCudaErrors(cudaFree(bvhNodes));
	checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(camera));
	checkCudaErrors(cudaFree(lights));
	checkCudaErrors(cudaFree(d_numHittables));
	checkCudaErrors(cudaFree(d_numNodes));
	checkCudaErrors(cudaFree(frameBuffer));

	return 0;
}
