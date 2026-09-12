# RayTracingInOneWeekendinCUDA

![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Language](https://img.shields.io/badge/language-C++-blue.svg)

**RayTracingInOneWeekendinCUDA**는 Peter Shirley의 유명한 저서인 *Ray Tracing in One Weekend* 시리즈(Book 1, 2, 3)를 기반으로 레이 트레이서를 구현하고, 이를 **CUDA C++**로 포팅하여 병렬 처리 및 고성능 최적화를 수행하는 프로젝트입니다.

이 프로젝트는 Book 1의 기초 이론을 구현한 직후 CUDA로 전환하여, 이후의 심화 과정(Book 2, 3)을 GPU 가속 환경에서 개발하는 것을 목표로 합니다.

## 👨‍💻 Author
**YamYamCoding**

## 📚 Project Roadmap (진행 계획)

### 📘 Phase 1: In One Weekend (CPU Prototype)
> 레이 트레이싱의 기초 이론 및 CPU 기반 프로토타입 구현
- [x] 광선(Ray) 생성 및 카메라 설정
- [x] 구(Sphere) 렌더링 및 히트 레코드(Hit Record)
- [x] 안티에일리어싱(Antialiasing) & 확산(Diffuse) 재질
- [x] 금속(Metal) 및 유전체(Dielectric) 재질 구현
- [x] 위치 조정 가능한 카메라 및 Defocus Blur

### 🚀 Phase 2: CUDA Porting & Core Optimization
> CPU 코드를 CUDA 커널로 변환하고 GPU 아키텍처에 맞게 최적화
- [x] **CUDA Kernel Launch**: 픽셀 단위 병렬 처리 구현
- [x] **Iterative Rendering**: 재귀(Recursion) 제거 및 반복문 변환 (Stack Overflow 방지)
- [x] **RNG**: cuRAND 사용 (렌더러는 XORWOW, 3권 데모는 초기화가 빠른 Philox)
- [ ] **Fast RNG**: 해시 기반 생성기(PCG/XORShift)로의 교체는 아직 미적용
- [x] **Memory Management**: Unified Memory 적용 및 데이터 구조체(SoA 등) 최적화
- [ ] **Float Precision**: 현재 전 구간 `double` 사용. 소비자용 GPU는 FP64가 1/64 속도라
      가장 큰 최적화 후보로 남아 있다 (3권 11장 문서 참고)

### 📗 Phase 3: The Next Week (on CUDA)
> GPU 기반에서의 렌더링 품질 향상 및 가속 구조 구현
- [x] **Motion Blur**: 시간 차원에 따른 모션 블러 구현
- [x] **GPU BVH**: GPU 메모리에 최적화된 BVH(Bounding Volume Hierarchies) 구축 및 순회
- [x] **Texture Mapping**: 절차적(체커)·이미지(stb)·펄린 노이즈 텍스처 구현
- [x] **Volume Rendering**: `ConstantMedium`으로 연기/안개 구현 (장면 8)

### 📙 Phase 4: The Rest of Your Life (on CUDA)
> 몬테카를로 적분 및 중요도 샘플링을 통한 수렴 속도 개선
- [x] **Monte Carlo Integration**: GPU 몬테카를로 데모(π 추정, 1차원 적분, 구면 적분)
- [x] **Stratified Sampling**: 픽셀 안 층화(지터링) 샘플링
- [x] **Importance Sampling**: 코사인 가중치 및 광원 직접 샘플링으로 노이즈 감소
- [x] **PDF**: `Pdf.h`(Sphere/Cosine/Hittable/Mixture) + `ScatterRecord`로 PDF 관리
- [x] **Orthonormal Basis**: `Onb.h` 생성 및 좌표계 변환

**결과**: 코넬 박스 기준 961 spp에서 뒷벽 노이즈 17.2 → 12.6, 렌더 시간 40.9초 → 9.9초
(RTX 5070 Ti, 600x600). 장별 정리는 `Docs/3권_*_CUDA적용판.md` 참고.

## 🛠️ Development Environment

* **OS**: Windows 10 / 11
* **GPU**: NVIDIA GPU (Compute Capability 6.0+)
* **Language**: C++17
* **Toolkit**: CUDA Toolkit 12.x
* **IDE**: Visual Studio Community 2022

## 🏗️ Build & Run

이 프로젝트는 **Visual Studio Community 2022** 솔루션으로 관리됩니다.

1. **Clone Repository**
   ```bash
   git clone [https://github.com/eazuooz/RayTracingInOneWeekendinCUDA.git](https://github.com/eazuooz/RayTracingInOneWeekendinCUDA.git)
   ```

2. **빌드**: `RayTracinginOneWeekendinCUDA.sln`을 VS2022로 열고 **Release / x64**로 빌드합니다.
   Debug 구성은 CUDA 설정상 GPU 디버그 정보(`-G`)가 켜져 디바이스 코드가 매우 느려지므로,
   렌더링은 Release 권장입니다.

   명령행 빌드:
   ```bash
   msbuild RayTracinginOneWeekendinCUDA.sln /t:RayTracinginOneWeekend /p:Configuration=Release /p:Platform=x64
   ```

## ▶️ 실행 방법 (Usage)

인자 없이 실행하면 기본 장면을 `output.ppm`으로 렌더합니다.

| 옵션 | 설명 |
|---|---|
| `--scene <0-12>` | 렌더할 장면 |
| `--spp <수>` | 픽셀당 샘플 수 (층화 격자 때문에 제곱수로 내림) |
| `--nostrat` | 픽셀 안 층화 샘플링 끄기 (비교용) |
| `--nolightsample` | 광원 직접 샘플링 끄기 (비교용) |
| `--out <파일>` | 출력 PPM 경로 |
| `--demo <이름>` | 렌더 대신 몬테카를로 데모 실행 |

```bash
RayTracinginOneWeekend.exe --scene 12 --spp 1000 --out cornell_glass.ppm
RayTracinginOneWeekend.exe --demo pi
```

### 장면 목록

| 번호 | 장면 | 출처 |
|---|---|---|
| 0 | bouncing_spheres (구 488개, 체커 바닥) | 1·2권 |
| 1 | checkered_spheres | 2권 |
| 2 | earth (이미지 텍스처) | 2권 |
| 3 | perlin_spheres | 2권 |
| 4 | quads | 2권 |
| 5 | simple_light | 2권 |
| 6 | cornell_box (빈 방) | 2권 |
| 7 | cornell_box (상자 2개) | 2권 |
| 8 | cornell_smoke (볼륨) | 2권 |
| 9 | final_scene (모든 기능) | 2권 |
| 10 | cornell_box (3권 기준 장면, 600x600) | 3권 |
| 11 | cornell_aluminum (금속 상자) | 3권 12장 |
| 12 | cornell_glass (유리 구 + 광원 리스트) | 3권 12장 |

### 몬테카를로 데모 (3권)

| 이름 | 장 | 내용 |
|---|---|---|
| `pi` | 2장 | π 추정: 1회 / 누적 수렴 / 일반 vs 층화 |
| `integrate` | 3장 | x², sin⁵, ln(sin) 균일 샘플링 적분 |
| `halfway` | 3장 | PDF의 "넓이 절반" 지점 (GPU 정렬 → 누적합 → 이진 탐색) |
| `importance` | 3장 | 균일/half-split/선형/2차 PDF 비교 (샘플당 표준편차) |
| `sphere` | 4장 | 구면에서 cos² 적분 + 거절법의 워프 비용 측정 |
| `lambert` | 5장 | Lambertian 산란 PDF 정규화 + cos θ 히스토그램 |
| `furnace` | 6장 | 화로 테스트로 f/p 추정기의 편향 확인 |
| `dirs` | 7장 | 역변환법 방향 생성: 적분·분산·생성 비용 비교 |

## 📄 문서

장별 정리 문서는 `Docs/` 아래에 있습니다.

- `Docs/2권_*_CUDA적용판.md` — Ray Tracing: The Next Week
- `Docs/3권_*_CUDA적용판.md` — Ray Tracing: The Rest of Your Life
- `Docs/빌드환경_및_트러블슈팅.md` — VS/CUDA 조합, 인코딩·줄끝 문제 등

