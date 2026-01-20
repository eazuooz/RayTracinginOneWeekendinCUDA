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
- [ ] 구(Sphere) 렌더링 및 히트 레코드(Hit Record)
- [ ] 안티에일리어싱(Antialiasing) & 확산(Diffuse) 재질
- [ ] 금속(Metal) 및 유전체(Dielectric) 재질 구현
- [ ] 위치 조정 가능한 카메라 및 Defocus Blur

### 🚀 Phase 2: CUDA Porting & Core Optimization
> CPU 코드를 CUDA 커널로 변환하고 GPU 아키텍처에 맞게 최적화
- [ ] **CUDA Kernel Launch**: 픽셀 단위 병렬 처리 구현
- [ ] **Iterative Rendering**: 재귀(Recursion) 제거 및 반복문 변환 (Stack Overflow 방지)
- [ ] **Fast RNG**: cuRAND 대체 및 고속 해시 기반 난수 생성기(PCG/XORShift) 적용
- [ ] **Memory Management**: Unified Memory 적용 및 데이터 구조체(SoA 등) 최적화
- [ ] **Float Precision**: 성능 향상을 위한 Float 자료형 전환

### 📗 Phase 3: The Next Week (on CUDA)
> GPU 기반에서의 렌더링 품질 향상 및 가속 구조 구현
- [ ] **Motion Blur**: 시간 차원에 따른 모션 블러 구현
- [ ] **GPU BVH**: GPU 메모리에 최적화된 BVH(Bounding Volume Hierarchies) 구축 및 순회
- [ ] **Texture Mapping**: 텍스처 메모리 및 서피스 객체 활용
- [ ] **Volume Rendering**: Perlin Noise 및 볼륨 렌더링 구현

### 📙 Phase 4: The Rest of Your Life (on CUDA)
> 몬테카를로 적분 및 중요도 샘플링을 통한 수렴 속도 개선
- [ ] **Monte Carlo Integration**: 몬테카를로 적분 구현
- [ ] **Importance Sampling**: 코사인 가중치 및 조명 샘플링을 통한 노이즈 감소
- [ ] **PDF**: 확률 밀도 함수(Probability Density Functions) 관리
- [ ] **Orthonormal Basis**: 정규 직교 기저 생성 및 좌표계 변환

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
