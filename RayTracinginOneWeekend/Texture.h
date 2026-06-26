#pragma once
#ifndef TEXTURE_H
#define TEXTURE_H

#include "Vec3.h"
#include "Interval.h"
#include "Perlin.h"

// === The Next Week Chapter 4: Texture Mapping (텍스처 매핑) ===
//
// "텍스처"는 표면에 입히는 효과(색/반짝임/요철 등)이고, "매핑"은 한 공간을
// 다른 공간으로 대응시키는 수학적 의미다. 가장 흔한 텍스처 매핑은 이미지를
// 물체 표면에 입히는 것이다. 우리는 이를 역방향으로 구현한다. 즉, 표면 위의
// 어떤 점이 주어지면 텍스처가 그 점에 정의해 둔 색을 "조회"한다.
//
// 핵심 인터페이스는 Value(u, v, p)다. 텍스처 좌표 (u,v)와 공간상의 점 p를
// 받아 그 위치의 색을 반환한다. (어떤 텍스처는 u,v만, 어떤 텍스처는 p만 쓴다.)
//
// CUDA 적용 메모:
//  - 원서는 shared_ptr<texture>를 쓰지만, 우리 프레임워크는 다른 디바이스
//    오브젝트(Material/Hittable)와 마찬가지로 커널 안에서 new로 만들고
//    raw 포인터(Texture*)로 참조한다. 해제는 호출자가 직접 delete 한다.
//  - 모든 메서드는 디바이스에서 호출되므로 __device__ 다.
class Texture
{
public:
    __device__ virtual ~Texture() {}

    __device__ virtual Color Value(double u, double v, const Point3& p) const = 0;
};

// 단색 텍스처: (u,v)와 무관하게 항상 같은 색을 반환한다.
// "모든 색을 텍스처로 다룰 수 있다"는 설계의 출발점. Lambertian에 단색을
// 주면 내부적으로 이 SolidColor로 감싼다.
class SolidColor : public Texture
{
public:
    __device__ SolidColor(const Color& albedo)
        : mAlbedo(albedo)
    {
    }

    __device__ SolidColor(double red, double green, double blue)
        : mAlbedo(Color(red, green, blue))
    {
    }

    __device__ Color Value(double u, double v, const Point3& p) const override
    {
        return mAlbedo;
    }

private:
    Color mAlbedo;
};

// 공간(solid) 체커 텍스처: 점 p의 위치만으로 색을 결정하는 3차원 체커 무늬.
// floor(x)+floor(y)+floor(z)의 홀짝으로 even/odd 색을 번갈아 고른다.
// invScale로 체커 칸의 크기를 조절한다(scale이 클수록 칸이 커짐).
class CheckerTexture : public Texture
{
public:
    __device__ CheckerTexture(double scale, Texture* even, Texture* odd)
        : mInvScale(1.0 / scale)
        , mEven(even)
        , mOdd(odd)
    {
    }

    __device__ Color Value(double u, double v, const Point3& p) const override
    {
        // 각 좌표의 floor를 정수로. truncation은 0 주변에서 같은 색이 나오므로
        // 항상 왼쪽(음의 무한대)으로 내림하는 floor를 쓴다.
        int xInteger = int(floor(mInvScale * p.X()));
        int yInteger = int(floor(mInvScale * p.Y()));
        int zInteger = int(floor(mInvScale * p.Z()));

        bool isEven = ((xInteger + yInteger + zInteger) % 2) == 0;

        return isEven ? mEven->Value(u, v, p) : mOdd->Value(u, v, p);
    }

private:
    double mInvScale;
    Texture* mEven;
    Texture* mOdd;
};

// 이미지 텍스처: (u,v) 좌표로 2D 이미지를 조회한다.
//
// CUDA 적용의 핵심 차이 — stb_image는 "호스트(CPU)"에서만 동작한다.
// 그래서 이미지 로딩/디코딩은 호스트(RtwImage)에서 끝내고, 픽셀 버퍼를
// cudaMemcpy로 "디바이스 글로벌 메모리"에 올린 뒤, 그 디바이스 포인터를
// 커널로 넘겨 여기 ImageTexture가 그것을 인덱싱한다.
//
// 버퍼 포맷: 픽셀당 3바이트(R,G,B), 왼→오 / 위→아래로 연속 저장된 8비트값.
// (원서 rtw_image의 bdata와 동일한 레이아웃)
class ImageTexture : public Texture
{
public:
    // devData : 디바이스 글로벌 메모리에 올라간 RGB 바이트 버퍼.
    // width/height : 이미지 크기(픽셀).
    __device__ ImageTexture(const unsigned char* devData, int width, int height)
        : mData(devData)
        , mWidth(width)
        , mHeight(height)
    {
    }

    __device__ Color Value(double u, double v, const Point3& p) const override
    {
        // 이미지 데이터가 없으면 디버깅용 청록색을 반환한다.
        if (mHeight <= 0 || mData == nullptr)
            return Color(0.0, 1.0, 1.0);

        // 입력 (u,v)를 [0,1]로 클램프. v는 이미지 좌표계(위가 0)로 뒤집는다.
        u = Interval(0.0, 1.0).Clamp(u);
        v = 1.0 - Interval(0.0, 1.0).Clamp(v);

        int i = int(u * mWidth);
        int j = int(v * mHeight);

        // 경계 보정 (u 또는 v가 1.0이면 인덱스가 폭을 넘을 수 있다)
        if (i >= mWidth)  i = mWidth - 1;
        if (j >= mHeight) j = mHeight - 1;

        const unsigned char* pixel = mData + (j * mWidth + i) * 3;

        double colorScale = 1.0 / 255.0;
        return Color(colorScale * pixel[0],
                     colorScale * pixel[1],
                     colorScale * pixel[2]);
    }

private:
    const unsigned char* mData;   // 디바이스 글로벌 메모리의 RGB 바이트 버퍼
    int mWidth;
    int mHeight;
};

// === The Next Week Chapter 5: 펄린 노이즈 텍스처 ===
//
// Perlin 노이즈로 만드는 절차적(solid) 텍스처. mScale로 노이즈의 주파수를
// 조절한다(클수록 무늬가 더 빨리 변한다). 아래 Value는 챕터의 최종형인
// "대리석(marble)" 효과다 — 색을 sin에 비례시키고 turbulence로 위상을 흔들어
// 줄무늬가 물결치게 한다.
//
// CUDA 메모: Perlin 생성자가 curandState로 격자 데이터를 채우므로,
// NoiseTexture도 생성 시 randState를 받아 멤버 Perlin을 초기화한다.
class NoiseTexture : public Texture
{
public:
    __device__ NoiseTexture(double scale, curandState* randState)
        : mNoise(randState)
        , mScale(scale)
    {
    }

    __device__ Color Value(double u, double v, const Point3& p) const override
    {
        // 대리석 무늬 (원서 Listing 47):
        //   color(.5,.5,.5) * (1 + sin(scale*z + 10*turb(p,7)))
        return Color(0.5, 0.5, 0.5)
            * (1.0 + sin(mScale * p.Z() + 10.0 * mNoise.Turb(p, 7)));

        // 참고 — 챕터의 다른 단계들(주석으로 남김):
        //   - 정수에서 벗어난 부드러운 노이즈:
        //       return Color(1,1,1) * 0.5 * (1.0 + mNoise.Noise(mScale * p));
        //   - 난류 직접 사용(위장막 느낌):
        //       return Color(1,1,1) * mNoise.Turb(p, 7);
    }

private:
    Perlin mNoise;
    double mScale;
};

#endif
