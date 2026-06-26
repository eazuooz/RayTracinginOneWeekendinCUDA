#pragma once
#ifndef RTW_IMAGE_H
#define RTW_IMAGE_H

// === The Next Week Chapter 4: host-side image loader for image textures ===
//
// stb_image is a HOST-only library, so image decoding happens here (on the
// host). The decoded RGB byte buffer is then uploaded to DEVICE global memory
// with cudaMemcpy, and the device-side ImageTexture indexes into that buffer.
//
// NOTE (comments in English on purpose): the stb_image headers combined with
// nvcc's device front-end (cudafe++) are fragile w.r.t. multibyte (CJK) source
// bytes in this translation unit, so this particular header is kept ASCII-only
// to keep the build robust. The Korean write-up lives in the Docs/ markdown.
//
// We do NOT #include "external/stb_image.h" here at all. Even just the stb
// declarations make nvcc's device front-end (cudafe++) crash in this .cu
// translation unit. Instead we forward-declare the two stb functions we use,
// matching stb's C linkage. The real definitions are built in StbImageImpl.cpp
// (which DOES include stb with STB_IMAGE_IMPLEMENTATION, compiled by cl) and
// are resolved at link time. This keeps stb_image.h entirely out of the .cu.
extern "C" {
    float* stbi_loadf(char const* filename, int* x, int* y,
                      int* channels_in_file, int desired_channels);
    void   stbi_image_free(void* retval_from_stbi_load);
}

#include "cuda_runtime.h"
#include <iostream>

// Loads an image on the host, converts it to RGB bytes, and uploads it to
// device global memory. Holds the device pointer/size; frees it on destruction.
// The device-side ImageTexture samples this buffer.
class RtwImage
{
public:
    RtwImage() {}

    explicit RtwImage(const char* filename) { Load(filename); }

    ~RtwImage()
    {
        if (mDeviceData)
            cudaFree(mDeviceData);
    }

    // Load the image and upload it to the device. Returns true on success.
    // stbi_loadf returns pixels as LINEAR float in [0,1] (all our shading math
    // is in linear color space). We convert back to [0,255] bytes for storage,
    // so the device ImageTexture only needs to divide by 255 to recover linear.
    bool Load(const char* filename)
    {
        int channelsInFile = 0;
        float* fdata = stbi_loadf(filename, &mWidth, &mHeight, &channelsInFile, 3);
        if (fdata == nullptr)
        {
            std::cerr << "ERROR: Could not load image file '" << filename << "'.\n";
            mWidth = mHeight = 0;
            return false;
        }

        int totalBytes = mWidth * mHeight * 3;

        // Host-side: linear float -> byte conversion.
        unsigned char* hostBytes = new unsigned char[totalBytes];
        for (int k = 0; k < totalBytes; k++)
            hostBytes[k] = FloatToByte(fdata[k]);
        stbi_image_free(fdata);

        // Upload to device global memory.
        cudaError_t err = cudaMalloc((void**)&mDeviceData, totalBytes);
        if (err != cudaSuccess)
        {
            std::cerr << "ERROR: cudaMalloc for image failed: "
                << cudaGetErrorString(err) << "\n";
            delete[] hostBytes;
            mDeviceData = nullptr;
            mWidth = mHeight = 0;
            return false;
        }
        cudaMemcpy(mDeviceData, hostBytes, totalBytes, cudaMemcpyHostToDevice);
        delete[] hostBytes;

        std::cerr << "Loaded image '" << filename << "' ("
            << mWidth << "x" << mHeight << ") and uploaded to device.\n";
        return true;
    }

    // Device RGB byte buffer (nullptr if no image was loaded).
    const unsigned char* DeviceData() const { return mDeviceData; }
    int Width()  const { return mWidth; }
    int Height() const { return mHeight; }

private:
    unsigned char* mDeviceData = nullptr;  // device buffer
    int mWidth = 0;
    int mHeight = 0;

    // linear [0,1] float -> [0,255] byte
    static unsigned char FloatToByte(float value)
    {
        if (value <= 0.0f) return 0;
        if (1.0f <= value) return 255;
        return static_cast<unsigned char>(256.0f * value);
    }
};

#endif
