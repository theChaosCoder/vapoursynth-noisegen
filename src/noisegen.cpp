
/**
 *  NoiseGen - VapourSynth Noise Generator
 *
 *  DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
 *  Version 2, December 2004
 *
 *  This program is free software. It comes without any warranty, to
 *  the extent permitted by applicable law. You can redistribute it
 *  and/or modify it under the terms of the Do What The Fuck You Want
 *  To Public License, Version 2, as published by Sam Hocevar. See
 *  http://sam.zoy.org/wtfpl/COPYING for more details.
 *
 **/

#include <VapourSynth.h>
#include <VSHelper.h>

#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#include <string>

#ifdef VS_TARGET_CPU_X86
#include <emmintrin.h>
const size_t sseBytes = 16;
#endif

static const size_t alignment = 32;
static const int noisePlaneHeightMultiplier = 4;

enum NoiseType
{
    NoiseTypeUniform = 1,
    NoiseTypeNormal
};

typedef struct NoiseData
{
    VSNodeRef *node;
    const VSVideoInfo *vi;

    float str, limit, mean, var;
    int type;
    bool dyn, full;
    bool planes[3];


    size_t noisePlaneWidth[3];
    size_t noisePlaneHeight[3];
    size_t noisePlaneStride[3];
    void *noiseBuffer[3];
    size_t noiseBufferStartRowSize;
    std::vector<int> noiseBufferStartRow[3];

    float pixelMax[3];
    float pixelMin[3];
} NoiseData;


#ifdef VS_TARGET_CPU_X86

static inline __m128i convSign8(const __m128i &v1)
{
    alignas(sizeof(__m128i)) static const uint8_t signMask8[16] = { 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 };
    return _mm_xor_si128(v1, _mm_load_si128(reinterpret_cast<const __m128i*>(signMask8)));
}

static inline __m128i convSign16(const __m128i &v1)
{
    alignas(sizeof(__m128i)) static const uint16_t signMask16[8] = { 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000 };
    return _mm_xor_si128(v1, _mm_load_si128(reinterpret_cast<const __m128i*>(signMask16)));
}

#endif

template <typename T>
static inline T clamp(T t, T mini, T maxi)
{
    return std::max(std::min(t, maxi), mini);
}

template <typename T>
void genenateNoiseBufferUniform(std::mt19937_64 &gen, NoiseData *d, int plane)
{
    std::uniform_real_distribution<float> ud(d->mean - d->var, d->mean + d->var);

    d->noiseBuffer[plane] = vs_aligned_malloc<void>(sizeof(T) * d->noisePlaneStride[plane] * d->noisePlaneHeight[plane], alignment);

    T *noiseBufferPtr = reinterpret_cast<T*>(d->noiseBuffer[plane]);
    for (size_t y = 0; y < d->noisePlaneHeight[plane]; ++y) {
        for (size_t x = 0; x < d->noisePlaneStride[plane]; ++x) {
            noiseBufferPtr[x] = static_cast<T>(clamp(ud(gen) * d->str, -d->limit, d->limit));
        }
        noiseBufferPtr += d->noisePlaneStride[plane];
    }
}

template <typename T>
void genenateNoiseBufferNormal(std::mt19937_64 &gen, NoiseData *d, int plane)
{
    const float sigma = std::sqrt(d->var);
    std::normal_distribution<float> nd(d->mean, sigma);

    d->noiseBuffer[plane] = vs_aligned_malloc<void>(sizeof(T) * d->noisePlaneStride[plane] * d->noisePlaneHeight[plane], alignment);

    T *noiseBufferPtr = reinterpret_cast<T*>(d->noiseBuffer[plane]);
    for (size_t y = 0; y < d->noisePlaneHeight[plane]; ++y) {
        for (size_t x = 0; x < d->noisePlaneStride[plane]; ++x) {
            noiseBufferPtr[x] = static_cast<T>(clamp(nd(gen) * d->str, -d->limit, d->limit));
        }
        noiseBufferPtr += d->noisePlaneStride[plane];
    }
}

template <typename T>
void genenateNoiseBuffer(std::mt19937_64 &gen, NoiseData *d, int plane)
{
    if (d->type == NoiseTypeUniform)
        genenateNoiseBufferUniform<T>(gen, d, plane);
    else
        genenateNoiseBufferNormal<T>(gen, d, plane);
}

template <typename PixelType, typename BufferType, typename IType>
static inline void addNoise_c(PixelType *dstp, const int dstStride, const int w, const int h, NoiseData *d, int plane, size_t noiseBufferPtrOffset)
{
    const auto pixelMax = static_cast<IType>(d->pixelMax[plane]);
    const auto pixelMin = static_cast<IType>(d->pixelMin[plane]);

    auto noiseBufferPtr = reinterpret_cast<BufferType*>(d->noiseBuffer[plane]) + noiseBufferPtrOffset * d->noisePlaneStride[plane];
    for (int y = 0; y < h; ++y) {

        for (int x = 0; x < w; ++x) {
            dstp[x] = clamp(static_cast<IType>(dstp[x] + noiseBufferPtr[x]), pixelMin, pixelMax);
        }

        dstp += dstStride;
        noiseBufferPtr += d->noisePlaneStride[plane];
    }
}

#ifdef VS_TARGET_CPU_X86

template <typename PixelType, typename IType>
static inline void addNoise_sse(PixelType *dstp, const int dstStride, const int w, const int h, NoiseData *d, int plane, size_t noiseBufferPtrOffset);

template <>
inline void addNoise_sse<uint8_t, int8_t>(uint8_t *dstp, const int dstStride, const int w, const int h, NoiseData *d, int plane, size_t noiseBufferPtrOffset)
{
    const auto pixelMax = static_cast<uint8_t>(d->pixelMax[plane]);
    const auto pixelMin = static_cast<uint8_t>(d->pixelMin[plane]);

    constexpr int pixelStep = sseBytes / sizeof(uint8_t);
    int modw = w & ~(pixelStep - 1);

    const auto mpixelMax = _mm_set1_epi8(pixelMax);
    const auto mpixelMin = _mm_set1_epi8(pixelMin);

    auto noiseBufferPtr = reinterpret_cast<int8_t*>(d->noiseBuffer[plane]) + noiseBufferPtrOffset * d->noisePlaneStride[plane];
    for (int y = 0; y < h; ++y) {

        int x;
        for (x = 0; x < modw; x += pixelStep) {
            const auto noise = _mm_load_si128(reinterpret_cast<const __m128i*>(noiseBufferPtr + x));
            const auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));
            const auto sum = _mm_adds_epi8(convSign8(src), noise);
            const auto clamp = _mm_min_epu8(_mm_max_epu8(convSign8(sum), mpixelMin), mpixelMax);
            _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), clamp);
        }

        for (; x < w; ++x) {
            dstp[x] = clamp(static_cast<uint8_t>(dstp[x] + noiseBufferPtr[x]), pixelMin, pixelMax);
        }

        dstp += dstStride;
        noiseBufferPtr += d->noisePlaneStride[plane];
    }
}

template <>
inline void addNoise_sse<uint16_t, int16_t>(uint16_t *dstp, const int dstStride, const int w, const int h, NoiseData *d, int plane, size_t noiseBufferPtrOffset)
{
    const auto pixelMax = static_cast<uint16_t>(d->pixelMax[plane]);
    const auto pixelMin = static_cast<uint16_t>(d->pixelMin[plane]);

    constexpr int pixelStep = sseBytes / sizeof(uint16_t);
    int modw = w & ~(pixelStep - 1);

    const auto mpixelMax = _mm_set1_epi16(pixelMax);
    const auto mpixelMin = _mm_set1_epi16(pixelMin);

    auto noiseBufferPtr = reinterpret_cast<int16_t*>(d->noiseBuffer[plane]) + noiseBufferPtrOffset * d->noisePlaneStride[plane];
    for (int y = 0; y < h; ++y) {

        int x;
        for (x = 0; x < modw; x += pixelStep) {
            const auto noise = _mm_load_si128(reinterpret_cast<const __m128i*>(noiseBufferPtr + x));
            const auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));
            const auto sum = _mm_adds_epi16(convSign16(src), noise);
            const auto clamp = _mm_min_epi16(_mm_max_epi16(sum, convSign16(mpixelMin)), convSign16(mpixelMax));
            _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), convSign16(clamp));
        }

        for (; x < w; ++x) {
            dstp[x] = clamp(static_cast<uint16_t>(dstp[x] + noiseBufferPtr[x]), pixelMin, pixelMax);
        }

        dstp += dstStride;
        noiseBufferPtr += d->noisePlaneStride[plane];
    }
}

template <>
inline void addNoise_sse<float, float>(float *dstp, const int dstStride, const int w, const int h, NoiseData *d, int plane, size_t noiseBufferPtrOffset)
{
    const auto pixelMax = static_cast<float>(d->pixelMax[plane]);
    const auto pixelMin = static_cast<float>(d->pixelMin[plane]);

    constexpr int pixelStep = sseBytes / sizeof(float);
    int modw = w & ~(pixelStep - 1);

    const auto mpixelMax = _mm_set1_ps(pixelMax);
    const auto mpixelMin = _mm_set1_ps(pixelMin);

    auto noiseBufferPtr = reinterpret_cast<float*>(d->noiseBuffer[plane]) + noiseBufferPtrOffset * d->noisePlaneStride[plane];
    for (int y = 0; y < h; ++y) {

        int x;
        for (x = 0; x < modw; x += pixelStep) {
            const __m128 noise = _mm_load_ps(noiseBufferPtr + x);
            const __m128 src = _mm_load_ps(dstp + x);
            const __m128 sum = _mm_add_ps(src, noise);
            const __m128 clamp = _mm_min_ps(_mm_max_ps(sum, mpixelMin), mpixelMax);
            _mm_store_ps(dstp + x, clamp);
        }

        for (; x < w; ++x) {
            dstp[x] = clamp(dstp[x] + noiseBufferPtr[x], pixelMin, pixelMax);
        }

        dstp += dstStride;
        noiseBufferPtr += d->noisePlaneStride[plane];
    }
}

#endif

static void VS_CC noiseInit(VSMap *in, VSMap *out, void **instanceData, VSNode* node, VSCore *core, const VSAPI *vsapi)
{
    NoiseData *d = reinterpret_cast<NoiseData*> (*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef *VS_CC noiseGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
    NoiseData *d = reinterpret_cast<NoiseData*> (*instanceData);

    if (activationReason == arInitial) {

        vsapi->requestFrameFilter(n, d->node, frameCtx);

    } else if (activationReason == arAllFramesReady) {

        auto src = vsapi->getFrameFilter(n, d->node, frameCtx);
        auto dst = vsapi->copyFrame(src, core);

        for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {

            if (!d->planes[plane])
                continue;

            size_t noiseBufferPtrOffset = 0;
            if (d->dyn)
                noiseBufferPtrOffset = d->noiseBufferStartRow[plane][n % d->noiseBufferStartRowSize];

            auto dstp = vsapi->getWritePtr(dst, plane);
            auto dstStride = vsapi->getStride(dst, plane) / d->vi->format->bytesPerSample;
            auto width = vsapi->getFrameWidth(src, plane);
            auto height = vsapi->getFrameHeight(src, plane);

#ifdef VS_TARGET_CPU_X86

            if (d->vi->format->sampleType == stInteger) {
                if (d->vi->format->bitsPerSample == 8)
                    addNoise_sse<uint8_t, int8_t>(dstp, dstStride, width, height, d, plane, noiseBufferPtrOffset);
                else
                    addNoise_sse<uint16_t, int16_t>(reinterpret_cast<uint16_t*>(dstp), dstStride, width, height, d, plane, noiseBufferPtrOffset);
            } else {
                addNoise_sse<float, float>(reinterpret_cast<float*>(dstp), dstStride, width, height, d, plane, noiseBufferPtrOffset);
            }

#else

            if (d->vi->format->sampleType == stInteger) {
                if (d->vi->format->bitsPerSample == 8)
                    addNoise_c<uint8_t, int8_t, int32_t>(dstp, dstStride, width, height, d, plane, noiseBufferPtrOffset);
                else
                    addNoise_c<uint16_t, int16_t, int32_t>(reinterpret_cast<uint16_t*>(dstp), dstStride, width, height, d, plane, noiseBufferPtrOffset);
            } else {
                addNoise_c<float, float, float>(reinterpret_cast<float*>(dstp), dstStride, width, height, d, plane, noiseBufferPtrOffset);
            }

#endif
        }
        vsapi->freeFrame(src);
        return dst;
    }
    return nullptr;
}

static void VS_CC noiseFree(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
    NoiseData *d = reinterpret_cast<NoiseData*> (instanceData);
    vsapi->freeNode(d->node);
    for (int plane = 0; plane < d->vi->format->numPlanes; ++plane)
        if (d->planes[plane])
            vs_aligned_free(d->noiseBuffer[plane]);
    delete d;
}

static void VS_CC noiseCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    NoiseData *d = new NoiseData();

    int err;

    d->node = vsapi->propGetNode(in, "clip", 0, 0);
    d->vi = vsapi->getVideoInfo(d->node);

    try {

        d->str = vsapi->propGetFloat(in, "str", 0, &err);
        if (err) d->str = 1.0f;

        if (d->str < 0.0 || d->str > 128.0)
            throw std::string("str must be 0.0 ... 128");

        d->limit = vsapi->propGetFloat(in, "limit", 0, &err);
        if (err) d->limit = 128.0f;

        if (d->limit < 0.0 || d->limit > 128.0)
            throw std::string("str must be 0.0 ... 128.0");


        d->type = vsapi->propGetInt(in, "type", 0, &err);
        if (err) d->type = NoiseTypeNormal;

        if (d->type < 1 || d->type > 2)
            throw std::string("type must be 1: uniform, 2: normal");

        d->mean = vsapi->propGetFloat(in, "mean", 0, &err);
        if (err) d->mean = 0.0f;

        d->var = vsapi->propGetFloat(in, "var", 0, &err);
        if (err) d->var = 1.0f;

        if (d->var <= 0.001)
            throw std::string("var must be larger than 0.001");

        d->dyn = !!vsapi->propGetInt(in, "dyn", 0, &err);
        if (err) d->dyn = true;

        d->full = !!vsapi->propGetInt(in, "full", 0, &err);
        if (err) d->full = false;
        if (d->vi->format->colorFamily == cmRGB) // always set true for RGB
            d->full = true;


        for (int i = 0; i < 3; ++i)
            d->planes[i] = false;

        int m = vsapi->propNumElements(in, "planes");

        if (m <= 0) {
            for (int i = 0; i < 3; ++i) {
                if (i == 0 || d->vi->format->colorFamily == cmRGB)
                    d->planes[i] = true;
            }
        } else {
            for (int i = 0; i < m; ++i) {
                int p = vsapi->propGetInt(in, "planes", i, &err);
                if (p < 0 || p > d->vi->format->numPlanes - 1)
                    throw std::string("planes index out of bound");
                d->planes[p] = true;
            }
        }

    } catch (std::string &errorMsg) {
        vsapi->freeNode(d->node);
        vsapi->setError(out, std::string("Noise: ").append(errorMsg).c_str());
        return;
    }

    if (d->vi->format->sampleType == stInteger) {
        for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
            if (d->full) {
                d->pixelMax[plane] = (1 << d->vi->format->bitsPerSample) - 1;
                d->pixelMin[plane] = 0;
            } else {
                if (plane == 0) {
                    d->pixelMax[plane] = 235 << (d->vi->format->bitsPerSample - 8);
                    d->pixelMin[plane] = 16 << (d->vi->format->bitsPerSample - 8);
                } else {
                    d->pixelMax[plane] = 240 << (d->vi->format->bitsPerSample - 8);
                    d->pixelMin[plane] = 16 << (d->vi->format->bitsPerSample - 8);
                }
            }
        }
    } else {
        for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
            if (plane == 0 || d->vi->format->colorFamily == cmRGB) {
                d->pixelMax[plane] = 1.0f;
                d->pixelMin[plane] = 0.0f;
            } else {
                d->pixelMax[plane] = 0.5f;
                d->pixelMin[plane] = -0.5f;
            }
        }
    }


    /* adjust str and limit for different sample type and bits */
    if (d->vi->format->sampleType == stInteger) {
        int64_t scale = 1 << (d->vi->format->bitsPerSample - 8);
        d->str *= scale;
        d->limit *= scale;
    } else {
        d->str *= (1.0 / 256.0);
        d->limit *= (1.0 / 256.0);
    }

    d->noisePlaneWidth[0] = d->vi->width;
    d->noisePlaneHeight[0] = d->vi->height;
    d->noisePlaneStride[0] = (d->vi->width + alignment - 1) & ~(alignment - 1);
    for (int plane = 1; plane < d->vi->format->numPlanes; ++plane) {
        d->noisePlaneWidth[plane] = d->vi->width >> d->vi->format->subSamplingW;
        d->noisePlaneHeight[plane] = d->vi->height >> d->vi->format->subSamplingH;
        d->noisePlaneStride[plane] = ((d->vi->width >> d->vi->format->subSamplingW) + alignment - 1) & ~(alignment - 1);
    }

    d->noiseBufferStartRowSize = 10 * d->vi->fpsNum / d->vi->fpsDen;

    std::random_device rd;
    std::mt19937_64 gen(rd());

    /* if use dynamic noise pattern for each frame,
       we generate a larger noiseBuffer.
       every frame will use random start point
       from the buffer to get different pattern */
    if (d->dyn) {
        for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
            d->noisePlaneHeight[plane] *= noisePlaneHeightMultiplier;
            d->noiseBufferStartRow[plane].resize(d->noiseBufferStartRowSize);

            std::uniform_int_distribution<> ud(0, d->noisePlaneHeight[plane] - d->noisePlaneHeight[plane] / noisePlaneHeightMultiplier);
            for (size_t i = 0; i < d->noiseBufferStartRowSize; ++i)
                d->noiseBufferStartRow[plane][i] = ud(gen);
        }
    }

    for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {

        if (!d->planes[plane])
            continue;

        if (d->vi->format->sampleType == stInteger) {
            if (d->vi->format->bitsPerSample == 8)
                genenateNoiseBuffer<int8_t>(gen, d, plane);
            else
                genenateNoiseBuffer<int16_t>(gen, d, plane);
        } else {
            genenateNoiseBuffer<float>(gen, d, plane);
        }
    }

    vsapi->createFilter(in, out, "noise", noiseInit, noiseGetFrame, noiseFree, fmParallel, 0, d, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
    configFunc("com.mio.noisegen", "noisegen", "VapourSynth Noise Generator", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Generate",
        "clip:clip;"
        "str:float:opt;"
        "limit:float:opt;"
        "type:int:opt;"
        "mean:float:opt;"
        "var:float:opt;"
        "dyn:int:opt;"
        "full:int:opt;"
        "planes:int[]:opt;",
        noiseCreate, nullptr, plugin);
}
