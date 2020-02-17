#ifndef __dynamic_hpp__
#define __dynamic_hpp__

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <sched.h>
#include <immintrin.h>
#include <set>
#include "pin.H"
extern "C" {
#include "xed-interface.h"
}

/**
 * Convert a FLT32 variable into the string representation of its value as an 8
 * digit hex number, padded with 0's.
 *
 * @param[in] fp Variable to convert.
 */
#define FLT32_TO_HEX(fp) \
    StringHex(*reinterpret_cast<const UINT32 *>(&fp), 8, FALSE)

using namespace LEVEL_CORE;

ofstream *out = 0;

// Register to use when RewriteMemOperand function
// is used
static REG scratchReg;

// Logging functions for __m512, __m256 and __mm128
#ifdef AVX512
template<class T>
inline void Log512(const __m512 &value)
{
    const size_t n = sizeof(__m512) / sizeof(T);
    T buffer[n];
    _mm512_storeu_ps((float*)buffer, value);

    for (unsigned int i = 0; i < n; i++)
        *out << "0x" << FLT32_TO_HEX(buffer[i]) << " ";

    *out << endl;
}
#endif

template<class T>
inline void Log256(const __m256 &value)
{
    const size_t n = sizeof(__m256) / sizeof(T);
    T buffer[n];
    _mm256_storeu_ps((float*)buffer, value);

    for (unsigned int i = 0; i < n; i++)
        *out << "0x" << FLT32_TO_HEX(buffer[i]) << " ";

    *out << endl;
}

template<class T>
inline void Log128(const __m128 &value)
{
    const size_t n = sizeof(__m128) / sizeof(T);
    T buffer[n];
    _mm_storeu_ps((float*)buffer, value);

    for (unsigned int i = 0; i < n; i++)
        *out << "0x" << FLT32_TO_HEX(buffer[i]) << " ";

    *out << endl;
}


// Operation modes
enum op_mode
{
    FMA_BF16,           // all bf16
    FMA_MP_FP32_WU_BN,  // baseline: bf16 + weight update (saxpy) in fp32 + FMA accumulator in 32b
    FMA_MP_FP32_WU,     // weight update (saxpy) in FP32
    FMA_MP,
    FMA_BF16_FP32_WU_BN,
    FMA_BF16_FP32_WU,
    NATIVE              // No instrumentation (detach)
};
op_mode mode;


// Function to check if a instruction is a
// floating point instruction and belongs
// to one of the instruction listed inside
// the conditional
bool isFP(OPCODE iclass)
{
    switch (iclass)
    {
    case XED_ICLASS_ADDSS:
        return true;
    case XED_ICLASS_SUBSS:
        return true;
    case XED_ICLASS_MULSS:
        return true;
    case XED_ICLASS_DIVSS:
        return true;
    case XED_ICLASS_ADDPS:
        return true;
    case XED_ICLASS_SUBPS:
        return true;
    case XED_ICLASS_MULPS:
        return true;
    case XED_ICLASS_DIVPS:
        return true;
    case XED_ICLASS_VADDSS:
        return true;
    case XED_ICLASS_VSUBSS:
        return true;
    case XED_ICLASS_VMULSS:
        return true;
    case XED_ICLASS_VDIVSS:
        return true;
    case XED_ICLASS_VADDPS:
        return true;
    case XED_ICLASS_VSUBPS:
        return true;
    case XED_ICLASS_VMULPS:
        return true;
    case XED_ICLASS_VDIVPS:
        return true;
    case XED_ICLASS_VFMADDSS:
        return true;
    case XED_ICLASS_VFMADDPS:
        return true;
    case XED_ICLASS_VFMADD132PS:
        return true;
    case XED_ICLASS_VFMADD213PS:
        return true;
    case XED_ICLASS_VFMADD231PS:
        return true;
    case XED_ICLASS_VFMADD132SS:
        return true;
    case XED_ICLASS_VFMADD213SS:
        return true;
    case XED_ICLASS_VFMADD231SS:
        return true;
    case XED_ICLASS_VFMADDSUB132PS:
        return true;
    case XED_ICLASS_VFMADDSUB213PS:
        return true;
    case XED_ICLASS_VFMADDSUB231PS:
        return true;
    case XED_ICLASS_VFMSUB132PS:
        return true;
    case XED_ICLASS_VFMSUB132SS:
        return true;
    case XED_ICLASS_VFMSUB213PS:
        return true;
    case XED_ICLASS_VFMSUB213SS:
        return true;
    case XED_ICLASS_VFMSUB231PS:
        return true;
    case XED_ICLASS_VFMSUB231SS:
        return true;
    case XED_ICLASS_VFMSUBADD132PS:
        return true;
    case XED_ICLASS_VFMSUBADD213PS:
        return true;
    case XED_ICLASS_VFMSUBADD231PS:
        return true;
    case XED_ICLASS_VFMSUBADDPS:
        return true;
    case XED_ICLASS_VFMSUBPS:
        return true;
    case XED_ICLASS_VFMSUBSS:
        return true;
    case XED_ICLASS_VFNMADD132PS:
        return true;
    case XED_ICLASS_VFNMADD132SS:
        return true;
    case XED_ICLASS_VFNMADD213PS:
        return true;
    case XED_ICLASS_VFNMADD213SS:
        return true;
    case XED_ICLASS_VFNMADD231PS:
        return true;
    case XED_ICLASS_VFNMADD231SS:
        return true;
    case XED_ICLASS_VFNMADDPS:
        return true;
    case XED_ICLASS_VFNMADDSS:
        return true;
    case XED_ICLASS_VFNMSUB132PS:
        return true;
    case XED_ICLASS_VFNMSUB132SS:
        return true;
    case XED_ICLASS_VFNMSUB213PS:
        return true;
    case XED_ICLASS_VFNMSUB213SS:
        return true;
    case XED_ICLASS_VFNMSUB231PS:
        return true;
    case XED_ICLASS_VFNMSUB231SS:
        return true;
    case XED_ICLASS_VFNMSUBPS:
        return true;
    case XED_ICLASS_VFNMSUBSS:
        return true;
    case XED_ICLASS_ADDSUBPS :
        return true;
    case XED_ICLASS_VADDSUBPS :
        return true;
    case XED_ICLASS_HADDPS :
        return true;
    case XED_ICLASS_VHADDPS :
        return true;
    case XED_ICLASS_HSUBPS :
        return true;
    case XED_ICLASS_VHSUBPS :
        return true;
    case XED_ICLASS_DPPS :
        return true;
    case XED_ICLASS_VDPPS :
        return true;
    case XED_ICLASS_RCPPS :
        return true;
    case XED_ICLASS_VRCPPS :
        return true;
    case XED_ICLASS_RSQRTPS :
        return true;
    case XED_ICLASS_VRSQRTPS :
        return true;
    case XED_ICLASS_RSQRTSS :
        return true;
    case XED_ICLASS_SQRTSS :
        return true;
    case XED_ICLASS_SQRTPS :
        return true;
    case XED_ICLASS_VSQRTPS :
        return true;
// AVX512 additional instructions
    case XED_ICLASS_V4FMADDPS:
        return true;
    case XED_ICLASS_V4FMADDSS:
        return true;
    case XED_ICLASS_V4FNMADDPS:
        return true;
    case XED_ICLASS_V4FNMADDSS:
        return true;
    }
    return false;
}

// Function to check if a instruction is a
// floating point instruction and belongs
// to one of the instruction listed inside
// the conditional
bool isFMA(OPCODE iclass)
{
    switch (iclass)
    {
    case XED_ICLASS_VFMADDSS:
    case XED_ICLASS_VFMADDPS:
    case XED_ICLASS_VFMADD132PS:
    case XED_ICLASS_VFMADD213PS:
    case XED_ICLASS_VFMADD231PS:
    case XED_ICLASS_VFMADD132SS:
    case XED_ICLASS_VFMADD213SS:
    case XED_ICLASS_VFMADD231SS:
    case XED_ICLASS_VFMADDSUB132PS:
    case XED_ICLASS_VFMADDSUB213PS:
    case XED_ICLASS_VFMADDSUB231PS:
    case XED_ICLASS_VFMSUB132PS:
    case XED_ICLASS_VFMSUB132SS:
    case XED_ICLASS_VFMSUB213PS:
    case XED_ICLASS_VFMSUB213SS:
    case XED_ICLASS_VFMSUB231PS:
    case XED_ICLASS_VFMSUB231SS:
    case XED_ICLASS_VFMSUBADD132PS:
    case XED_ICLASS_VFMSUBADD213PS:
    case XED_ICLASS_VFMSUBADD231PS:
    case XED_ICLASS_VFMSUBADDPS:
    case XED_ICLASS_VFMSUBPS:
    case XED_ICLASS_VFMSUBSS:
    case XED_ICLASS_VFNMADD132PS:
    case XED_ICLASS_VFNMADD132SS:
    case XED_ICLASS_VFNMADD213PS:
    case XED_ICLASS_VFNMADD213SS:
    case XED_ICLASS_VFNMADD231PS:
    case XED_ICLASS_VFNMADD231SS:
    case XED_ICLASS_VFNMADDPS:
    case XED_ICLASS_VFNMADDSS:
    case XED_ICLASS_VFNMSUB132PS:
    case XED_ICLASS_VFNMSUB132SS:
    case XED_ICLASS_VFNMSUB213PS:
    case XED_ICLASS_VFNMSUB213SS:
    case XED_ICLASS_VFNMSUB231PS:
    case XED_ICLASS_VFNMSUB231SS:
    case XED_ICLASS_VFNMSUBPS:
    case XED_ICLASS_VFNMSUBSS:
        return true;
    }
    return false;
}

bool isVFMA132(OPCODE iclass)
{
    switch (iclass)
    {
    case XED_ICLASS_VFMADD132PS:
    case XED_ICLASS_VFMADD132SS:
    case XED_ICLASS_VFMADDSUB132PS:
    case XED_ICLASS_VFMSUB132PS:
    case XED_ICLASS_VFMSUB132SS:
    case XED_ICLASS_VFMSUBADD132PS:
    case XED_ICLASS_VFNMADD132PS:
    case XED_ICLASS_VFNMADD132SS:
    case XED_ICLASS_VFNMSUB132PS:
    case XED_ICLASS_VFNMSUB132SS:
        return true;
    }
    return false;
}

bool isVFMA213(OPCODE iclass)
{
    switch (iclass)
    {
    case XED_ICLASS_VFMADDSS:
    case XED_ICLASS_VFMADDPS:
    case XED_ICLASS_VFMADD213PS:
    case XED_ICLASS_VFMADD213SS:
    case XED_ICLASS_VFMADDSUB213PS:
    case XED_ICLASS_VFMSUB213PS:
    case XED_ICLASS_VFMSUB213SS:
    case XED_ICLASS_VFMSUBADD213PS:
    case XED_ICLASS_VFMSUBADDPS:
    case XED_ICLASS_VFMSUBPS:
    case XED_ICLASS_VFMSUBSS:
    case XED_ICLASS_VFNMADD213PS:
    case XED_ICLASS_VFNMADD213SS:
    case XED_ICLASS_VFNMADDPS:
    case XED_ICLASS_VFNMADDSS:
    case XED_ICLASS_VFNMSUB213PS:
    case XED_ICLASS_VFNMSUB213SS:
    case XED_ICLASS_VFNMSUBPS:
    case XED_ICLASS_VFNMSUBSS:
        return true;
    }
    return false;
}

bool isVFMA231(OPCODE iclass)
{
    switch (iclass)
    {
    case XED_ICLASS_VFMADD231PS:
    case XED_ICLASS_VFMADD231SS:
    case XED_ICLASS_VFMADDSUB231PS:
    case XED_ICLASS_VFMSUB231PS:
    case XED_ICLASS_VFMSUB231SS:
    case XED_ICLASS_VFMSUBADD231PS:
    case XED_ICLASS_VFNMADD231PS:
    case XED_ICLASS_VFNMADD231SS:
    case XED_ICLASS_VFNMSUB231PS:
    case XED_ICLASS_VFNMSUB231SS:
        return true;
    }
    return false;
}

inline __m256 ToBFloatSimpleRNEVect32(__m256* input)
{
#ifdef LOGGING
    Log256<float>(*input);
#endif

    __m256i hulp = _mm256_set1_epi32(1 << 15);
    __m256i mask = _mm256_set1_epi32(0xFFFF0000);
    __m256i MSB_mask = _mm256_set1_epi32(0x80000000);

    // Save MSB bits set
    __m256i MSB_set = _mm256_and_si256(*(__m256i*)input, MSB_mask);

    // Set all MSB to 0
    __m256i tmp = _mm256_xor_si256(*(__m256i*)input, MSB_set);

    //AVX2 does not have unsigned 32b addition :(
    //Do the addition now that all MSB are cleared
    tmp = _mm256_add_epi32(tmp, hulp);

    // Reset MSB bits that were set to 1 originally
    tmp = _mm256_or_si256(tmp, MSB_set);

    // Truncate
    tmp = _mm256_and_si256(tmp, mask);

#ifdef LOGGING
    Log256<float>(*(__m256*)&tmp);
    *out << "-----------------" << endl;
#endif

    *input = *(__m256*)&tmp;
    return *input;
}

inline __m128 ToBFloatSimpleRNEVect16(__m128* input)
{
#ifdef LOGGING
    Log128<float>(*input);
#endif

    __m128i hulp = _mm_set1_epi32(1 << 15);
    __m128i mask = _mm_set1_epi32(0xFFFF0000);
    __m128i MSB_mask = _mm_set1_epi32(0x80000000);

    // Save MSB bits set
    __m128i MSB_set = _mm_and_si128(*(__m128i*)input, MSB_mask);

    // Set all MSB to 0
    __m128i tmp = _mm_xor_si128(*(__m128i*)input, MSB_set);

    //AVX2 does not have unsigned 32b addition :(
    //Do the addition now that all MSB are cleared
    tmp = _mm_add_epi32(tmp, hulp);

    // Reset MSB bits that were set to 1 originally
    tmp = _mm_or_si128(tmp, MSB_set);

    // Truncate
    tmp = _mm_and_si128(tmp, mask);

#ifdef LOGGING
    Log128<float>(*(__m128*)&tmp);
    *out << "-----------------" << endl;
#endif

    *input = *(__m128*)&tmp;
    return *input;
}

// Truncate without rounding
inline __m256 ToBFloatTruncVect32(__m256* input)
{
#ifdef LOGGING
    Log256<float>(*input);
#endif

    __m256i mask = _mm256_set1_epi32(0xFFFF0000);

    __m256i tmp = *(__m256i*)input;
    tmp = _mm256_and_si256(tmp, mask);

#ifdef LOGGING
    Log256<float>(*(__m256*)&tmp);
    *out << "-----------------" << endl;
#endif

    *input = *(__m256*)&tmp;
    return *input;
}

// Truncate without rounding
inline __m128 ToBFloatTruncVect16(__m128* input)
{
#ifdef LOGGING
    Log128<float>(*input);
#endif

    __m128i mask = _mm_set1_epi32(0xFFFF0000);

    __m128i tmp = *(__m128i*)input;
    tmp = _mm_and_si128(tmp, mask);

#ifdef LOGGING
    Log128<float>(*(__m128*)&tmp);
    *out << "-----------------" << endl;
#endif

    *input = *(__m128*)&tmp;
    return *input;
}


// This union is used to make the conversion process
// to BF16 using the rounding method introduced by
// TensorFlow.
union FP32
{
    unsigned int u;
    float f;
};

inline FLT32 ToBFloatSimpleRNE(float floatNumber)
{
    uint32_t input;
    FP32 f;
    f.f = floatNumber;
    input = f.u;
    float output;
    uint32_t hulp = 1 << 15;
    uint32_t mask = 0xFFFF0000;

    uint32_t tmp = (input + hulp) & mask;
    output = *(float*)&tmp;
    return output;
}

// Direct truncation method. Intel paper uses this
// truncation method to test purposes, works but
// lose some precision. However the models converge
inline FLT32 ToBFloatTrunc(float floatNumber)
{
    int temp;
    temp = *((int*)&floatNumber);
    temp = (temp & 0xFFFF0000);
    floatNumber = *(float*)&temp;
    return floatNumber;
}

// Vectorized version as in tensorflow
// https://github.com/tensorflow/tensorflow/blob/0ff7955a0c1a42e2767afb0a5cc202dfe4d6ff19/tensorflow/core/lib/bfloat16/bfloat16.h#L184
// Return a __m512 to avoid vzeroupper instruction
#ifdef AVX512
inline __m512 ToBFloatTensorFlowVect64 (__m512* input) {

#ifdef LOGGING
    Log512<float>(*input);
#endif

    __m512i MSB_mask = _mm512_set1_epi32(0x80000000);
    __m512i LSB_mask = _mm512_set1_epi32(1);
    __m512i mask = _mm512_set1_epi32(0xFFFF0000);
    __m512i qnan_mask = _mm512_set1_epi32(0x7FC00000);
    __m512i rounding_mask = _mm512_set1_epi32(0x7FFF);

    // shift + get LSB bits + generate rounding bias
    __m512i tmp = _mm512_srli_epi32(*(__m512i*)input, 16);
    tmp = _mm512_and_si512(tmp, LSB_mask);
    __m512i rounding_bias = _mm512_add_epi32(tmp, rounding_mask);

    // Save MSB bits set
    __m512i MSB_set = _mm512_and_si512(*(__m512i*)input, MSB_mask);

    // Set all MSB to 0
    tmp = _mm512_xor_si512(*(__m512i*)input, MSB_set);

    //AVX does not have unsigned 32b addition :(
    //Do the addition now that all MSB are cleared
    tmp = _mm512_add_epi32(tmp, rounding_bias);

    // Reset MSB bits that were set to 1 originally
    tmp = _mm512_or_si512(tmp, MSB_set);
    // is nan? Use ordered comparison, if both NaN returns false, true otherwise
    __mmask16 not_isnan_mask = _mm512_cmp_ps_mask(*input, *input, _CMP_EQ_OQ);

    // Truncate and insert qNaN if input was NaN
    tmp = _mm512_mask_and_epi32(qnan_mask, not_isnan_mask, tmp, mask);

    // Return value to avoid vzeroupper
    *input = *(__m512*)&tmp;
#ifdef LOGGING
    Log512<float>(*input);
    *out << "-----------------" << endl;
#endif
    return *input;
}
#endif

// Vectorized version as in tensorflow
// https://github.com/tensorflow/tensorflow/blob/0ff7955a0c1a42e2767afb0a5cc202dfe4d6ff19/tensorflow/core/lib/bfloat16/bfloat16.h#L184
// Return a __m256 to avoid vzeroupper instruction
inline __m256 ToBFloatTensorFlowVect32 (__m256* input) {

#ifdef LOGGING
    Log256<float>(*input);
#endif

    __m256i MSB_mask = _mm256_set1_epi32(0x80000000);
    __m256i LSB_mask = _mm256_set1_epi32(1);
    __m256i mask = _mm256_set1_epi32(0xFFFF0000);
    __m256i qnan_mask = _mm256_set1_epi32(0x7FC00000);
    __m256i rounding_mask = _mm256_set1_epi32(0x7FFF);

    // shift + get LSB bits + generate rounding bias
    __m256i tmp = _mm256_srli_epi32(*(__m256i*)input, 16);
    tmp = _mm256_and_si256(tmp, LSB_mask);
    __m256i rounding_bias = _mm256_add_epi32(tmp, rounding_mask);

    // Save MSB bits set
    __m256i MSB_set = _mm256_and_si256(*(__m256i*)input, MSB_mask);

    // Set all MSB to 0
    tmp = _mm256_xor_si256(*(__m256i*)input, MSB_set);

    //AVX2 does not have unsigned 32b addition :(
    //Do the addition now that all MSB are cleared
    tmp = _mm256_add_epi32(tmp, rounding_bias);

    // Reset MSB bits that were set to 1 originally
    tmp = _mm256_or_si256(tmp, MSB_set);

    // Truncate
    tmp = _mm256_and_si256(tmp, mask);

    // is nan? Use ordered comparison, if both NaN returns false, true otherwise
    __m256 not_isnan_mask = _mm256_cmp_ps(*input, *input, _CMP_EQ_OQ);

    // negate not_isnan_mask
    __m256i isnan_mask = ~(*(__m256i*)&not_isnan_mask);

    // clear nans from current result, and clear non_nans from qnan_mask
    tmp = _mm256_and_si256(tmp, *(__m256i*)&not_isnan_mask);
    qnan_mask = _mm256_and_si256(qnan_mask, isnan_mask);

    // Merge to apply changes
    tmp = _mm256_or_si256(tmp, qnan_mask);

    // Return value to avoid vzeroupper
    *input = *(__m256*)&tmp;
#ifdef LOGGING
    Log256<float>(*input);
    *out << "-----------------" << endl;
#endif
    return *input;
}

// Vectorized version as in tensorflow
inline __m128 ToBFloatTensorFlowVect16 (__m128* input) {

#ifdef LOGGING
    Log128<float>(*input);
#endif
    __m128i MSB_mask = _mm_set1_epi32(0x80000000);
    __m128i LSB_mask = _mm_set1_epi32(1);
    __m128i mask = _mm_set1_epi32(0xFFFF0000);
    __m128i qnan_mask = _mm_set1_epi32(0x7FC00000);
    __m128i rounding_mask = _mm_set1_epi32(0x7FFF);

    // shift + get LSB bits + generate rounding bias
    __m128i tmp = _mm_srli_epi32(*(__m128i*)input, 16);
    tmp = _mm_and_si128(tmp, LSB_mask);
    __m128i rounding_bias = _mm_add_epi32(tmp, rounding_mask);

    // Save MSB bits set
    __m128i MSB_set = _mm_and_si128(*(__m128i*)input, MSB_mask);

    // Set all MSB to 0
    tmp = _mm_xor_si128(*(__m128i*)input, MSB_set);

    //AVX2 does not have unsigned 32b addition :(
    //Do the addition now that all MSB are cleared
    tmp = _mm_add_epi32(tmp, rounding_bias);

    // Reset MSB bits that were set to 1 originally
    tmp = _mm_or_si128(tmp, MSB_set);

    // Truncate
    tmp = _mm_and_si128(tmp, mask);

    // is nan? Use ordered comparison, if both NaN returns false, true otherwise
    __m128 not_isnan_mask = _mm_cmp_ps(*input, *input, _CMP_EQ_OQ);

    // negate not_isnan_mask
    __m128i isnan_mask = ~(*(__m128i*)&not_isnan_mask);

    // clear nans from tmp, and clear non_nans from qnan_mask
    tmp = _mm_and_si128(tmp, *(__m128i*)&not_isnan_mask);
    qnan_mask = _mm_and_si128(qnan_mask, isnan_mask);

    // Merge to apply changes
    tmp = _mm_or_si128(tmp, qnan_mask);

    // Return value to avoid vzeroupper
    *input = *(__m128*)&tmp;
#ifdef LOGGING
    Log128<float>(*input);
    *out << "-----------------" << endl;
#endif
    return *input;
}

// This is the rounding method used by TensorFlow
FLT32 ToBFloatTensorFlow(float floatNumber){
    uint32_t input;
    FP32 f;
    f.f = floatNumber;
    input = f.u;

#ifdef LOGGING
    *out << "0x" << FLT32_TO_HEX(floatNumber) << " " << endl;
#endif

    uint32_t lsb = (input >> 16) & 1;
    uint32_t rounding_bias = 0x7fff + lsb;
    input += rounding_bias;

    int32_t temp = static_cast<int16_t>(input>>16);
    temp = temp << 16;

    // If the value is a NaN, squash it to a qNaN with msb of fraction set,
    // this makes sure after truncation we don't end up with an inf.
    if(isnan(floatNumber))
        temp = 0x7FC00000;
    floatNumber = *(float*)&temp;
#ifdef LOGGING
    *out << "0x" << FLT32_TO_HEX(floatNumber) << " " << endl;
    *out << "-----------------" << endl;
#endif
    return floatNumber;

}
#endif
