//===-- Implementation of sqrtf128 function -------------------------------===//
//
// Copyright (c) 2024 Alexei Sibidanov <sibid@uvic.ca>
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sqrtf128.h"
#include "src/__support/common.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/uint128.h"
#include "src/__support/macros/properties/cpu_features.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/FEnvImpl.h"

#include <x86intrin.h>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

namespace LIBC_NAMESPACE_DECL {
  typedef UInt128 u128;
  typedef Int128 i128;
  typedef uint64_t u64;
  typedef int64_t i64;
  typedef uint32_t u32;
  typedef int32_t i32;

  union int128_64_32{
    u128 a;
    i128 as;
    u64 b[2];
    i64 bs[2];
    u32 c[4];
    i32 cs[4];
    float128 f;
    __m128i x;
  };

  float128 reinterpret_u128_as_f128(u128 x){
#ifdef LIBC_TARGET_CPU_HAS_SSE4_2
    u64 x0 = (u64)x, x1 = (u64)(x>>64);
    __m128i m = {0, 0};
    m = _mm_insert_epi64 (m, x0, 0);
    m = _mm_insert_epi64 (m, x1, 1);
    return cpp::bit_cast<float128>(m);
#else
    return cpp::bit_cast<float128>(x);
#endif
  }

  static inline __m128i f128_as_m128i(float128 z){
    return cpp::bit_cast<__m128i>(z);
  }

  // get high part of unsigned 64x64 bit multiplication
  static inline u64 mhuu(u64 _a, u64 _b){
    return ((u128)_a*_b)>>64;
  }

  // get high part of signed 64x64 bit multiplication
  static inline i64 mhii(i64 x, i64 y){
    return ((i128)x*y)>>64;
  }

  // get high 128 bit part of unsigned 64x128 bit multiplication
  static inline u128 mhuU(u64 y, u128 x){
    union int128_64_32 ux; ux.a = x;
    u128 xy0 = ux.b[0]*(u128)y;
    u128 xy1 = ux.b[1]*(u128)y;
    return xy1 + (xy0>>64);
  }

  // get high 128 bit part of unsigned 128x128 bit multiplication
  static inline u128 mhUU(u128 _a, u128 _b){
    int128_64_32 a, b, a1b0, a0b1, a1b1;
    a.a = _a;
    b.a = _b;
    a1b0.a = (u128)a.b[1]*b.b[0];
    a0b1.a = (u128)a.b[0]*b.b[1];
    a1b1.a = (u128)a.b[1]*b.b[1];
    a1b1.a += a1b0.b[1];
    a1b1.a += a0b1.b[1];
    return a1b1.a;
  }

  // get high 128 bit part of (unsigned 128)x(signed 128) bit
  // multiplication with sign mask
  static inline i128 mhUIm(u128 _a, i128 _b, u64 mask){
    union int128_64_32 sub; sub.a = _a;
    sub.b[0] &= mask;
    sub.b[1] &= mask;
    return mhUU(_a,_b) - sub.a;
  }

  // get high 128 bit part of (unsigned 128)x(signed 128) bit
  // multiplication
  static inline i128 mhIU(i128 _b, u128 _a){
    return mhUIm(_a,_b,(u64)(_b>>127));
  }

  // for low precision approxmation use 64x64->64 bit multiplication
  static inline u64 rsqrt9(u64 m){
    static const unsigned c[][4] = {
      {0xffffffff, 0xfffff780, 0xbff55815, 0x9bb5b6e7}, {0xfc0bd889, 0xfa1d6e7d, 0xb8a95a89, 0x938bf8f0},
      {0xf82ec882, 0xf473bea9, 0xb1bf4705, 0x8bed0079}, {0xf467f280, 0xeefff2a1, 0xab309d4a, 0x84cdb431},
      {0xf0b6848c, 0xe9bf46f4, 0xa4f76232, 0x7e24037b}, {0xed19b75e, 0xe4af2628, 0x9f0e1340, 0x77e6ca62},
      {0xe990cdad, 0xdfcd2521, 0x996f9b96, 0x720db8df}, {0xe61b138e, 0xdb16ffde, 0x94174a00, 0x6c913cff},
      {0xe2b7dddf, 0xd68a967b, 0x8f00c812, 0x676a6f92}, {0xdf6689b7, 0xd225ea80, 0x8a281226, 0x62930308},
      {0xdc267bea, 0xcde71c63, 0x8589702c, 0x5e05343e}, {0xd8f7208e, 0xc9cc6948, 0x81216f2e, 0x59bbbcf8},
      {0xd5d7ea91, 0xc5d428ee, 0x7cecdb76, 0x55b1c7d6}, {0xd2c8534e, 0xc1fccbc9, 0x78e8bb45, 0x51e2e592},
      {0xcfc7da32, 0xbe44d94a, 0x75124a0a, 0x4e4b0369}, {0xccd6045f, 0xbaaaee41, 0x7166f40f, 0x4ae66284},
      {0xc9f25c5c, 0xb72dbb69, 0x6de45288, 0x47b19045}, {0xc71c71c7, 0xb3cc040f, 0x6a882804, 0x44a95f5f},
      {0xc453d90f, 0xb0849cd4, 0x67505d2a, 0x41cae1a0}, {0xc1982b2e, 0xad566a85, 0x643afdc8, 0x3f13625c},
      {0xbee9056f, 0xaa406113, 0x6146361f, 0x3c806169}, {0xbc46092e, 0xa7418293, 0x5e70506d, 0x3a0f8e8e},
      {0xb9aedba5, 0xa458de58, 0x5bb7b2b1, 0x37bec572}, {0xb72325b7, 0xa1859022, 0x591adc9a, 0x358c09e2},
      {0xb4a293c2, 0x9ec6bf52, 0x569865a7, 0x33758476}, {0xb22cd56d, 0x9c1b9e36, 0x542efb6a, 0x31797f8a},
      {0xafc19d86, 0x9983695c, 0x51dd5ffb, 0x2f96647a}, {0xad60a1d1, 0x96fd66f7, 0x4fa2687c, 0x2dcab91f},
      {0xab099ae9, 0x9488e64b, 0x4d7cfbc9, 0x2c151d8a}, {0xa8bc441a, 0x92253f20, 0x4b6c1139, 0x2a7449ef},
      {0xa6785b42, 0x8fd1d14a, 0x496eaf82, 0x28e70cc3}, {0xa43da0ae, 0x8d8e042a, 0x4783eba7, 0x276c4900},
      {0xa20bd701, 0x8b594648, 0x45aae80a, 0x2602f493}, {0x9fe2c315, 0x89330ce4, 0x43e2d382, 0x24aa16ec},
      {0x9dc22be4, 0x871ad399, 0x422ae88c, 0x2360c7af}, {0x9ba9da6c, 0x85101c05, 0x40826c88, 0x22262d7b},
      {0x99999999, 0x83126d70, 0x3ee8af07, 0x20f97cd2}, {0x97913630, 0x81215480, 0x3d5d0922, 0x1fd9f714},
      {0x95907eb8, 0x7f3c62ef, 0x3bdedce0, 0x1ec6e994}, {0x93974369, 0x7d632f45, 0x3a6d94a9, 0x1dbfacbb},
      {0x91a55615, 0x7b955498, 0x3908a2be, 0x1cc3a33b}, {0x8fba8a1c, 0x79d2724e, 0x37af80bf, 0x1bd23960},
      {0x8dd6b456, 0x781a2be4, 0x3661af39, 0x1aeae458}, {0x8bf9ab07, 0x766c28ba, 0x351eb539, 0x1a0d21a2},
      {0x8a2345cc, 0x74c813dd, 0x33e61feb, 0x19387676}, {0x88535d90, 0x732d9bdc, 0x32b7823a, 0x186c6f3e},
      {0x8689cc7e, 0x719c7297, 0x3192747d, 0x17a89f21}, {0x84c66df1, 0x70144d19, 0x30769424, 0x16ec9f89},
      {0x83091e6a, 0x6e94e36c, 0x2f63836f, 0x16380fbf}, {0x8151bb87, 0x6d1df079, 0x2e58e925, 0x158a9484},
      {0x7fa023f1, 0x6baf31de, 0x2d567053, 0x14e3d7ba}, {0x7df43758, 0x6a4867d3, 0x2c5bc811, 0x1443880e},
      {0x7c4dd664, 0x68e95508, 0x2b68a346, 0x13a958ab}, {0x7aace2b0, 0x6791be86, 0x2a7cb871, 0x131500ee},
      {0x79113ebc, 0x66416b95, 0x2997c17a, 0x12863c29}, {0x777acde8, 0x64f825a1, 0x28b97b82, 0x11fcc95c},
      {0x75e9746a, 0x63b5b822, 0x27e1a6b4, 0x11786b03}, {0x745d1746, 0x6279f081, 0x2710061d, 0x10f8e6da},
      {0x72d59c46, 0x61449e06, 0x26445f86, 0x107e05ac}, {0x7152e9f4, 0x601591be, 0x257e7b4d, 0x10079327},
      {0x6fd4e793, 0x5eec9e6b, 0x24be2445, 0x0f955da9}, {0x6e5b7d16, 0x5dc9986e, 0x24032795, 0x0f273620},
      {0x6ce6931d, 0x5cac55b7, 0x234d5496, 0x0ebcefdb}, {0x6b7612ec, 0x5b94adb2, 0x229c7cbc, 0x0e56606e},
    };
    // The range [1,2] is splitted into 64 equal sub-ranges and the
    // reciprocal square root is approximated by a cubic polynomial by
    // the minimax method in each subrange. The approximation accuracy
    // fits into 32-33 bits and thus it is natural to round
    // coefficients into 32 bit. The constant coefficient can be
    // rounded to 33 bits since the most significant bit is always 1
    // and implicitly assumed in the table.
    u64 indx = m>>58; // subrange index 
    u64 c3 = c[indx][3], c0 = c[indx][0], c1 = c[indx][1], c2 = c[indx][2];
    c0 <<= 31; // to 64 bit with the space for the implicit bit
    c0 |= 1ul<<63; // add implicit bit
    c1 <<= 25; // to 64 bit format
    u64 d = (m<<6)>>32; // local coordinate in the subrange [0, 2^32]
    u64 d2 = ((u64)(d*d))>>32; // square of the local coordinate
    u64 re = c0 + (d2*c2>>13); // even part of the polynomial (positive)
    u64 ro = d*((c1 + ((d2*c3)>>19))>>26)>>6; // odd part of the polynomial (negative)
    u64 r = re - ro; // maximal error < 1.55e-10 and it is less than 2^-32
    // Newton-Raphson first order step to improve accuracy of the result to almost 64 bits
    // r1 = r0 - r0*(r0^2*x - 1)/2
    u64 r2 = mhuu(r,r);
    i64 h = mhuu(m,r2) + r2; // h = r0^2*x - 1
    i64 hr = mhii(h,r>>1); // r0*h/2
    r -= hr;
    if(unlikely(!r)) r--; // adjust in the unlucky case x~1
    return r;
  }

  LLVM_LIBC_FUNCTION(float128, sqrtf128, (float128 x)) {
    using FPBits = fputil::FPBits<float128>;
    // get status register (rounding mode is there)
    u32 rm = fputil::get_round();
    // cast float128 to __m128i it is just a type change without actial data transfer
    __m128i m = f128_as_m128i(x);
    int128_64_32 u;
    // here SIMD data moves to GP registers
    u.b[1] = m[1];
    u.b[0] = m[0];
    i32 e = u.b[1]>>48; // exponent
    if(unlikely(e==0)){ // x is subnormal or x=+0
      i32 ns = -15;
      if(u.b[1]){
	ns += __builtin_clzll(u.b[1]);
      } else {
	if(u.b[0]) {
	  ns += __builtin_clzll(u.b[0]) + 64;
	} else
	  return x; // x = +0
      }
      e = 1 - ns;
      u.a <<= ns; // normalize mantissa
      u.b[1] ^= (u64)(ns&1)<<48; // set proper last bit of exponent
      m = u.x;
    }
    if(unlikely(e>=0x7fff)){// other special cases: NAN, inf, negative numbers
      FPBits xbits(x);
      if(xbits.is_zero() || xbits == xbits.inf()) return x; // x = -0 or x = inf
      if(xbits.is_nan()){ // x is nan
	if(xbits.is_quiet_nan()) return x; // pass through quiet nan
	return xbits.quiet_nan().get_val(); // transform signaling nan to quiet and return
      }
      // x<0 or x=-inf
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return xbits.quiet_nan().get_val();
    }

    e++; // adjust parity
    i32 q2 = e>>1, i = e&1;
    // exponent of the final result
    i64 e2 = (q2+8191ul-1)<<48;

    u.a <<= 16;
    const u64 rsqrt_2[] = {~0ul,0xb504f333f9de6484}; // 2^64/sqrt(2)
    u64 rx = u.b[1], r = rsqrt9(rx);
    u128 r2 = (u128)r*rsqrt_2[i];
    unsigned shft = 2-i;
    u.a >>= shft;
    u.b[1] |= 1ul<<(62+i);
    r = r2>>64;
    u128 sx = mhuU(r, u.a);
    i128 h  = mhuU(r, sx)<<2, ds = mhIU(h, sx);
    sx <<= 1;
    int128_64_32 v; v.a = sx - ds;
    u32 nrst = rm == FE_TONEAREST;
    shft = 49 + nrst;
    i64 dd = (v.bs[0]<<shft)>>shft;
    if(unlikely(!(dd<-8||dd>3))){ // can round correctly?
      // m is almost the final result it can be only 1 ulp off so we
      // just need to test both possibilities. We square it and
      // compare with the initial argument.
      u128 m = v.a>>15, m2 = m*m;
      int128_64_32 t0, t1;
      // the difference of the squared result and the argument
      t0.a = m2 - (u.a<<98);
      if(unlikely(t0.a==0)){
	// the square root is exact
	v.a = m<<15;
      } else {
	// add +-1 ulp to m depend on the sign of the difference. Here
	// we do not need to square again since (m+1)^2 = m^2 + 2*m +
	// 1 so just need to add shifted m and 1.
	t1 = t0;
	i64 sgn = t0.bs[1]>>63; // sign of the difference
	t1.a -= (m<<1)^sgn;
	t1.a += 1 + sgn;

	i64 sgn1 = t1.bs[1]>>63;
	if(unlikely(sgn == sgn1)){
	  t0 = t1;
	  v.a -= sgn<<15;
	  t1.a -= (m<<1)^sgn;
	  t1.a += 1 + sgn;
	}

	if(unlikely(t1.a==0)){
	  // 1 ulp offset brings again an exact root
	  v.a = (m - (2*sgn + 1))<<15;
	} else {
	  t1.a += t0.a;
	  i64 side = t1.bs[1]>>63; // select what is closer m or m+-1
	  v.b[0] &= ~0ul<<15; // wipe the fractional bits
	  v.a -= ((sgn&side)|(~sgn&1l))<<(15+side);
	  v.a |= 1; // add sticky bit since we cannot have an exact mid-point situation
	}
      }
    }

    u64 frac = v.b[0]&0x7ffful; // fractional part
    u64 rnd; // round bit
    if(likely(nrst)){
      rnd = frac>>14;  // round to nearest tie to even
    } else if(rm == FE_UPWARD){
      rnd = !!frac; // round up
    } else if(rm == FE_DOWNWARD){
      rnd = 0; // round down
    } else {
      rnd = 0; // round to zero
    }
    v.a >>= 15; // position mantissa
    v.a += rnd; // round

    // set inexact flag only if square root is inexact
    //    if(frac) fputil::raise_except_if_required(FE_INEXACT);

    v.b[1] += e2; // place exponent
    return reinterpret_u128_as_f128(v.a); // put into xmm register
  }
} // namespace LIBC_NAMESPACE_DECL
