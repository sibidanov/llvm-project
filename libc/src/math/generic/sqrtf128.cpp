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

#include <fenv.h>
#include <x86intrin.h>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define __as_isnan(x) ((x.b[1]&(~0ul>>16))|x.b[0])
#define __as_issnan(x) (!(x.b[1]&(1l<<47)))
#define __as_snan2qnan(x) (x.b[1]|=(1l<<47))

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

  static inline u64 mhuu(u64 _a, u64 _b){
    return ((u128)_a*_b)>>64;
  }

  static inline i64 mhii(i64 x, i64 y){
    return ((i128)x*y)>>64;
  }

  static inline u128 mhuU(u64 y, u128 x){
    union int128_64_32 ux; ux.a = x;
    u128 xy0 = ux.b[0]*(u128)y;
    u128 xy1 = ux.b[1]*(u128)y;
    return xy1 + (xy0>>64);
  }

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

  static inline i128 mhUIm(u128 _a, i128 _b, u64 mask){
    union int128_64_32 sub; sub.a = _a;
    sub.b[0] &= mask;
    sub.b[1] &= mask;
    return mhUU(_a,_b) - sub.a;
  }

  static inline i128 mhIU(i128 _b, u128 _a){
    return mhUIm(_a,_b,(u64)(_b>>127));
  }

  LLVM_LIBC_FUNCTION(float128, sqrtf128, (float128 x)) {
    u32 flagp = _mm_getcsr(), oflagp = flagp;
    __m128i m = f128_as_m128i(x);
    int128_64_32 u;
    u.b[1] = m[1];
    u.b[0] = m[0];
    u32 e = u.b[1]>>48;
    i32 q2 = (e+1)>>1, i = (e+1)&1;
    i64 e2 = (q2+8191ul-1)<<48;
    if (unlikely((e-1)>=0x7ffe)){// all special cases NAN, inf, subnormals and negative numbers
      i32 sign = e>>15;
      e &= 0x7fff;
      if(unlikely(e==0x7fff)){// NAN or inifinity
	if(__as_isnan(u)){ // NAN
	  if(__as_issnan(u)){ // signaling nan
	    flagp |= FE_INVALID;
	    __as_snan2qnan(u); // transform snan to qnan
	  }
	  x = reinterpret_u128_as_f128(u.a);
	} else { //infinity
	  if (sign){
	    flagp |= FE_INVALID;
	    x = __builtin_nanf128("negsqrt");
	  }
	}
	if(flagp != oflagp) _mm_setcsr(flagp);
	return x;
      }
      if (unlikely((u.a<<1)==0)) return x; // signed zero
      if (unlikely(sign)){ // negative number
	_mm_setcsr(flagp|FE_INVALID);
	return __builtin_nanf128("negsqrt");
      }
      // denormal
      i32 ns = -15;
      if (u.b[1]){
	ns += __builtin_clzll(u.b[1]);
      } else {
	ns += __builtin_clzll(u.b[0]) + 64;
      }
      u.a <<= ns;
      e -= ns-122;
      u.b[1] ^= (u64)(ns&1)<<48;
      m = cpp::bit_cast<__m128i>(reinterpret_u128_as_f128(u.a));
      i = e&1;
      q2 = e>>1;
      e2 = (q2+8191ul-1-60)<<48;
    }

    const __m128i msk = {~0ull>>40, 0}, off = {0x81ll<<23, 0}, sexp = {0x1ffll<<23,0};
    m = _mm_srli_si128(m, 11);
    m = _mm_srli_epi64(m, 1);
    m = _mm_and_si128(m, msk);
    m = _mm_sub_epi32(m, off);
    m = _mm_xor_si128(m, sexp);
    __m128 mf = cpp::bit_cast<__m128>(m);
    mf = _mm_rsqrt_ss(mf); // get first approximation of reciprocal square root
    u.a <<= 16;
    m = cpp::bit_cast<__m128i>(mf);
    u32 r = (u32)m[0];
    r = (r<<8)|1ul<<31;
    u32 shft = 2 - i;
    u.a >>= shft;
    u.b[1] |= 1ul<<(62+i);

    u64 R = r, r2 = (u64)r*r;
    i64 h = mhuu(u.b[1],r2)<<2; // approximation correction
    u64 h2 = mhii(h,h);
    i64 h3 = (u128)h2*h>>64;
    u64 h4 = (h2>>16)*(h2>>16);
    h -= (h2*3>>2) - (h3*5>>3) + (35*h4>>38); // refine correction

    R<<=31;
    u64 dR = mhii(h,R);
    R <<= 1;
    R -= dR;

    u128 sx = mhuU(R, u.a);
    i128 H  = mhuU(R, sx)<<2;
    i64 hh = (i64)(H>>(14+32+2)), hh2 = 3*(hh*hh);
    H -= hh2>>(38-4);
    i128 ds = mhIU(H, sx);
    sx <<= 1;
    int128_64_32 v; v.a = sx - ds;

    u32 rm = flagp&_MM_ROUND_MASK, nrst = rm == _MM_ROUND_NEAREST;
    shft = 49 + nrst;
    i64 dd = (v.bs[0]<<shft)>>shft;

    if(unlikely(!(dd<-2||dd>5))){ // can round correctly?
      u128 m = v.a>>15, m2 = m*m;
      int128_64_32 t0, t1;
      t0.a = m2 - (u.a<<98);
      if(unlikely(t0.a==0)){
	v.a = m<<15;
      } else {
	t1 = t0;
	i64 sgn = t0.bs[1]>>63;
	t1.a -= (m<<1)^sgn;
	t1.a += 1 + sgn;
	if(unlikely(t1.a==0)){
	  v.a = (m - (2*sgn + 1))<<15;
	} else {
	  t1.a += t0.a;
	  i64 side = t1.bs[1]>>63;
	  v.b[0] &= ~0ul<<15;
	  v.a -= ((sgn&side)|(~sgn&1l))<<(15+side);
	  v.a |= 1; // sticky bit
	}
      }
    }

    u64 frac = v.b[0]&0x7ffful; // fractional part
    u64 rnd; // round bit
    if(likely(nrst)){
      rnd = frac>>14;  // round to nearest tie to even
    } else if(rm == _MM_ROUND_UP){
      rnd = !!frac; // round up
    } else if(rm == _MM_ROUND_DOWN){
      rnd = 0; // round down
    } else {
      rnd = 0; // round to zero
    }
    v.a >>= 15; // position mantissa
    v.a += rnd; // round

    // set inexact flag only if square root is really inexact or not already set
    if(likely(frac)) flagp |= FE_INEXACT;
    if(unlikely(oflagp != flagp)) _mm_setcsr(flagp);

    v.b[1] += e2; // place exponent
    return reinterpret_u128_as_f128(v.a); // put into xmm register
  }
} // namespace LIBC_NAMESPACE_DECL
