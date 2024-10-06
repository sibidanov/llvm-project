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

    // For the initial approximation of the reciprocal square root we
    // use the SSE instruction rsqrtss. First, the argument is
    // truncated from float128 to float32 inside SIMD registers so
    // data do not travel between SIMD and GP registers.
    const __m128i msk = {~0ull>>40, 0}, off = {0x81ll<<23, 0}, sexp = {0x1ffll<<23,0};
    // shift right by 11*8 bits since only byte granularity is
    // avaliable for a full width xmms register
    m = _mm_srli_si128(m, 11);
    // right shift by 1 bit so 23 upper bits of mantissa fills target float32
    m = _mm_srli_epi64(m, 1);
    // clear upper 8 bits of float32 and retain the last bit of the exponent
    m = _mm_and_si128(m, msk);
    // subtract the offset
    m = _mm_sub_epi32(m, off);
    // flip exponent bits so now the number is in [1,4) range
    m = _mm_xor_si128(m, sexp);
    // cast to floating point format
    __m128 mf = cpp::bit_cast<__m128>(m);
    mf = _mm_rsqrt_ss(mf); // get first approximation of reciprocal square root
    // cast the reciprocal in (0.5,1] to the integer domain
    m = cpp::bit_cast<__m128i>(mf);
    // move the reciprocal to GPR
    u32 r = (u32)m[0];
    // add implict bit to the reciprocal and shift left to fill 32
    // bits (the exponent bits at this stage are not needed)
    r = (r<<8)|1ul<<31;
    u.b[1] &= ~0ul>>16; // clear the exponent
    // shift mantissa depend on the exponent parity
    u.a <<= 14+i;
    // add the implict bit
    u.b[1] |= 1ul<<(62+i);

    // Fourth order Newton iteration to have almost 60 bit precision
    // reciprocal.  Let r is a reciprocal approximation then h =
    // r^2*x-1 and a better approximation is 1/sqrt(x) ~= r - r*(1/2*h
    // - 3/8*h^2 + 5/16*h^3 - 35/128*h^4 + ...)
    u64 R = r, r2 = (u64)r*r;
    i64 h = mhuu(u.b[1],r2)<<2; // first order correction
    u64 h2 = mhii(h,h);
    i64 h3 = (u128)h2*h>>64;
    u64 h4 = (h2>>16)*(h2>>16);
    h -= (h2*3>>2) - (h3*5>>3) + (35*h4>>38); // refine correction

    R<<=31;
    u64 dR = mhii(h,R);
    R <<= 1;
    R -= dR; // now we have ~60 bit precision reciprocal

    // if R is an approximation of the reciprocal square root then the
    // square root itself is sqrt(x) ~= R*x. A better estimation need
    // additional refinement. Let H = R^2*x-1 then sqrt(x) = R*x -
    // R*x*(1/2*h - 3/8*h^2 + ...)
    //
    // first approximation of the square root
    u128 sx = mhuU(R, u.a);
    // H = R^2*x - 1
    i128 H  = mhuU(R, sx)<<2;
    // for the second order corrrection only several bits of H^2 is enough so it fits in 64 bits
    i64 hh = (i64)(H>>(14+32+2)), hh2 = 3*(hh*hh);
    H -= hh2>>(38-4);
    // correction for the square root itself
    i128 ds = mhIU(H, sx);
    // adjust position
    sx <<= 1;
    // the square root with 125 bit precision is ready
    int128_64_32 v; v.a = sx - ds;

    u32 nrst = rm == FE_TONEAREST;
    // the result lies within (-2,5) of true square root so we now
    // test that we can correctly round the result taking into account
    // the rounding mode
    u64 dd = ((v.b[0]^(nrst<<14)) + 2)&0x7fff; // how close the result to the rounded value
    if(unlikely(dd<8)){ // can round correctly?
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
    if(frac) fputil::raise_except_if_required(FE_INEXACT);

    v.b[1] += e2; // place exponent
    return reinterpret_u128_as_f128(v.a); // put into xmm register
  }
} // namespace LIBC_NAMESPACE_DECL
