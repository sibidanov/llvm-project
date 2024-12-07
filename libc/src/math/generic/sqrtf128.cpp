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
    static const u64 c[][2] = {
      {0xffffffff7848373eul, 0x01ffffef0133d347ul}, {0xfe05ec4488ed02e5ul, 0x01f43adcfb4b1581ul},
      {0xfc176440ea358a60ul, 0x01e8e77d522cf376ul}, {0xfa33f94036efa6a0ul, 0x01ddffe5430c9859ul},
      {0xf85b42462df88906ul, 0x01d37e8de98e3abbul}, {0xf68cdbaedb54a44aul, 0x01c95e4c515be4feul},
      {0xf4c866d6500b3d58ul, 0x01bf9a4a4394f4b2ul}, {0xf30d89c7370219b6ul, 0x01b62dffbc528b6dul},
      {0xf15beeefa7aafc3bul, 0x01ad152cf6ce9ebdul}, {0xefb344dba7bdfcdful, 0x01a44bd500c19009ul},
      {0xee133df4dbc0e158ul, 0x019bce38c74e23daul}, {0xec7b9046f2f9fefdul, 0x019398d2915e91dbul},
      {0xeaebf54866d9df85ul, 0x018ba851dcaf174aul}, {0xe96429a7300fa501ul, 0x0183f99793f1a3a9ul},
      {0xe7e3ed191c8893a3ul, 0x017c89b2958097a0ul}, {0xe66b022f79ad86c6ul, 0x017555dc83010088ul},
      {0xe4f92e2dcd6734caul, 0x016e5b76d3236cd8ul}, {0xe38e38e35ee5ebedul, 0x016798081f68fd6dul},
      {0xe229ec8755f177cdul, 0x01610939a873ca82ul}, {0xe0cc15973cb5db88ul, 0x015aacd50bf8b9d8ul},
      {0xdf7482b7b4aabedcul, 0x015480c227e7ca2bul}, {0xde2304973364f158ul, 0x014e830526d09a66ul},
      {0xdcd76dd29fe88038ul, 0x0148b1bcb1ed4a4cul}, {0xdb9192dbac7a3125ul, 0x01430b2045973cdcul},
      {0xda5149e0cc030903ul, 0x013d8d7ea5390f11ul}, {0xd9166ab6a4e0e528ul, 0x0138373c6c18624ful},
      {0xd7e0cec2e5841468ul, 0x013306d2b891e542ul}, {0xd6b050e861825c27ul, 0x012dfacdef9b2acbul},
      {0xd584cd745fda2d50ul, 0x012911cc969126edul}, {0xd45e220d050289dcul, 0x01244a7e41883478ul},
      {0xd33c2da0c51a2addul, 0x011fa3a2947be80dul}, {0xd21ed056cc17389ful, 0x011b1c0855e242b1ul},
      {0xd105eb804b444256ul, 0x0116b28c91476ab0ul}, {0xcff1618a9ca2919bul, 0x01126619c8b562fcul},
      {0xcee115f22df892c2ul, 0x010e35a733c69b06ul}, {0xcdd4ed3626679b15ul, 0x010a20380b5af461ul},
      {0xccccccccba615501ul, 0x010624dae0fd45c6ul}, {0xcbc89b1822bcc073ul, 0x010242a9011bb343ul},
      {0xcac83f5c2c7f0208ul, 0x00fe78c5df47a01dul}, {0xc9cba1b457aef7acul, 0x00fac65e8bc2ba09ul},
      {0xc8d2ab0a7c3d50d3ul, 0x00f72aa931add201ul}, {0xc7dd450decaf2e4aul, 0x00f3a4e49d3c0c30ul},
      {0xc6eb5a2b0ed05ed8ul, 0x00f03457c9598961ul}, {0xc5fcd583633d57baul, 0x00ecd8517440272ful},
      {0xc511a2e5f5151a02ul, 0x00e99027ba7f6d94ul}, {0xc429aec82b9944faul, 0x00e65b37b8065054ul},
      {0xc344e63ef7ef8028ul, 0x00e338e52ec627f5ul}, {0xc26336f8599bf5d0ul, 0x00e0289a328e43f1ul},
      {0xc1848f3534a97f36ul, 0x00dd29c6d9c6c8c4ul}, {0xc0a8ddc374ca0ae5ul, 0x00da3be0f2b84044ul},
      {0xbfd011f87909412ful, 0x00d75e63bd1367ceul}, {0xbefa1babc3f5037eul, 0x00d490cfa7726b5eul},
      {0xbe26eb31ec639422ul, 0x00d1d2aa1091ea10ul}, {0xbd567157cb3e6e11ul, 0x00cf237d0c04e2c5ul},
      {0xbc889f5de2f37db4ul, 0x00cc82d72a2b07d4ul}, {0xbbbd66f3fd64e23ful, 0x00c9f04b4334ffc2ul},
      {0xbaf4ba34fd61f809ul, 0x00c76b704505ce82ul}, {0xba2e8ba2e0e3754bul, 0x00c4f3e103c40dfcul},
      {0xb96ace22f170230cul, 0x00c2893c0cf0c3a8ul}, {0xb8a974fa203874c0ul, 0x00c02b237cdc8a2cul},
      {0xb7ea73c98b9d2dd5ul, 0x00bdd93cd65675adul}, {0xb72dbe8b2bf89b85ul, 0x00bb9330dc729598ul},
      {0xb673498ea5a2dce5ul, 0x00b958ab6e484260ul}, {0xb5bb09763e48710bul, 0x00b7295b648a85f9ul},
      {0xb504f333f3c6f565ul, 0x00b504f270dee576ul}};
    static const u32 ch[][2] = {
      {0xbff55815, 0x9bb5b6e7}, {0xb8a95a89, 0x938bf8f0}, {0xb1bf4705, 0x8bed0079}, {0xab309d4a, 0x84cdb431},
      {0xa4f76232, 0x7e24037b}, {0x9f0e1340, 0x77e6ca62}, {0x996f9b96, 0x720db8df}, {0x94174a00, 0x6c913cff},
      {0x8f00c812, 0x676a6f92}, {0x8a281226, 0x62930308}, {0x8589702c, 0x5e05343e}, {0x81216f2e, 0x59bbbcf8},
      {0x7cecdb76, 0x55b1c7d6}, {0x78e8bb45, 0x51e2e592}, {0x75124a0a, 0x4e4b0369}, {0x7166f40f, 0x4ae66284},
      {0x6de45288, 0x47b19045}, {0x6a882804, 0x44a95f5f}, {0x67505d2a, 0x41cae1a0}, {0x643afdc8, 0x3f13625c},
      {0x6146361f, 0x3c806169}, {0x5e70506d, 0x3a0f8e8e}, {0x5bb7b2b1, 0x37bec572}, {0x591adc9a, 0x358c09e2},
      {0x569865a7, 0x33758476}, {0x542efb6a, 0x31797f8a}, {0x51dd5ffb, 0x2f96647a}, {0x4fa2687c, 0x2dcab91f},
      {0x4d7cfbc9, 0x2c151d8a}, {0x4b6c1139, 0x2a7449ef}, {0x496eaf82, 0x28e70cc3}, {0x4783eba7, 0x276c4900},
      {0x45aae80a, 0x2602f493}, {0x43e2d382, 0x24aa16ec}, {0x422ae88c, 0x2360c7af}, {0x40826c88, 0x22262d7b},
      {0x3ee8af07, 0x20f97cd2}, {0x3d5d0922, 0x1fd9f714}, {0x3bdedce0, 0x1ec6e994}, {0x3a6d94a9, 0x1dbfacbb},
      {0x3908a2be, 0x1cc3a33b}, {0x37af80bf, 0x1bd23960}, {0x3661af39, 0x1aeae458}, {0x351eb539, 0x1a0d21a2},
      {0x33e61feb, 0x19387676}, {0x32b7823a, 0x186c6f3e}, {0x3192747d, 0x17a89f21}, {0x30769424, 0x16ec9f89},
      {0x2f63836f, 0x16380fbf}, {0x2e58e925, 0x158a9484}, {0x2d567053, 0x14e3d7ba}, {0x2c5bc811, 0x1443880e},
      {0x2b68a346, 0x13a958ab}, {0x2a7cb871, 0x131500ee}, {0x2997c17a, 0x12863c29}, {0x28b97b82, 0x11fcc95c},
      {0x27e1a6b4, 0x11786b03}, {0x2710061d, 0x10f8e6da}, {0x26445f86, 0x107e05ac}, {0x257e7b4d, 0x10079327},
      {0x24be2445, 0x0f955da9}, {0x24032795, 0x0f273620}, {0x234d5496, 0x0ebcefdb}, {0x229c7cbc, 0x0e56606e},
      {0x21f07377, 0x0df35f8b} };

    u64 indx = m>>58;
    u64 c3 = ch[indx][1], c0 = c[indx][0], c1 = c[indx][1], c2 = ch[indx][0];
    u64 d = (m<<6)>>32;
    u64 d2 = ((u64)(d*d))>>32;
    u64 re = c0 + (d2*c2>>13);
    u64 ro = d*((c1 + ((d2*c3)>>19))>>26)>>6;
    u64 r = re - ro; // error < 1e-7
    u64 r2 = mhuu(r,r);
    i64 h = mhuu(m,r2) + r2;
    i64 hr = mhii(h,r>>1);
    r -= hr;
    if(unlikely(!r)) r--;
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
