//===-- Implementation of expf128 function -------------------------------===//
//
// Copyright (c) 2019-2024 Alexei Sibidanov <sibid@uvic.ca>
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/expf128.h"
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
    float128 f;
    __m128i x;
  };

  static inline float128 reinterpret_u128_as_f128(u128 x){
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
 
  // get high 128 bit part of unsigned 128 bit squaring
  static inline u128 sqrhU(u128 _a){
    int128_64_32 a, a1a0, a1a1;
    a.a = _a;
    a1a0.a = (u128)a.b[1]*a.b[0];
    a1a1.a = (u128)a.b[1]*a.b[1];
    a1a1.a += a1a0.b[1];
    a1a1.a += a1a0.b[1];
    return a1a1.a;
  }
  
  // get high 3 limbs of unsigned 3x64 x 2x64 bit multiplication
  static inline void mhAU(u64 c[3], const u64 a[3], const u64 b[2]){
    int128_64_32 a2b0, a2b1, a1b0, a0b1, a1b1;
    a2b0.a = (u128)a[2]*b[0];
    a1b0.a = (u128)a[1]*b[0];
    a2b1.a = (u128)a[2]*b[1];
    a1b1.a = (u128)a[1]*b[1];
    a0b1.a = (u128)a[0]*b[1];
    unsigned long k;
    c[0] = __builtin_addcl(a1b0.b[1], a0b1.b[1], 0, &k);
    c[1] = __builtin_addcl(a2b1.b[0], 0, k, &k);
    c[2] = __builtin_addcl(a2b1.b[1], 0, k, &k);
    
    c[0] = __builtin_addcl(c[0], a1b1.b[0], 0, &k);
    c[1] = __builtin_addcl(c[1], a1b1.b[1], k, &k);
    c[2] = __builtin_addcl(c[2], 0, k, &k);
    
    c[0] = __builtin_addcl(c[0], a2b0.b[0], 0, &k);
    c[1] = __builtin_addcl(c[1], a2b0.b[1], k, &k);
    c[2] = __builtin_addcl(c[2], 0, k, &k);
  }
  
  static inline u128 hornerUU(u128 x, int n, const u64 c[][2]){
    n--;
    u128 y = (u128)c[n][1]<<64|c[n][0];
    while(n){n--; y = ((u128)c[n][1]<<64|c[n][0]) + mhUU(x,y);}
    return y;
  }

  static inline u128 hornerUUm(u128 x, int n, const u64 c[][2]){
    n--;
    u128 y = (u128)c[n][1]<<64|c[n][0];
    while(n){n--; y = ((u128)c[n][1]<<64|c[n][0]) - mhUU(x,y);}
    return y;
  }

  // 2^(i/32)
  static const u64 twopow[] = 
    {0x0000000000000000,                  0, 0x7c548eb68ca417fe, 0x059b0d31585743ae,
     0x8b92b71842a98364, 0x0b5586cf9890f629, 0xbbf1aed9318ceac6, 0x11301d0125b50a4e,
     0xf7c8c50eb14a7920, 0x172b83c7d517adcd, 0x5b8028990f07a98b, 0x1d4873168b9aa780,
     0x1fadb1c15cb593b0, 0x2387a6e75623866c, 0x5d15f5a24aa3bca9, 0x29e9df51fdee12c2,
     0x8d5a46305c85eded, 0x306fe0a31b7152de, 0x45502f4547987e3e, 0x371a7373aa9caa71,
     0x41223e13d773fba3, 0x3dea64c12342235b, 0x36f409df019fbd4f, 0x44e086061892d031,
     0x397afec42e20e036, 0x4bfdad5362a271d4, 0xa83c49d86a63f4e6, 0x5342b569d4f81df0,
     0x93015191eb345d89, 0x5ab07dd48542958c, 0x0fa06fd2da42bb1d, 0x6247eb03a5584b1f,
     0xb2fb1366ea957d3e, 0x6a09e667f3bcc908, 0x370f2ef0acd6cb43, 0x71f75e8ec5f73dd2,
     0x51023f6cda1f5ef4, 0x7a11473eb0186d7d, 0xf88afab34a010f6b, 0x82589994cce128ac,
     0x7c55a192c9bb3e6f, 0x8ace5422aa0db5ba, 0x01c3f2540a22d2fc, 0x93737b0cdc5e4f45,
     0xc46b071f2be58ddb, 0x9c49182a3f0901c7, 0x24491caf87bc8051, 0xa5503b23e255c8b4,
     0x734d1773205a7fbc, 0xae89f995ad3ad5e8, 0x7b081ab53c5354c9, 0xb7f76f2fb5e46eaa,
     0x0cb12a091ba66794, 0xc199bdd85529c222, 0x3cbd1e949db761d9, 0xcb720dcef9069150,
     0xa05aeb66e0dca9f6, 0xd5818dcfba48725d, 0x8cac39ed291b7226, 0xdfc97337b9b5eb96,
     0xf73a18f5db301f87, 0xea4afa2a490d9858, 0xf84b762862baff99, 0xf50765b6e4540674,
     0x0000000000000000,                  0
    };
    
  // polynomial coefficients for the range -1/64<=x<=1/64 of the function 2^x - 1
  static const u64 expcoef[][2] =
    {
      {0xc9e3b39803f2f576ul, 0xb17217f7d1cf79abul},
      {0xde2d60dd92e6bf2ful, 0x3d7f7bff058b1d50ul},
      {0x99d3b15d9b828df5ul, 0x0e35846b82505fc5ul},
      {0x39977c16a8371320ul, 0x0276556df749cee5ul},
      {0x41c5fc9498d461fcul, 0x005761ff9e299cc4ul},
      {0xb7a58526812b7e9bul, 0x000a184897c363c3ul},
      {0x34704072430dbd1eul, 0x0000ffe5fe2c4586ul},
      {0x24027b3f54eba625ul, 0x0000162c0223a5c8ul},
      {0x5ea0a58cd9f8e9d1ul, 0x000001b5253d3958ul},
      {0x24a83379d7015057ul, 0x0000001e4cf5158bul},
      {0xf3582f88eefb9c62ul, 0x00000001e8cb157cul},
      {0xcb8e291e50490585ul, 0x000000001c3bdac9ul} 
    };

  LLVM_LIBC_FUNCTION(float128, expf128, (float128 x)) {
    unsigned flagp = _mm_getcsr();
    int128_64_32 u; u.f = x;
    i64 sign = u.b[1]&(1ul<<63);
    u.b[1] &= (~0ul)>>1;
    int e = (u.b[1]>>48)&0x7fff;
    int128_64_32 xmax, xmin;
    xmax.f = 0x1.62e42fefa39ef35793c7673007e6p+13q; // above is only infinity
    xmin.f = 0x1.654bb3b2c73ebb059fabb506ff34p+13q; // for negative x round to zero
    if(unlikely(u.a>=xmax.a)){
      if (unlikely(e==0x7fff)){
	if((u.b[1]&(~0ul>>16))|u.b[0]){ // NAN
	  if(!(u.b[1]&(1l<<47))){ // signaling nan
	    _mm_setcsr(flagp|FE_INVALID);
	  }
	  u.b[1] |= 1l<<47; // transform snan to qnan
	  x = reinterpret_u128_as_f128(u.a);
	} else { //infinity
	  if (sign)
	    x = 0.0q;
	  else 
	    x = __builtin_inff128();
	}
	return x;
      }
      
      if(sign){
	if(u.a>=xmin.a){
	  _mm_setcsr(flagp|FE_INEXACT|FE_UNDERFLOW);
	  return 0.0q;
	}
      } else {
	_mm_setcsr(flagp|FE_INEXACT|FE_OVERFLOW);
	return __builtin_inff128();
      }
    }
    
    if (unlikely(e < 16383-38)) { // for small x exp(x) = 1 + x + x^2/2
      e -= 16383;
      if (e<-113) { // for really small x exp(x) = 1
	if(u.a!=0) _mm_setcsr(flagp|FE_INEXACT);
	return 1.0q;      
      }
      _mm_setcsr(flagp|FE_INEXACT);
      
      union int128_64_32 z;
      z.b[0] = u.b[0];
      z.b[1] = u.b[1]|(1ul<<48);
      z.a <<= 15;
      if(e>-64){
	u128 z2 = sqrhU(z.a);
	z2 >>= -e;
	if(sign)
	  z.a -= z2;
	else
	  z.a += z2;
      }
      int ds = !!sign;
      z.a >>= -e-ds;
      if(sign) z.a = -z.a;
      u64 ztail = (z.b[0]>>14)&1;
      z.a>>= 15;
      z.b[1] &= ~0ul>>16;
      z.a += ztail;
      z.b[1] += (16383ul-ds)<<48;
      return reinterpret_u128_as_f128(z.a);
    }

    e -= 16383;
    u.a <<= 15;
    u.b[1] |= 1ul<<63;
    i64 E = 0;
    int shift = e+2; // -35 <= shift < 16
    union int128_64_32 z;
    const u64 rln2[] = {0xeb577aa8dd695a59, 0xbe87fed0691d3e88, 0xb8aa3b295c17f0bb}; // 2^191/ln(2)

    u64 c[3]; mhAU(c, rln2, u.b);
    if(shift==0){
      c[2] = c[2];
      c[1] = c[1];
    } else if(shift<0){
      c[1] = c[1]>>-shift|c[2]<<(64+shift);
      c[2] = c[2]>>-shift;
    } else {
      E = c[2]>>(64-shift);
      c[2] = c[2]<<shift|c[1]>>(64-shift);
      c[1] = c[1]<<shift|c[0]>>(64-shift);
    }
    z.b[1] = c[2];
    z.b[0] = c[1];

    sign >>= 63;
    z.as ^= sign;
    z.as -= sign;
    u64 ik = (z.a>>(64+59))+((z.a>>(64+58))&1);
    i128 dz = z.a - ((i128)ik<<(64+59));

    const union int128_64_32 *utwopow = (const union int128_64_32 *)twopow;
    union int128_64_32 out = utwopow[ik], dout;
    if(dz<0){
      u128 adz = -dz;
      dout.a = adz;
      dout.a = hornerUUm(dout.a, 12, expcoef);
      dout.as = -mhUU(dout.as, adz);
    } else {
      dout.a = dz;
      dout.a = hornerUU(dz, 12, expcoef);
      dout.as = mhUU(dout.as, dz);
    }

    dout.as = dout.as<<(ik>>5);
    out.a = mhUU(out.a, dout.as)-((dout.as>>127)&out.a) + out.a + dout.as;
  
    if(unlikely(E>16381 && sign)){
      out.a >>= 1;
      out.b[1] |= 1ul<<63;
      E -= 16382-16;
      u64 outtail = (out.a>>(E-1))&1;
      out.a >>= 15; // split shift into 2 parts to deal with maximal E=128
      out.a >>= E-15;
      out.a += outtail;
      flagp |= FE_UNDERFLOW;
    } else {
      u64 outtail = (out.a>>15)&1;
      out.a >>= 16;
      out.a += outtail;
      E ^= sign;
      E += 16383;
      out.b[1] += E<<48;
    }
    _mm_setcsr(flagp|FE_INEXACT);
    return reinterpret_u128_as_f128(out.a);
  }
} // namespace LIBC_NAMESPACE_DECL
