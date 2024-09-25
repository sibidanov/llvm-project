//===-- Unittests for sqrtf128---------------------------------------------===//
//
// Copyright (c) 2024 Alexei Sibidanov <sibid@uvic.ca>
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SqrtTest.h"

#include "src/math/sqrtf128.h"

LIST_SQRT_TESTS(float128, LIBC_NAMESPACE::sqrtf128)

TEST_F(LlvmLibcSqrtTest, HardToRound) {
  // Since there is no exact half cases for square root I encode the
  // round direction in the sign of the result. E.g. if the number is
  // negative it means that the exact root is below the rounded value
  // (the absolute value). Thus I can test not only hard to round
  // cases for the round to nearest mode but also the directional
  // modes.
  struct {float128 x, y;} ts[] = {
    {0x0.000000dee2f5b6a26c8f07f05442p-16382q, -0x1.ddbd8763a617cff753e2a31083p-8204q},
    {0x0.000000c86d174c5ad8ae54a548e7p-16382q, 0x1.c507bb538940719890851ec1ca88p-8204q},
    {0x0.000020ab15cfe0b8e488e128f535p-16382q, -0x1.6dccb402560213bc0d62d62e910bp-8201q},
    {0x0.0000219e97732a9970f2511989bap-16382q, 0x1.73163d28be706f4b5052791e28a5p-8201q},
    {0x0.000026e477546ae99ef57066f9fdp-16382q, -0x1.8f20dd0d0c570a23ea59bc2bf009p-8201q},
    {0x0.00002d0f88d27a496b3e533f5067p-16382q, 0x1.ad9d4abe9f047225a7352bcc52c1p-8201q},
    {0x1.0000000000000000000000000001p+0q, 0x1p+0q},
    {0x1.0000000000000000000000000003p+0q, 0x1.0000000000000000000000000001p+0q},
    {0x1.0000000000000000000000000005p+0q, 0x1.0000000000000000000000000002p+0q},
    {0x1.2af17a4ae6f93d11310c49c11b59p+0q, -0x1.14a3bdf0ea5231f12d421a5dbe33p+0q},
    {0x1.c4f5074269525063a26051a0ad27p+0q, 0x1.54864e9b1daa4d9135ff00663366p+0q},
    {0x1.035cb5f298a801dc4be9b1f8cd97p+1q, -0x1.6c688775bffcb3f507ba11d0abb9p+0q},
    {0x1.274be02380427e709beab4dedeb4p+1q, -0x1.84d5763281f2318422392e506b1cp+0q},
    {0x1.64e797cfdbaa3f7e2f33279dbc6p+1q, 0x1.ab79b164e255b26eca00ff99cc99p+0q},
    {0x1.d78d8352b48608b510bfd5c75315p+1q, -0x1.eb5c420f15adce0ed2bde5a241cep+0q},
    {0x1.fffffffffffffffffffffffffffbp+1q, 0x1.fffffffffffffffffffffffffffdp+0q},
    {0x1.fffffffffffffffffffffffffffdp+1q, 0x1.fffffffffffffffffffffffffffep+0q},
    {0x1.ffffffffffffffffffffffffffffp+1q, 0x1.ffffffffffffffffffffffffffffp+0q},
  };

  auto rnd = [](float128 x, int rm){
    union {float128 f; unsigned __int128 u;} y;
    int d = x<0;
    if(d)
      y.f = -x;
    else
      y.f = x;
    if(rm==0){//nearest
    } else if(rm==1){ // up
      if(!d) y.u += 1;
    } else { // down or zero
      if(d) y.u -= 1;
    }
    return y.f;
  };

  for(auto &t: ts){
    EXPECT_FP_EQ_ROUNDING_NEAREST     (rnd(t.y,0), LIBC_NAMESPACE::sqrtf128(t.x));
    EXPECT_FP_EQ_ROUNDING_UPWARD      (rnd(t.y,1), LIBC_NAMESPACE::sqrtf128(t.x));
    EXPECT_FP_EQ_ROUNDING_DOWNWARD    (rnd(t.y,2), LIBC_NAMESPACE::sqrtf128(t.x));
    EXPECT_FP_EQ_ROUNDING_TOWARD_ZERO (rnd(t.y,3), LIBC_NAMESPACE::sqrtf128(t.x));
  }

  auto checkexact = [this](float128 x, float128 y){
    EXPECT_FP_EQ_ROUNDING_NEAREST     (y, LIBC_NAMESPACE::sqrtf128(x));
    EXPECT_FP_EQ_ROUNDING_UPWARD      (y, LIBC_NAMESPACE::sqrtf128(x));
    EXPECT_FP_EQ_ROUNDING_DOWNWARD    (y, LIBC_NAMESPACE::sqrtf128(x));
    EXPECT_FP_EQ_ROUNDING_TOWARD_ZERO (y, LIBC_NAMESPACE::sqrtf128(x));
  };

  auto checkexactl = [checkexact](unsigned long k){
    __int128 kx = (__int128)k*k;
    float128 x = kx, y = k;
    return checkexact(x,y);
  };

  // exact results for subnormal arguments
  struct {float128 x, y;} te[] = {
    {0x0.0000000000000000000000000001p-16382q, 0x1p-8247q},
    {0x0.0000000000000000000000000004p-16382q, 0x1p-8246q},
    {0x0.0000000000001000000000000000p-16382q, 0x1p-8217q},
    {0x0.0000000000010000000000000000p-16382q, 0x1p-8215q},
    {0x0.0000000000100000000000000000p-16382q, 0x1p-8213q},
  };

  for(auto &t: te) checkexact(t.x,t.y);

  // check exact cases starting from small numbers
  for(unsigned long k=1;k<1000l*1000l;k++) checkexactl(k);

  // then from the largest number
  unsigned long k0 = 101904826760412362l;
  for(unsigned long k=k0;k>k0-1000l*1000l;k--) checkexactl(k);
}
