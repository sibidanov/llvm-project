//===-- Unittests for expf128 --------------------------------------------===//
//
// Copyright (c) 2024 Alexei Sibidanov <sibid@uvic.ca>
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/rand.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include "src/math/expf128.h"

#include "src/__support/FPUtil/FPBits.h"

using LlvmLibcExpf128Test = LIBC_NAMESPACE::testing::FPTest<float128>;

TEST_F(LlvmLibcExpf128Test, RandomInputs) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float128>;
  using StorageType = typename FPBits::StorageType;

  // generate uniformly random numbers in the [a,b] range using libc rand() function
  auto uniform = [](float128 a, float128 b){
    StorageType z0 = static_cast<StorageType>(LIBC_NAMESPACE::rand());
    StorageType z1 = static_cast<StorageType>(LIBC_NAMESPACE::rand());
    StorageType z2 = static_cast<StorageType>(LIBC_NAMESPACE::rand());
    StorageType z3 = static_cast<StorageType>(LIBC_NAMESPACE::rand());
    StorageType z = z3<<93|z2<<62|z1<<31|z0;
    z &= (~static_cast<StorageType>(0))>>16;
    z |= static_cast<StorageType>(0x3fff)<<(64+48);
    float128 x = FPBits(z).get_val();
    x = a + (b-a)*x - (b-a);
    return x;
  };
  
  float128 beg = -10.60q, end = 10.60q;
  for(size_t i = 0; i < 10ul*1000ul*1000ul; i++){
    float128 x = uniform(beg, end);
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, LIBC_NAMESPACE::expf128(x), 1);
  }

  float128 X[] = {
    0x1.62e42fefa39ef35793c7673007e5p13q,
    0x1.62e42fefa39ef35793c7673007e6p13q,
    -0x1.62d918ce2421d65ff90ac8f4ce66p13q,
    -0x1.62d918ce2421d65ff90ac8f4ce65p13q,
    -0x1.6546282207802c89d24d65e96274p13q,
    -0x1.654bb3b2c73ebb059fabb506ff33p13q,
    -0x1.654bb3b2c73ebb059fabb506ff34p13q,
    -0x1.6412664075b048b2cc5b3efde6cdp13q,
  };
  
  for(size_t i = 0; i < sizeof(X)/sizeof(X[0]); i++){
    float128 x = X[i];
    ASSERT_MPFR_MATCH_DEFAULT(LIBC_NAMESPACE::testing::mpfr::Operation::Exp, x, LIBC_NAMESPACE::expf128(x), 1);
  }
}
