//===-- Performance test for expf128 ----------------------------------------===//
//
// Copyright (c) 2024 Alexei Sibidanov <sibid@uvic.ca>
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/rand.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/src/math/performance_testing/Timer.h"
#include <fstream>

#include "src/math/expf128.h"

int main(){
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

  std::ofstream log("expf128_perf.log");
  const size_t n = 100ul*1000ul, rounds = 1000;
  
  // fill the test array with uniformly distributed random numbers in the [-10,10] range
  float128 z[n]; for(size_t i = 0; i < n; i++) z[i] = uniform(-10.0q, 10.0q);
  
  LIBC_NAMESPACE::testing::Timer timer;
  timer.start();
  [[maybe_unused]] float128 result;
  for (size_t i = 0; i < rounds; i++) {
    for(size_t i = 0; i < n; i++)
      result = LIBC_NAMESPACE::expf128(z[i]);
  }
  timer.stop();
  
  uint64_t nn = timer.nanoseconds();
  double myAverage = static_cast<double>(nn) / (n * rounds);
  log << "-- expf128 function --\n"<<std::dec;
  log << "    Test duration: " << nn << " ns or " << nn/1e9 << " sec\n";
  log << "    Time/call    : " << myAverage << " ns/call \n";
  log << "    Call/second  : " << 1e3 / myAverage << "M call/s \n";
  return 0;
}
