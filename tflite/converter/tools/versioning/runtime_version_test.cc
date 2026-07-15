/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tflite/converter/tools/versioning/runtime_version.h"

#include <string>

#include <gtest/gtest.h>

namespace tflite {

TEST(OpVersionTest, CompareRuntimeVersion) {
  EXPECT_TRUE(CompareRuntimeVersion("1.9", "1.13"));
  EXPECT_FALSE(CompareRuntimeVersion("1.13", "1.13"));
  EXPECT_TRUE(CompareRuntimeVersion("1.14", "1.14.1"));
  EXPECT_FALSE(CompareRuntimeVersion("1.14.1", "1.14"));
  EXPECT_FALSE(CompareRuntimeVersion("1.14.1", "1.9"));
  EXPECT_FALSE(CompareRuntimeVersion("1.0.9", "1.0.8"));
  EXPECT_FALSE(CompareRuntimeVersion("2.1.0", "1.2.0"));
  EXPECT_TRUE(CompareRuntimeVersion("", "1.13"));
  EXPECT_FALSE(CompareRuntimeVersion("", ""));
}

TEST(OpVersionTest, Float8OperatorVersions) {
  struct OpVersion {
    BuiltinOperator op;
    int version;
  };
  constexpr OpVersion kFloat8OperatorVersions[] = {
      {BuiltinOperator_GATHER, 8},
      {BuiltinOperator_SPLIT, 5},
      {BuiltinOperator_UNPACK, 6},
      {BuiltinOperator_DEQUANTIZE, 9},
      {BuiltinOperator_REVERSE_V2, 4},
      {BuiltinOperator_PACK, 5},
      {BuiltinOperator_GATHER_ND, 6},
      {BuiltinOperator_FILL, 5},
      {BuiltinOperator_PAD, 6},
      {BuiltinOperator_PADV2, 6},
      {BuiltinOperator_CONCATENATION, 7},
      {BuiltinOperator_SPLIT_V, 3},
      {BuiltinOperator_BROADCAST_TO, 4},
      {BuiltinOperator_CAST, 9},
  };

  for (const OpVersion& op_version : kFloat8OperatorVersions) {
    EXPECT_EQ(FindMinimumRuntimeVersionForOp(op_version.op, op_version.version),
              "2.23.0");
  }
}

}  // namespace tflite
