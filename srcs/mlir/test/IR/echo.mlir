// RUN: %lox-mlir-opt -emit-bytecode %s | %lox-mlir-opt | FileCheck %s

func.func @constant_op() {
  // CHECK: lox.constant
  %c0_num = lox.constant {value = 0.0 : f64} : f64
  %c0_bool = lox.constant {value = 0 : i1} : i1
  %c_str = lox.constant {value = "some string"} : memref<*xi8>
  %c_tensor = lox.constant {value = dense<0.0> : tensor<32xf64>} : tensor<*xf64>
  %c_struct = lox.constant {value = [0.0 : f64, 0 : i1, "other string", dense<2.0> : tensor<6xf64>]} : !lox.struct<f64, i1, memref<*xi8>, tensor<*xf64>>
  return
}

func.func @assign_op() {
  %c0_f64 = lox.constant {value = 0.0 : f64} : f64
  // CHECK: lox.assign
  %named = lox.assign %c0_f64 {var_name = "foo"} : f64
  return
}

func.func @logial_op() {
  %c0_f64 = lox.constant {value = 0.0 : f64} : f64
  %c1_bool = lox.constant {value = 0 : i1} : i1
  // CHECK: lox.eq
  %0 = lox.eq %c0_f64 , %c1_bool : (f64, i1) -> (i1)
  // CHECK: lox.ne
  %1 = lox.ne %c0_f64 , %c1_bool : (f64, i1) -> (i1)
  // CHECK: lox.ge
  %2 = lox.ge %c0_f64 , %c1_bool : (f64, i1) -> (i1)
  // CHECK: lox.gt
  %3 = lox.gt %c0_f64 , %c1_bool : (f64, i1) -> (i1)
  // CHECK: lox.le
  %4 = lox.le %c0_f64 , %c1_bool : (f64, i1) -> (i1)
  // CHECK: lox.lt
  %5 = lox.lt %c0_f64 , %c1_bool : (f64, i1) -> (i1)
  return
}


func.func @binary_op() {
  %c0_f64 = lox.constant {value = 0.0 : f64} : f64
  %c1_f64 = lox.constant {value = 1.0 : f64} : f64
  // CHECK: lox.add
  %0 = lox.add %c0_f64 , %c1_f64 : f64
  // CHECK: lox.sub
  %1 = lox.sub %c0_f64 , %c1_f64 : f64
  // CHECK: lox.mul
  %2 = lox.mul %c0_f64 , %c1_f64 : f64
  // CHECK: lox.div
  %3 = lox.div %c0_f64 , %c1_f64 : f64
  // CHECK: lox.mod
  %4 = lox.mod %c0_f64 , %c1_f64 : f64
  return
}

func.func @grouping_op() {
  %c0_f64 = lox.constant {value = 0.0 : f64} : f64
  %c1_f64 = lox.constant {value = 1.0 : f64} : f64
  %0 = lox.add %c0_f64 , %c1_f64 : f64
  // CHECK: lox.grouping
  %1 = lox.grouping %0 : f64
  return
}

func.func @unary_op() {
  %c0_f64 = lox.constant {value = 0.0 : f64} : f64
  %c1_bool = lox.constant {value = 1 : i1} : i1
  // CHECK: lox.neg
  %0 = lox.neg %c0_f64 : f64
  // CHECK: lox.not
  %1 = lox.not %c1_bool : i1
  return
}

func.func @attr_set_get_op() {
  %c_struct = lox.constant {value = [0.0 : f64, 0 : i1, "other string", dense<2.0> : tensor<6xf64>]} : !lox.struct<f64, i1, memref<*xi8>, tensor<*xf64>>
  %c0_f64 = lox.constant {value = 0.0 : f64} : f64
  // CHECK: lox.set_attr
  lox.set_attr  %c0_f64 : f64  {attr_name = "foo"} >> %c_struct : !lox.struct<f64, i1, memref<*xi8>, tensor<*xf64>>
  // CHECK: lox.get_attr
  %attr_v = lox.get_attr %c_struct : !lox.struct<f64, i1, memref<*xi8>, tensor<*xf64>> {attr_name = "foo"} >> f64
  return
}