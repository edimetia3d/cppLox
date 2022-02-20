#ifndef LOX_SRCS_LOX_BACKEND_JIT_MLIR_INCLUDE_MLIR_DIALECT_LOX_DIALECT_H
#define LOX_SRCS_LOX_BACKEND_JIT_MLIR_INCLUDE_MLIR_DIALECT_LOX_DIALECT_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "mlir/Dialect/lox/ShapeInferenceInterface.h"
#include "mlir/Dialect/lox/loxDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/lox/lox.h.inc"

#endif  // LOX_SRCS_LOX_BACKEND_JIT_MLIR_INCLUDE_MLIR_DIALECT_LOX_DIALECT_H
