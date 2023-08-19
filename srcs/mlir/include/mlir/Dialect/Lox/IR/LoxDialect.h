//
// License: MIT
//

/**
 * Contains every IR elements of the Lox dialect.
 * There is a convention that such file should be named "FooDialect.h" or "FooOps.h".
 */

#ifndef LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_LOXDIALECT_H
#define LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_LOXDIALECT_H

#include <llvm/Support/Error.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

// Usually many generated file should be included before the dialect ops at top level
#include "mlir/Dialect/Lox/IR/LoxDialect.h.inc"
#include "mlir/Dialect/Lox/IR/LoxShapeInferInterface.h.inc"
#include "mlir/Dialect/Lox/IR/LoxTypes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Lox/IR/LoxTypes.h.inc"

// usually ops should be included at last, for it may relay on the types and interfaces defined above.
#define GET_OP_CLASSES
#include "mlir/Dialect/Lox/IR/Lox.h.inc"

#endif // LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_LOXDIALECT_H
