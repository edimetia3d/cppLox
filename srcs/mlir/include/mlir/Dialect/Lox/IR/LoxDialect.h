//
// License: MIT
//

/**
 * Contains the Lox dialect.
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
#include <mlir/Transforms/InliningUtils.h>

// Usually many generated file should be included before the dialect ops at top level
#include "mlir/Dialect/Lox/IR/LoxDialect.h.inc"
#include "mlir/Dialect/Lox/IR/LoxShapeInferInterface.h.inc"
#include "mlir/Dialect/Lox/IR/LoxTypes.h.inc"

namespace mlir::lox {

struct LoxInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// Checks If callable is legal to inline into caller
  bool isLegalToInline(mlir::Operation *caller, mlir::Operation *callee, bool wouldBeCloned) const final {
    // usually, we need to check callee's results type/number are same as caller, here we just return true.
    return true;
  }

  /// Check If the src op is legal to inline into region
  bool isLegalToInline(mlir::Operation *src, mlir::Region *dest, bool, mlir::IRMapping &) const final {
    // src must be a callable, and src's region type must be same as dest's.
    return true;
  }

  /// Checks If the src region is legal to inline into the dest regin
  bool isLegalToInline(mlir::Region *dest, mlir::Region *src, bool, mlir::IRMapping &) const final {
    // usually, region type just need to be same. In lox , we only have SSACFG region, so this should always be true.
    return true;
  }

  /// Terminator may return control flow to the caller, thus we must update caller's result value when necessary.
  /// `valuesToRepl` contains the callers result values.
  void handleTerminator(mlir::Operation *op, mlir::ArrayRef<mlir::Value> valuesToRepl) const final;

  /// Caller's input arguments may not match the callee's signature, thus we need to materialize a type conversion.
  mlir::Operation *materializeCallConversion(mlir::OpBuilder &builder, mlir::Value input, mlir::Type resultType,
                                             mlir::Location conversionLoc) const final;
};

} // namespace mlir::lox

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Lox/IR/LoxTypes.h.inc"

// usually ops should be included at last, for it may relay on the types and interfaces defined above.
#define GET_OP_CLASSES
#include "mlir/Dialect/Lox/IR/Lox.h.inc"

#endif // LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_LOXDIALECT_H
