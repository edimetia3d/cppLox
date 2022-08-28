//
// License: MIT
//

/**
 * @file LoxInterface.h
 * @brief Contains the Lox dialect interface. Currently are:
 *       - ShapeInferenceOpInterface: in LoxBase.td, used to support Tensor shape inference
 *       - LoxInlinerInterface: in this file, used to support inlining of Lox ops
 */

#ifndef LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_CUOSTOMINTERFACE_H_INC
#define LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_CUOSTOMINTERFACE_H_INC

#include <mlir/Transforms/InliningUtils.h>

#include "mlir/Dialect/Lox/IR/LoxInterface.h.inc"

// Interfaces should generally be defined in global namespace.
struct LoxInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// Checks If callable is legal to inline into caller
  bool isLegalToInline(mlir::Operation *caller, mlir::Operation *callee, bool wouldBeCloned) const final {
    // usually, we need to check callee's results type/number are same as caller, here we just return true.
    return true;
  }

  /// Check If the src op is legal to inline into region
  bool isLegalToInline(mlir::Operation *src, mlir::Region *dest, bool, mlir::BlockAndValueMapping &) const final {
    // src must be a callable, and src's region type must be same as dest's.
    return true;
  }

  /// Checks If the src region is legal to inline into the dest regin
  bool isLegalToInline(mlir::Region *dest, mlir::Region *src, bool, mlir::BlockAndValueMapping &) const final {
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

#endif // LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_CUOSTOMINTERFACE_H_INC
