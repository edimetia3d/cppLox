//
// License: MIT
//

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Transforms/InliningUtils.h>

#include "mlir/Dialect/Lox/IR/LoxDialect.h"

#include "mlir/Dialect/Lox/IR/LoxDialect.cpp.inc"

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
  void handleTerminator(mlir::Operation *terminator, mlir::ArrayRef<mlir::Value> valuesToRepl) const final;

  /// 1. Only used to handle CallOp in LoxDialect. (e.g. func.call does not belong to LoxDialect, so it will not be
  /// handled here)
  /// 2. Used to handle type mismatch between caller and callee, because MLIR does not require the caller's input
  /// operands
  ///    and callee's input parameters to be same type.
  mlir::Operation *materializeCallConversion(mlir::OpBuilder &builder, mlir::Value input, mlir::Type resultType,
                                             mlir::Location conversionLoc) const final;
};

void LoxInlinerInterface::handleTerminator(Operation *terminator, ArrayRef<Value> valuesToRepl) const {
  // Only "return" needs to be handled here.
  auto returnOp = cast<mlir::lox::ReturnOp>(terminator);

  // Replace the values directly with the return operands.
  assert(returnOp.getNumOperands() == valuesToRepl.size());
  for (const auto &it : llvm::enumerate(returnOp.getOperands()))
    valuesToRepl[it.index()].replaceAllUsesWith(it.value());
}

Operation *LoxInlinerInterface::materializeCallConversion(OpBuilder &builder, Value input, Type resultType,
                                                          Location conversionLoc) const {
  return builder.create<mlir::lox::CastOp>(conversionLoc, resultType, input);
}

void LoxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Lox/IR/Lox.cpp.inc"
      >();
  addInterfaces<LoxInlinerInterface>();
  InitTypes();
}

Operation *LoxDialect::materializeConstant(mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type,
                                           mlir::Location loc) {
  if (type.isa<StructType>())
    return builder.create<ConstantOp>(loc, type, value);
  if (type.isa<mlir::TensorType>())
    return builder.create<ConstantOp>(loc, value.cast<DenseElementsAttr>());
  if (type.isa<mlir::FloatType>())
    return builder.create<ConstantOp>(loc, value.cast<FloatAttr>().getValueAsDouble());
  if (type.isa<mlir::MemRefType>())
    return builder.create<ConstantOp>(loc, value.cast<StringAttr>());
  if (type.isa<mlir::IntegerType>())
    return builder.create<ConstantOp>(loc, (bool)value.cast<IntegerAttr>().getSInt());
  llvm_unreachable("unexpected type");
}

} // namespace mlir::lox