//
// License: MIT
//
#include "mlir/Dialect/Lox/IR/Lox.h"

#include "mlir/Dialect/Lox/IR/LoxInterface.cpp.inc"

using namespace mlir;
void LoxInlinerInterface::handleTerminator(Operation *op, ArrayRef<Value> valuesToRepl) const {
  // Only "return" needs to be handled here.
  auto returnOp = cast<mlir::lox::ReturnOp>(op);

  // Replace the values directly with the return operands.
  assert(returnOp.getNumOperands() == valuesToRepl.size());
  for (const auto &it : llvm::enumerate(returnOp.getOperands()))
    valuesToRepl[it.index()].replaceAllUsesWith(it.value());
}
Operation *LoxInlinerInterface::materializeCallConversion(OpBuilder &builder, Value input, Type resultType,
                                                          Location conversionLoc) const {
  return builder.create<mlir::lox::CastOp>(conversionLoc, resultType, input);
}