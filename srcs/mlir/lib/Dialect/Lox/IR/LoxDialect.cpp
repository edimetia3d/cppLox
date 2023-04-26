//
// License: MIT
//

#include <mlir/IR/DialectImplementation.h>

#include "mlir/Dialect/Lox/IR/LoxDialect.cpp.inc"
#include "mlir/Dialect/Lox/IR/LoxDialect.h"

#include "PrintDialectType.h"
#include "StructTypeStorage.h" // contains detail

namespace mlir::lox {

void LoxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Lox/IR/Lox.cpp.inc"
      >();
  addInterfaces<LoxInlinerInterface>();
  addTypes<StructType>();
}

mlir::Operation *LoxDialect::materializeConstant(mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type,
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

/**
 * Types defined in dialect WILL be parsed/printed by **Dialect**, not **Types** themself, and MLIR is designed to do
 * so.
 */

/// Parse an instance of a type registered to the toy dialect.
mlir::Type LoxDialect::parseType(mlir::DialectAsmParser &parser) const { return TypedParse<StructType>(parser); }

/// Print an instance of a type registered to the toy dialect.
void LoxDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  return TypedPrint<StructType>(type, printer);
}

} // namespace mlir::lox