//
// License: MIT
//

/**
 * @file LoxInterface.h
 * @brief Contains the Lox types. Currently are:
 *      - StructType: A abstract struct type. Any lox class instance in lox will be a mlir value of StructType.
 */

#ifndef LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_LOXTYPES_H
#define LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_LOXTYPES_H

#include <mlir/IR/BuiltinTypes.h>

#include "mlir/Dialect/Lox/IR/LoxTypes.h.inc"

namespace mlir::lox {

namespace detail {
// A totally opaque struct type. that should not be accessed by users.
struct StructTypeStorage;
} // namespace detail

/**
 * StructType is actually some kind of meta type. We use the `StructType::get()` to get a "concrete" struct type.
 * and all these "concrete" types will be treated as StructType.
 * eg. `!lox.struct<int>` `!lox.struct<int,tensor>` are both `StructType`
 */
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type, detail::StructTypeStorage> {
public:
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getElementNum() { return getElementTypes().size(); }
};
} // namespace mlir::lox

#endif // LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_LOXTYPES_H
