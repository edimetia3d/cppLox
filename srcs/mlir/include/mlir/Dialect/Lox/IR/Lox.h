#ifndef LOX_SRCS_LOX_BACKEND_JIT_MLIR_INCLUDE_MLIR_DIALECT_LOX_DIALECT_H
#define LOX_SRCS_LOX_BACKEND_JIT_MLIR_INCLUDE_MLIR_DIALECT_LOX_DIALECT_H

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "mlir/Dialect/Lox/IR/CuostomInterface.h.inc"
#include "mlir/Dialect/Lox/IR/LoxDialect.h.inc"
#include "mlir/Dialect/Lox/IR/LoxInterface.h.inc"
#include "mlir/Dialect/Lox/IR/LoxTypes.h.inc"

namespace mlir::lox {

namespace detail {
struct StructTypeStorage;
} // namespace detail

/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type, detail::StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};
} // namespace mlir::lox

#define GET_OP_CLASSES
#include "mlir/Dialect/Lox/IR/Lox.h.inc"
#endif // LOX_SRCS_LOX_BACKEND_JIT_MLIR_INCLUDE_MLIR_DIALECT_LOX_DIALECT_H
