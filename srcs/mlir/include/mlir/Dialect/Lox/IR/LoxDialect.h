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
#include "mlir/Dialect/Lox/IR/LoxInterface.h.inc"
#include "mlir/Dialect/Lox/IR/LoxTypes.h.inc"

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

  /// A util function to parse a struct type from a string, note that this function will not be called by MLIR's parsing
  /// framework. It is called by us in the LoxDialect::parseType manually.
  static llvm::Expected<StructType> parse(mlir::DialectAsmParser &parser);

  /// A util function to print a struct type to a string, note that this function will not be called by MLIR's printing
  /// framework. It is called by us in the LoxDialect::printType manually.
  static void print(StructType type, mlir::DialectAsmPrinter &printer);
};
} // namespace mlir::lox

// usually ops should be included at last, for it may relay on the types and interfaces defined above.
#define GET_OP_CLASSES
#include "mlir/Dialect/Lox/IR/Lox.h.inc"

#endif // LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_LOXDIALECT_H
