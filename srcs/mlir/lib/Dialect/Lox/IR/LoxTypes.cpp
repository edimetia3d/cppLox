//
// License: MIT
//

#include <mlir/IR/BuiltinTypes.h>

#include "mlir/Dialect/Lox/IR/LoxDialect.h"
#include "mlir/Dialect/Lox/IR/LoxTypes.cpp.inc"

#include "StructTypeStorage.h"

namespace mlir::lox {

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in. The
  // parameters after the context are forwarded to the storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, elementTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}
llvm::Expected<StructType> StructType::parse(DialectAsmParser &parser) {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return llvm::make_error<llvm::StringError>(std::make_error_code(std::errc::wrong_protocol_type));

  // Parse the element types of the struct.
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return llvm::make_error<llvm::StringError>(std::make_error_code(std::errc::protocol_error));

    // Check that the type is either a TensorType or another StructType.
    if (!elementType.isa<mlir::TensorType, StructType>()) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
      return llvm::make_error<llvm::StringError>(std::make_error_code(std::errc::protocol_error));
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return llvm::make_error<llvm::StringError>(std::make_error_code(std::errc::protocol_error));
  return StructType::get(elementTypes);
}

void StructType::print(StructType structType, DialectAsmPrinter &printer) {

  // Print the struct type according to the parser format.
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}
} // namespace mlir::lox