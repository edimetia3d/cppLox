//
// License: MIT
//

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>

#include "mlir/Dialect/Lox/IR/LoxDialect.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Lox/IR/LoxTypes.cpp.inc"

/**
 * Follow MLIR's convention, all Type instances are proxy to some interned TypeStorage object:
 * 1. Type object are copiable
 * 2. Type object act as a proxy class that provide utility functions, while TypeStorage only stores data.
 *
 * Most types could be defined in tablegen, and use the generated TypeStorage, but when the generated TypeStorage
 * does not fit your need, you may use `let storageClass = "MyStorageType"` to change it.
 *
 * e.g. A "mutable" type is one that its Storage stores some data that are not used when interning, these kind of
 * TypeStorage could only be implemented in C++ for now.
 *
 * There are two core point when implementing a custom StorageType
 * 1. Define "KeyType" and related hooks to help MLIR to intern instance of StorageType.
 * 2. Hold datas used in "KeyType", and any other useful information
 *
 * A StorageType must provide these hooks:
 * 1. `using KeyType = ....`
 * 2. `operator==(const KeyTy &key)`
 * 3. `static llvm::hash_code hashKey(const KeyTy & key)`, only provide when MLIR cannot generate hash function for
 * `keyTy`
 * 4. `static TypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key)`.
 * 5. `static KeyTy getKey(...)`, only provide when MLIR cannot create `keyTy` instance from Ctor arguments.
 *
 * Also, you may treat MLIR Attribute as a special Type whose `KeyTy` contains datas, it may help you understand how to
 * implement your custom attribute.
 */

namespace mlir::lox {

bool isValidLoxType(Type type) {
  // clang-format off
  return type.isF64() ||
         type.isInteger(1) ||
         isa<UnrankedMemRefType>(type) ||
         isa<UnrankedTensorType>(type) ||
         isa<StructType>(type);
  // clang-format on
}

::mlir::Type StructType::parse(AsmParser &parser) {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  // Parse: `struct` `<`
  auto typeLoc = parser.getCurrentLocation();
  // Parse the element types of the struct.
  auto elementTypes = SmallVector<mlir::Type>();
  if (parser.parseLess()) {
    goto FAIL_EXIT;
  }

  do {
    // Parse the current element type.
    typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType)) {
      goto FAIL_EXIT;
    }

    if (!isValidLoxType(elementType)) {
      parser.emitError(typeLoc, "element type invalid") << elementType;
      goto FAIL_EXIT;
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater()) {
    goto FAIL_EXIT;
  }
  return StructType::get(parser.getContext(), elementTypes);
FAIL_EXIT:
  return Type();
}

void StructType::print(::mlir::AsmPrinter &odsPrinter) const {

  // Print the struct type according to the parser format.
  odsPrinter << "<";
  llvm::interleaveComma(getElementTypes(), odsPrinter);
  odsPrinter << '>';
}

mlir::Type StructType::TypeAt(int index) { return getElementTypes()[index]; }

llvm::ArrayRef<mlir::Type> StructType::getElementTypes() { return getImpl()->elementTypes; }

::mlir::LogicalResult StructType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                         SmallVector<Type> elementTypes) {
  // todo: add type verification
  return success();
}

void LoxDialect::InitTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Lox/IR/LoxTypes.cpp.inc"
      >();
}

} // namespace mlir::lox