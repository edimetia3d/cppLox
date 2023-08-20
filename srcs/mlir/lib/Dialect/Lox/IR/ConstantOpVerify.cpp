//
// License: MIT
//
#include "ConstantOpVerify.h"

#include <llvm/ADT/TypeSwitch.h>

#include "mlir/Dialect/Lox/IR/LoxDialect.h"

namespace mlir::lox {
using EmitErrFn = std::function<InFlightDiagnostic(const Twine &msg)>;
LogicalResult verifyConstantForType(Type type, Attribute opaqueValue, EmitErrFn err);

// Tensor
static LogicalResult verifyConstant(UnrankedTensorType type, Attribute opaqueValue, EmitErrFn err) {
  auto attrValue = opaqueValue.dyn_cast<DenseFPElementsAttr>();
  if (!attrValue)
    return err("constant of TensorType must be initialized by "
               "a DenseFPElementsAttr, got ")
           << opaqueValue;

  // Check that the rank of the attribute must be a static shaped tensor
  auto attrType = attrValue.getType().cast<TensorType>();
  if (!attrType.hasStaticShape()) {
    return err("attribute must have static shape");
  }
  if (!attrType.getElementType().isF64() || !type.getElementType().isF64()) {
    return err("element must be of type f64");
  }

  return success();
}

// Struct
static LogicalResult verifyConstant(StructType type, Attribute opaqueValue, EmitErrFn err) {
  llvm::ArrayRef<Type> resultElementTypes = type.getElementTypes();

  // Verify that the initializer is an Array.
  auto attrValue = opaqueValue.dyn_cast<ArrayAttr>();
  if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
    return err("constant of StructType must be initialized by an "
               "ArrayAttr with the same number of elements, got ")
           << opaqueValue;

  // Check that each of the elements are valid.
  llvm::ArrayRef<Attribute> attrElementValues = attrValue.getValue();
  for (const auto it : llvm::zip(resultElementTypes, attrElementValues)) {
    if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), err))) {
      return failure();
    }
  }
  return success();
}

// String
static LogicalResult verifyConstant(UnrankedMemRefType type, Attribute opaqueValue, EmitErrFn err) {
  auto attrValue = opaqueValue.dyn_cast<StringAttr>();
  if (!attrValue)
    return err("constant of string must be initialized by "
               "a StringAttr, got ")
           << opaqueValue;

  // check memref of i8;
  if (!type.getElementType().isInteger(8))
    return err("constant of string must be 1-D memref of i8, got ") << type;
  return success();
}

// Bool
static LogicalResult verifyConstant(IntegerType type, Attribute opaqueValue, EmitErrFn err) {
  // attr type must be BoolAttr
  auto attrValue = opaqueValue.dyn_cast<BoolAttr>();
  if (!attrValue)
    return err("constant of bool must be initialized by "
               "a BoolAttr, got ")
           << opaqueValue;

  // result type must be i1
  if (type.getWidth() != 1)
    return err("constant of bool must be type of i1, got ") << type;
  return success();
}

// Number
static LogicalResult verifyConstant(Float64Type type, Attribute opaqueValue, EmitErrFn err) {
  // attr type must be Float64Attr
  auto attrValue = opaqueValue.dyn_cast<FloatAttr>();
  if (!attrValue)
    return err("constant of number must be initialized by "
               "a FloatAttr, got ")
           << opaqueValue;
  auto resultType = type.dyn_cast<Float64Type>();
  if (!resultType)
    return err("constant of number must be type of float64, got ") << resultType;
  return success();
}

LogicalResult verifyConstantForType(Type type, Attribute opaqueValue, EmitErrFn err) {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<UnrankedTensorType, StructType, UnrankedMemRefType, Float64Type, IntegerType>(
          [&](auto type) { return verifyConstant(type, opaqueValue, err); })
      .Default([&](auto type) { return err("constant of type ") << type << " is not supported"; });
}

/// Verify that the given attribute value is valid for the given type.
LogicalResult verifyConstantOp(Attribute opaqueValue, Operation *op) {
  EmitErrFn emitError = [&](const Twine &msg) { return op->emitOpError(msg); };
  return verifyConstantForType(op->getResult(0).getType(), opaqueValue, emitError);
}
} // namespace mlir::lox