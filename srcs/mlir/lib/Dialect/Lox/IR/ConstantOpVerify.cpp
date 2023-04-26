//
// License: MIT
//
#include "mlir/Dialect/Lox/IR/LoxDialect.h"

namespace mlir::lox {
mlir::LogicalResult verifyConstantForType(mlir::Type type, mlir::Attribute opaqueValue, mlir::Operation *op);

static LogicalResult verifyConstantTensor(Type type, Attribute opaqueValue, Operation *op) {
  auto attrValue = opaqueValue.dyn_cast<DenseFPElementsAttr>();
  if (!attrValue)
    return op->emitError("constant of TensorType must be initialized by "
                         "a DenseFPElementsAttr, got ")
           << opaqueValue;

  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = type.dyn_cast<RankedTensorType>();
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the
  // constant result type.
  auto attrType = attrValue.getType().cast<TensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return op->emitOpError("return type must match the one of the attached "
                           "value attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return op->emitOpError("return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim] << " != " << resultType.getShape()[dim];
    }
  }
  return success();
}
static LogicalResult verifyConstantStruct(Type type, Attribute opaqueValue, Operation *op) {
  auto resultType = type.cast<StructType>();
  llvm::ArrayRef<Type> resultElementTypes = resultType.getElementTypes();

  // Verify that the initializer is an Array.
  auto attrValue = opaqueValue.dyn_cast<ArrayAttr>();
  if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
    return op->emitError("constant of StructType must be initialized by an "
                         "ArrayAttr with the same number of elements, got ")
           << opaqueValue;

  // Check that each of the elements are valid.
  llvm::ArrayRef<Attribute> attrElementValues = attrValue.getValue();
  for (const auto it : llvm::zip(resultElementTypes, attrElementValues))
    if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), op)))
      return failure();
  return success();
}

static LogicalResult verifyConstantString(Type type, Attribute opaqueValue, Operation *op) {
  auto attrValue = opaqueValue.dyn_cast<StringAttr>();
  if (!attrValue)
    return op->emitError("constant of string must be initialized by "
                         "a StringAttr, got ")
           << opaqueValue;

  if (type.dyn_cast<UnrankedTensorType>()) {
    return success();
  }

  // The ranked memref must be a 1-D memref of i8;
  auto resultType = type.dyn_cast<MemRefType>();
  bool check_fail = !resultType || (resultType.getRank() != 1) ||
                    (resultType.getElementType() != IntegerType::get(op->getContext(), 8));
  if (check_fail)
    return op->emitError("constant of string must be 1-D memref of i8, got ") << resultType;
  return success();
}

static LogicalResult verifyConstantBool(Type type, Attribute opaqueValue, Operation *op) {
  // attr type must be BoolAttr
  auto attrValue = opaqueValue.dyn_cast<BoolAttr>();
  if (!attrValue)
    return op->emitError("constant of bool must be initialized by "
                         "a BoolAttr, got ")
           << opaqueValue;

  // result type must be i1
  auto resultType = type.dyn_cast<IntegerType>();
  if (!resultType || resultType.getWidth() != 1)
    return op->emitError("constant of bool must be initialized by "
                         "a BoolAttr, got ")
           << opaqueValue;
  return success();
}

static LogicalResult verifyConstantNumber(Type type, Attribute opaqueValue, Operation *op) {
  // attr type must be Float64Attr
  auto attrValue = opaqueValue.dyn_cast<FloatAttr>();
  if (!attrValue)
    return op->emitError("constant of number must be initialized by "
                         "a FloatAttr, got ")
           << opaqueValue;
  return success();
}
/// Verify that the given attribute value is valid for the given type.
mlir::LogicalResult verifyConstantForType(mlir::Type type, mlir::Attribute opaqueValue, mlir::Operation *op) {
  if (type.isa<mlir::TensorType>()) {
    return verifyConstantTensor(type, opaqueValue, op);
  }
  if (type.isa<StructType>()) {
    return verifyConstantStruct(type, opaqueValue, op);
  }
  if (type.isa<MemRefType>()) {
    return verifyConstantString(type, opaqueValue, op);
  }
  if (type.isa<Float64Type>()) {
    return verifyConstantNumber(type, opaqueValue, op);
  }
  if (type.isa<IntegerType>()) {
    return verifyConstantBool(type, opaqueValue, op);
  }
  llvm_unreachable("Unknown type");
  return op->emitError("constant of type ") << type << " is not supported";
}
} // namespace mlir::lox