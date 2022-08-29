
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include "mlir/Dialect/Lox/IR/Lox.h"
#include "mlir/Dialect/Lox/Transforms/ReshapeReWriter.h"
#include "mlir/Dialect/Lox/Transforms/TransposeRewriter.h"

using namespace mlir;
using namespace mlir::lox;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
namespace mlir::lox {
mlir::LogicalResult verifyConstantForType(mlir::Type type, mlir::Attribute opaqueValue, mlir::Operation *op);
}

void ConstantOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state, DenseElementsAttr value) {
  ConstantOp::build(builder, state, value.getType(), value);
}
void ConstantOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state, llvm::StringRef value) {
  auto resultType = MemRefType::get({(int64_t)value.size() + 1}, builder.getI8Type());
  auto dataAttribute = StringAttr::get(value, resultType);
  ConstantOp::build(builder, state, resultType, dataAttribute);
}
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
  auto resultType = builder.getF64Type();
  auto dataAttribute = builder.getF64FloatAttr(value);
  ConstantOp::build(builder, state, resultType, dataAttribute);
}
void ConstantOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state, bool value) {
  auto resultType = builder.getI1Type();
  auto dataAttribute = builder.getBoolAttr(value);
  ConstantOp::build(builder, state, resultType, dataAttribute);
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
mlir::LogicalResult ConstantOp::verify() { return verifyConstantForType(getResult().getType(), getValue(), *this); }

/// Infer the output shape of the ConstantOp, this is required by the shape
/// inference interface.
void ConstantOp::inferShapes() {
  if (getResult().getType().isa<mlir::TensorType>() and getValue().isa<DenseElementsAttr>()) {
    return getResult().setType(getValue().getType());
  }
  // todo: support other types
  getOperation()->emitError("unable to infer shape of operation without shape "
                            "inference interface");
}

//===----------------------------------------------------------------------===//
// Binary Arith Op
//===----------------------------------------------------------------------===//

#define BINARY_SHAPE_INFER()                                                                                           \
  do {                                                                                                                 \
    getResult().setType(getOperand(0).getType());                                                                      \
  } while (0)
void AddOp::inferShapes() { BINARY_SHAPE_INFER(); }
void SubOp::inferShapes() { BINARY_SHAPE_INFER(); }
void MulOp::inferShapes() { BINARY_SHAPE_INFER(); }
void DivOp::inferShapes() { BINARY_SHAPE_INFER(); }
void ModOp::inferShapes() { BINARY_SHAPE_INFER(); }
#undef BINARY_SHAPE_INFER

//===----------------------------------------------------------------------===//
// Cast Op
//===----------------------------------------------------------------------===//

void CastOp::inferShapes() { getResult().setType(getOperand().getType()); }

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // The inputs must be Tensors with the same element type.
  TensorType input = inputs.front().dyn_cast<TensorType>();
  TensorType output = outputs.front().dyn_cast<TensorType>();
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // The shape is required to match if both types are ranked.
  return !input.hasRank() || !output.hasRank() || input == output;
}

void StructAccessOp::build(mlir::OpBuilder &b, mlir::OperationState &state, mlir::Value input, size_t index) {
  // Extract the result type from the input type.
  StructType structTy = input.getType().cast<StructType>();
  assert(index < structTy.getElementNum());
  mlir::Type resultType = structTy.getElementTypes()[index];

  // Call into the auto-generated build method.
  build(b, state, resultType, input, b.getI64IntegerAttr(index));
}

mlir::LogicalResult StructAccessOp::verify() {
  StructType structTy = getInput().getType().cast<StructType>();
  size_t indexValue = getIndex();
  if (indexValue >= structTy.getElementNum())
    return emitOpError() << "index should be within the range of the input struct type";
  mlir::Type resultType = getResult().getType();
  if (resultType != structTy.getElementTypes()[indexValue])
    return emitOpError() << "must have the same result type as the struct "
                            "element referred to by the index";
  return mlir::success();
}

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType = [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
                          llvm::ArrayRef<mlir::Type> results, mlir::function_interface_impl::VariadicFlag,
                          std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(parser, result, /*allowVariadic=*/false, buildFuncType);
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(p, *this,
                                                 /*isVariadic=*/false);
}

/// Returns the region on the function operation that is callable.
mlir::Region *FuncOp::getCallableRegion() { return &getBody(); }

/// Returns the results types that the callable region produces when
/// executed.
llvm::ArrayRef<mlir::Type> FuncOp::getCallableResults() { return getFunctionType().getResults(); }

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, StringRef callee,
                          ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee", mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() { return (*this)->getAttrOfType<SymbolRefAttr>("callee"); }

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

mlir::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values (" << getNumOperands()
                         << ") as the enclosing function (" << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return emitError() << "type of return operand (" << inputType << ") doesn't match function result type ("
                     << resultType << ")";
}

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

void TransposeOp::inferShapes() {
  auto arrayTy = getOperand().getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

mlir::LogicalResult TransposeOp::verify() {
  auto inputType = getOperand().getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();
  if (!inputType || !resultType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(), resultType.getShape().rbegin())) {
    return emitError() << "expected result shape to be a transpose of the input";
  }
  return mlir::success();
}
/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  populateTransposeCanonicalRewriter(results);
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  populateReshapeCanonicalRewriter(results);
}

/// Fold constants.
OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) { return getValue(); }

/// Fold simple struct access operations that access into a constant.
OpFoldResult StructAccessOp::fold(ArrayRef<Attribute> operands) {
  auto structAttr = operands.front().dyn_cast_or_null<mlir::ArrayAttr>();
  if (!structAttr)
    return nullptr;

  size_t elementIndex = getIndex();
  return structAttr[elementIndex];
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Lox/IR/Lox.cpp.inc"
