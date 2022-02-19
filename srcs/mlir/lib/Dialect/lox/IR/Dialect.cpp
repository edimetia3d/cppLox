
#include "mlir/Dialect/lox/Dialect.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Transforms/InliningUtils.h>

using namespace mlir;
using namespace mlir::lox;

#include "mlir/Dialect/lox/loxDialect.cpp.inc"

struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable, bool wouldBeCloned) const final { return true; }

  bool isLegalToInline(Operation *, Region *, bool, BlockAndValueMapping &) const final { return true; }

  void handleTerminator(Operation *op, ArrayRef<Value> valuesToRepl) const final {
    // Only "return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  Operation *materializeCallConversion(OpBuilder &builder, Value input, Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};

void LoxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/lox/lox.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
}

static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
  llvm::SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = type.dyn_cast<FunctionType>()) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc, result.operands)) return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands)) return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << op->getName() << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(), [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

void TensorOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  TensorOp::build(builder, state, dataType, dataAttribute);
}

static mlir::ParseResult parseTensorOp(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

static void print(mlir::OpAsmPrinter &printer, TensorOp op) {
  printer << "lox.tensor ";
  printer.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << op.value();
}

static mlir::LogicalResult verify(TensorOp op) {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType) return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = op.value().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return op.emitOpError(
               "return type must match the one of the attached value "
               "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return op.emitOpError("return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim] << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}


void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, StringRef callee,
                          ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

static mlir::LogicalResult verify(ReturnOp op) {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>(op->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (op.getNumOperands() > 1) return op.emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError() << "does not return the same number of values (" << op.getNumOperands()
                            << ") as the enclosing function (" << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!op.hasOperand()) return mlir::success();

  auto inputType = *op.operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return op.emitError() << "type of return operand (" << inputType << ") doesn't match function result type ("
                        << resultType << ")";
}

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

static mlir::LogicalResult verify(TransposeOp op) {
  auto inputType = op.getOperand().getType().dyn_cast<RankedTensorType>();
  auto resultType = op.getType().dyn_cast<RankedTensorType>();
  if (!inputType || !resultType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(), resultType.getShape().rbegin())) {
    return op.emitError() << "expected result shape to be a transpose of the input";
  }
  return mlir::success();
}

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  // The inputs must be Tensors with the same element type.
  TensorType input = inputs.front().dyn_cast<TensorType>();
  TensorType output = outputs.front().dyn_cast<TensorType>();
  if (!input || !output || input.getElementType() != output.getElementType()) return false;
  // The shape is required to match if both types are ranked.
  return !input.hasRank() || !output.hasRank() || input == output;
}

CallInterfaceCallable GenericCallOp::getCallableForCallee() { return (*this)->getAttrOfType<SymbolRefAttr>("callee"); }

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return inputs(); }

#define GET_OP_CLASSES
#include "mlir/Dialect/lox/lox.cpp.inc"
