
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include "mlir/Dialect/Lox/IR/Lox.h"
#include "mlir/Dialect/Lox/IR/LoxDialect.cpp.inc"
#include "mlir/Dialect/Lox/Transforms/ReshapeReWriter.h"
#include "mlir/Dialect/Lox/Transforms/TransposeRewriter.h"

#include "mlir/Dialect/Lox/IR/LoxTypes.cpp.inc"

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

//===----------------------------------------------------------------------===//
// Toy Types
//===----------------------------------------------------------------------===//

namespace mlir::lox {
namespace detail {
/// This class represents the internal storage of the Toy `StructType`.
struct StructTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// A constructor for the type storage instance.
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes) : elementTypes(elementTypes) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) { return llvm::hash_value(key); }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) { return KeyTy(elementTypes); }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>()) StructTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<mlir::Type> elementTypes;
};
} // namespace detail
} // namespace mlir::lox

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

/// Parse an instance of a type registered to the toy dialect.
mlir::Type LoxDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // Parse the element types of the struct.
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    // Check that the type is either a TensorType or another StructType.
    if (!elementType.isa<mlir::TensorType, StructType>()) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}

/// Print an instance of a type registered to the toy dialect.
void LoxDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  StructType structType = type.cast<StructType>();

  // Print the struct type according to the parser format.
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
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