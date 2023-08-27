#ifndef LOX_OPS_TD
#define LOX_OPS_TD

/**
 * @brief The LoxOps ODS definitions, LoxOps will be a AST level IR for Lox, i.e., it will
 * contain all AST elements that defined in the "srcs/lox/ast/ast_node_def.yaml"
 */

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


include "mlir/Dialect/Lox/IR/LoxBase.td"
include "mlir/Dialect/Lox/IR/LoxShapeInferInterface.td"
include "mlir/Dialect/Lox/IR/LoxTypes.td"

// Some operation may support any kind of type used in lox.
def AnyLoxType : AnyTypeOf<[Lox_TensorType, Lox_StructType, Lox_NumberType, Lox_BoolType, Lox_StringType]>;
// Only some type can be used in arithmetic operation, and we disabled string concatenation explicitly.
def ArithLoxType : AnyTypeOf<[Lox_TensorType, Lox_NumberType, Lox_BoolType]>;
// todo: add folder to all ops

//===----------------------------------------------------------------------===//
// Literal/ConstantOp
//===----------------------------------------------------------------------===//

// 1. Constant tensor may infer it's shape from the attribute it contains, so it supports the ShapeInferenceOpInterface
// 2. We need a ConstantOp mainly for 3 reasons:
//   a. The MLIR does not support constant natively.
//   b. The constant data in attribute will be stored in the final binary, but we may need some extra instruction to
//      load them, these extra instruction could be welly handled when lowering the ConstantOp.
//   c. ConstantOp is an import source to create values. In MLIR, values comes in only two ways, one is the arguments
//      of BB/Region, the other is the result of operation. ConstantOp will be the main operation to create values without
//      needs to access other operation's result. And because of this, ConstantOp is semantically the default constructor
//      of any lox literal.
def ConstantOp : LoxOp<"constant", [ConstantLike,
                                    Pure,
                                    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Create a constant Tensor.";

  let description = [{
    ConstantOp convert literal/attributes into a concrete SSA value with known Type.

    | LoxType | AttrType (value)  | ResultType (result) |
    | :------ | :---------------- | :------------------ |
    | Tensor  | DenseElementsAttr | F64Tensor           |
    | str     | StringAttr        | I8MemRef            |
    | float   | FloatAttr         | F64                 |
    | bool    | IntAttr           | I1                  |
    | class   | ArrayAttr         | Lox_StructType      |

    For example:

        %0 = "lox.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];


  let arguments = (ins AnyAttr:$value);

  let results = (outs AnyLoxType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let hasVerifier = 1;

  // Sometimes when all input value of an XOp are constexpr,
  // XOp's result may be computed at compilation time, use `hasFolder = 1` to declare this behavior.
  // Note that, the source of constexpr values are:
  //  1. ConstantLike Operation's result.
  //  2. Folded result. (i.e. Constant propagation)
  let hasFolder = 1;

  let builders = [
    // Build a constant tensor
    OpBuilder<(ins "DenseElementsAttr": $value)>,
    // Build a constant string(memref)
    OpBuilder<(ins "llvm::StringRef": $value)>,
    // Build a constant Float64Type
    OpBuilder<(ins "double":$value)>,
    // Build a constant IntegerType
    OpBuilder<(ins "bool":$value)>,
  ];
}

//===----------------------------------------------------------------------===//
// AssignOp/AssignExpr
//===----------------------------------------------------------------------===//

def AssignOp : LoxOp<"assign", [Pure, SameOperandsAndResultType]> {
  let summary = "Give value a internal name, and return the value itself.";

  let arguments = (ins AnyLoxType:$value, StrAttr:$var_name);

  let results = (outs AnyLoxType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  // todo: verify/folder/canonicalizer may needed
  // let hasVerifier = 1;
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// LogicalOp/LogicalExpr
//===----------------------------------------------------------------------===//

class LoxLogicalOp<string mnemonic, list<Trait> traits = []> : LoxOp<mnemonic,
                          traits # [Pure]> {
  let summary = "logical " #mnemonic# " operation";

  let arguments = (ins ArithLoxType:$lhs, ArithLoxType:$rhs);
  let results = (outs Lox_BoolType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";

  // todo: verify/folder/canonicalizer may needed
  // let hasVerifier = 1;
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

def EqOp : LoxLogicalOp<"eq">;
def NEOp : LoxLogicalOp<"ne">;
def GEOp : LoxLogicalOp<"ge">;
def GTOp : LoxLogicalOp<"gt">;
def LEOp : LoxLogicalOp<"le">;
def LTOp : LoxLogicalOp<"lt">;


//===----------------------------------------------------------------------===//
// Binary Arith Op
//===----------------------------------------------------------------------===//

class LoxBinaryOp<string mnemonic, list<Trait> traits = []> : LoxOp<mnemonic,
                          traits # [Pure,
                          SameOperandsAndResultType,
                          DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise " #mnemonic# " operation";

  let arguments = (ins ArithLoxType:$lhs, ArithLoxType:$rhs);
  let results = (outs ArithLoxType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  // todo: add canonicalization
  // todo: add fold
}

def AddOp : LoxBinaryOp<"add">;
def SubOp : LoxBinaryOp<"sub">;
def MulOp : LoxBinaryOp<"mul">;
def DivOp : LoxBinaryOp<"div">;
def ModOp : LoxBinaryOp<"mod">;


//===----------------------------------------------------------------------===//
// GroupingOp/GroupingExpr
//===----------------------------------------------------------------------===//

def GroupingOp : LoxOp<"grouping", [Pure, SameOperandsAndResultType]> {
  let summary = "Grouping an op, and return its value";

  let arguments = (ins AnyLoxType:$value);

  let results = (outs AnyLoxType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  // todo: verify/folder/canonicalizer may needed
  // let hasVerifier = 1;
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}


//===----------------------------------------------------------------------===//
// UnaryOp/UnaryExpr
//===----------------------------------------------------------------------===//

def NegOp : LoxOp<"neg", [Pure, SameOperandsAndResultType]> {
  let summary = "Unary neg operation";

  let arguments = (ins Lox_NumberType:$value);

  let results = (outs Lox_NumberType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  // todo: verify/folder/canonicalizer may needed
  // let hasVerifier = 1;
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

def NotOp : LoxOp<"not", [Pure, SameOperandsAndResultType]> {
  let summary = "Unary not operation";

  let arguments = (ins Lox_BoolType:$value);

  let results = (outs Lox_BoolType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  // todo: verify/folder/canonicalizer may needed
  // let hasVerifier = 1;
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// GetAttrOp/GetAttrExpr
//===----------------------------------------------------------------------===//

def GetAttrOp : LoxOp<"get_attr", [Pure]> {
  let summary = "Unary not operation";

  let arguments = (ins Lox_StructType:$object, StrAttr:$attr_name);

  let results = (outs AnyLoxType:$result);

  let assemblyFormat = "$object `:` type($object) attr-dict `>``>`  type($result)";

  // todo: verify/folder/canonicalizer may needed
  // let hasVerifier = 1;
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

def SetAttrOp : LoxOp<"set_attr", [Pure]> {
  let summary = "Unary not operation";

  let arguments = (ins Lox_StructType:$object, StrAttr:$attr_name, AnyLoxType:$value);

  let assemblyFormat = " $value `:` type($value)  attr-dict `>``>` $object `:` type($object)";

  // todo: verify/folder/canonicalizer may needed
  // let hasVerifier = 1;
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//
def CastOp : LoxOp<"cast", [
  DeclareOpInterfaceMethods<CastOpInterface>,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
  Pure,
]> {
  let summary = "Type cast operation";
  let description = [{
    Cast one type to another type, currently only support following type conversion:
    tensor <-> tensor : Since we only have reshape, so cast on tensor is mainly used to convert element type.
    tensor <-> bool : only support tensor with shape [1]
    tensor <-> float : only support tensor with shape [1]
    float <-> bool : 0.0 -> false, 1.0 -> true
  }];

  let arguments = (ins ArithLoxType : $input);
  let results = (outs ArithLoxType : $output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp
    : LoxOp<"func", [DeclareOpInterfaceMethods<CallableOpInterface>,
                     FunctionOpInterface,
                     IsolatedFromAbove,
                     Symbol]> {
  let summary = "user defined function operation";
  let description = [{
    The "lox.func" operation represents a user defined function.These are callable SSA -
    region operations that contain computations.

    Example :

    ```mlir lox.func @main() {
      %0 = lox.constant dense<5.500000e+00> : tensor<f64>
      %1 = lox.reshape(%0: tensor<f64>) to tensor<2x2xf64>
      lox.print %1 : tensor<2x2xf64>
      lox.return
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr : $sym_name,
                   TypeAttrOf<FunctionType> : $function_type,
                   OptionalAttr<DictArrayAttr> : $arg_attrs,
                   OptionalAttr<DictArrayAttr> : $res_attrs
                   );

  let regions = (region AnyRegion : $body);

  let builders = [OpBuilder<(ins
    "StringRef":$name,
    "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    }];

    let hasCustomAssemblyFormat = 1;

    let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

def GenericCallOp : LoxOp<"generic_call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = lox.generic_call @my_func(%1, %3)
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyLoxType>:$inputs);

  // The generic call of lox returns a single value
  let results = (outs AnyLoxType);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : LoxOp<"print"> {
  let summary = "print operation";

  let arguments = (ins AnyLoxType : $input);
  // fixme: only tensor lowering supported, add more lowering support.

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def ReshapeOp : LoxOp<"reshape", [Pure]> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with the same number of elements but different
        shapes.

    For example:

    ```mlir
     %0 = lox.reshape(% arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
  }];

  let arguments = (ins F64Tensor : $input);

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[F64]>);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)}];

  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;

}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : LoxOp<"return", [Pure, HasParent<"::mlir::lox::FuncOp">, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a
    function.The operation takes an optional tensor operand and produces no
    results.The operand type must match the signature of the function that contains the operation.

    For example :

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        lox.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Optional<AnyLoxType>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, Value()); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() {
      return getNumOperands() != 0;
    }
  }];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StructAccessOp
//===----------------------------------------------------------------------===//

def StructAccessOp : LoxOp<"struct_access", [Pure]> {
  let summary = "struct access";
  let description = [{
    Access the Nth element of a value returning a struct type.
  }];

  let arguments = (ins Lox_StructType:$input, I64Attr:$index);
  let results = (outs AnyLoxType:$output);

  let assemblyFormat = [{
    $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
  }];

  // Allow building a StructAccessOp with just a struct value and an index.
  let builders = [
    OpBuilder<(ins "Value":$input, "size_t":$index)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  // Set the folder bit so that we can fold constant accesses.
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

def TransposeOp : LoxOp<"transpose", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor : $input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)}];

  // Allow building a TransposeOp with from the input operand.
  let builders = [OpBuilder<(ins "Value" : $input)>];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

#endif // LOX_OPS_TD