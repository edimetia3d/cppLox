#ifndef LOX_TD
#define LOX_TD

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


include "mlir/Dialect/Lox/IR/LoxBase.td"

def LoxDialect : Dialect {
  let name = "lox";

  let summary = "A high-level dialect of the lox programming language that maps to lox AST directly.";

  let description = [{The Lox language is a teaching language.}];

  let cppNamespace = "::mlir::lox";

  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;

  // We set this bit to generate a declaration of the `materializeConstant`
  // method so that we can materialize constants for our toy operations.
  let hasConstantMaterializer = 1;

  // We set this bit to generate the declarations for the dialect's type parsing
  // and printing hooks.
  let useDefaultTypePrinterParser = 1;
}

class LoxOp<string mnemonic, list<Trait> traits = []> : Op<LoxDialect, mnemonic, traits>;

// Provide a definition for the StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef. We use `DialectType`
// to demarcate the StructType as belonging to the dialect.
def LoxStructType :
    DialectType<LoxDialect, CPred<"$_self.isa<StructType>()">,
                "Lox struct type">;

// Provide a definition of the types that are used within the dialect.
def LoxType : AnyTypeOf<[F64Tensor, LoxStructType]>;

def ConstantOp : LoxOp<"constant", [ConstantLike,
                                    NoSideEffect,
                                    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Create a constant Tensor.";

  let description = [{
    Tensor operation turns a literal into an SSA value.The data is attached to the operation as an attribute.For
    example:

        %0 = "lox.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];


  let arguments = (ins TypedAttrInterface:$value);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let hasVerifier = 1;

  // Set the folder bit so that we can implement constant folders.
  let hasFolder = 1;

  let builders = [
    // Build a constant tensor from DenseElementsAttr, when is_plain_array is true, result type will change to memref
    OpBuilder<(ins "DenseElementsAttr": $value)>,
    // Build a constant string(memref) from StringRef
    OpBuilder<(ins "llvm::StringRef": $value)>,
    // Build a constant Float64Type from double
    OpBuilder<(ins "double":$value)>,
    // Build a constant IntegerType from bool
    OpBuilder<(ins "bool":$value)>,
  ];
}

def AddOp : LoxOp<"add", [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];
}

def CastOp : LoxOp<"cast", [
  DeclareOpInterfaceMethods<CastOpInterface>, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>, NoSideEffect,
  SameOperandsAndResultShape
]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type without changing any data elements
        .The source and destination types must both be tensor types with the same element type.If both are ranked,
        then shape is required to match.The operation is invalid if converting to a mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor : $input);
  let results = (outs F64Tensor : $output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

def FuncOp
    : LoxOp<"func", [DeclareOpInterfaceMethods<CallableOpInterface>, FunctionOpInterface, IsolatedFromAbove, Symbol]> {
  let summary = "user defined function operation";
  let description = [{
    The "lox.func" operation represents a user defined function.These are callable SSA -
        region operations that contain computations
            .

        Example :

    ```mlir lox.func @main() {
      %0 = lox.tensor dense<5.500000e+00> : tensor<f64> 
      %1 = lox.reshape(%0: tensor<f64>) to tensor<2x2xf64>
      lox.print %1 : tensor<2x2xf64>
      lox.return
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr : $sym_name, TypeAttrOf<FunctionType> : $function_type);

  let regions = (region AnyRegion : $body);

  let builders = [OpBuilder<(ins
    "StringRef":$name,
    "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>",
    "{}">:$attrs)
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
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<LoxType>:$inputs);

  // The generic call operation returns a single value of TensorType.
  let results = (outs LoxType);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

def MulOp : LoxOp<"mul", [
                   NoSideEffect,
                   DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
                   ]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Specify a parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def PrintOp : LoxOp<"print"> {
  let summary = "print operation";
  let description = [{The "print" builtin operation prints a given input tensor, and produces no results.}];

  // The print operation takes an input tensor to print.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]> : $input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReshapeOp : LoxOp<"reshape", [NoSideEffect]> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with the same number of elements but different
        shapes.For example:

    ```mlir % 0 = lox.reshape(% arg1
                               : tensor<10xf64>) to tensor<5x2xf64>
    ```
  }];

  let arguments = (ins F64Tensor : $input);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)}];

  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[F64]>);
}

def ReturnOp : LoxOp<"return", [NoSideEffect, HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a
        function.The operation takes an optional tensor operand and produces no
            results.The operand type must match the signature of the function that contains the operation
                .For example :

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        lox.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<LoxType>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0;
}
}];

// Indicate that additional verification for this operation is necessary.
let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StructAccessOp
//===----------------------------------------------------------------------===//

def StructAccessOp : LoxOp<"struct_access", [NoSideEffect]> {
  let summary = "struct access";
  let description = [{
    Access the Nth element of a value returning a struct type.
  }];

  let arguments = (ins LoxStructType:$input, I64Attr:$index);
  let results = (outs LoxType:$output);

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


def TransposeOp : LoxOp<"transpose", [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
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

#endif  // LOX_TD