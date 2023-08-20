
include "mlir/Dialect/Lox/IR/LoxBase.td"

def Lox_StructType : LoxType<"Struct", "struct"> {
  let summary = "StructType";
  let description = [{
      All struct type has no name.
  }];

  // parameters will be the "KeyTy" when interning this type
  let parameters = (ins "ArrayRef<Type>":$elementTypes);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;


  let extraClassDeclaration = [{
    mlir::Type TypeAt(int index);

    llvm::ArrayRef<mlir::Type> getElementTypes();

  }];
}

// We can technically define our own types, as this is a toy language, but we will really gone too far doing so.
def Lox_NumberType : F64;

def Lox_StringType : MemRefOf<[I8]>;

def Lox_BoolType :  I1;

def Lox_TensorType : TensorOf<[F64]>;
