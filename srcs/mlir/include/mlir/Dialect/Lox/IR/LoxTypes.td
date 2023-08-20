
include "mlir/Dialect/Lox/IR/LoxBase.td"

def Lox_StructType : LoxType<"Struct", "struct"> {
  let summary = "StructType";
  let description = [{
      All struct type has no name.
  }];

  // parameters will be the "KeyTy" when interning this type
  let parameters = (ins "SmallVector<Type>":$elementTypes);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;


  let extraClassDeclaration = [{
    mlir::Type TypeAt(int index);

    llvm::ArrayRef<mlir::Type> getElementTypes();

  }];
}

// We can technically define our own types, as this is a toy language, but we will really gone too far doing so.
def Lox_NumberType : F<64>;

def Lox_StringType : UnrankedMemRefOf<[I8]>;

def Lox_BoolType :  I<1>;

def Lox_TensorType : UnrankedTensorOf<[F64]>;
