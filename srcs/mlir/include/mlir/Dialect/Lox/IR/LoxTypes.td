
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

