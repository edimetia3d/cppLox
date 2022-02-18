#ifndef LOX_DIALECT
#define LOX_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LoxDialect : Dialect {
  let name = "lox";

  let summary = "A high-level dialect of the lox programming language that maps to lox AST directly.";

  let description = [{The Lox language is a teaching language.}];

  let cppNamespace = "::mlir::lox";
}

#endif  // LOX_DIALECT