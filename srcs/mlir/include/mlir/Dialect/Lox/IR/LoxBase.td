//
// License: MIT
//

// Most dialect will have some common def about dialect/types/interface.
// and these definition should have been split into different files.
// But also for most dialect, the code for dialect/types/interface are pretty short,
// so store them in just a xxxBase.td will also be acceptable.

#ifndef LOX_BASE_TD
#define LOX_BASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def LoxDialect : Dialect {
  let name = "lox";

  let summary = "A high-level dialect of the lox programming language that maps to lox AST directly.";

  let description = [{
  The Lox language is a teaching language.
  }];

  let cppNamespace = "::mlir::lox";

  // Canonicalization may fold some value into attribute
  // use ConstantMaterializer to convert these attribute to a constant result.
  let hasConstantMaterializer = 1;

  // When all types are defined in tblgen, we can use ods generated type printer / parser.
  // If not, two methods must be defined
  // mlir::Type LoxDialect::parseType(mlir::DialectAsmParser &parser);
  // void LoxDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer);
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
     void InitTypes();
  }];
}

/**
 * A common base for lox operation.
 */
class LoxOp<string mnemonic, list<Trait> traits = []> : Op<LoxDialect, mnemonic, traits>;

class LoxType<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LoxDialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

// A common base to forward types defined in cpp
class LoxCppType<string cppTypeName, string description> :
    DialectType<LoxDialect, CPred<"::llvm::isa<"#cppTypeName#">($_self)">, description>{
}


#endif // LOX_BASE_TD
