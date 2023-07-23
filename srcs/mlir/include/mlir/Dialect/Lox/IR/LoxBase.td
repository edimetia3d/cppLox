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

  // We set this bit to generate the declarations for the dialect's type parsing
  // and printing hooks.
  let useDefaultTypePrinterParser = 1;
}

/**
 * A common base for lox operation.
 */
class LoxOp<string mnemonic, list<Trait> traits = []> : Op<LoxDialect, mnemonic, traits>;

// `StructType` is defined in c++, we provide a ODS declaration to enable it in ods.
def Lox_StructType :
    DialectType<LoxDialect, CPred<"::llvm::isa<StructType>($_self)">,
                "Lox struct type">;

// Some operation may support any kind of type used in lox.
// The corresponding lox types are [Tensor,class,float,bool,str]
def AnyLoxType : AnyTypeOf<[F64Tensor, Lox_StructType, F64, I1, I8MemRef]>;


#endif // LOX_BASE_TD
