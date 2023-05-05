//
// License: MIT
//

// Lox.td is the main entry for cmake command add_mlir_dialect()
// It is only used to glue all other td together, no definitions should be added in this file


#ifndef LOX_TD
#define LOX_TD

include "mlir/Dialect/Lox/IR/LoxBase.td"
include "mlir/Dialect/Lox/IR/LoxShapeInferInterface.td"
include "mlir/Dialect/Lox/IR/LoxOps.td"

#endif  // LOX_TD