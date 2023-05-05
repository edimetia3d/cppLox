//
// License: MIT
//

#ifndef LOX_SRCS_MLIR_LIB_DIALECT_LOX_IR_CONSTANTOPVERIFY_H
#define LOX_SRCS_MLIR_LIB_DIALECT_LOX_IR_CONSTANTOPVERIFY_H
#include "mlir/IR/Operation.h"
namespace mlir::lox {
mlir::LogicalResult verifyConstantForType(mlir::Type type, mlir::Attribute opaqueValue, mlir::Operation *op);
}
#endif // LOX_SRCS_MLIR_LIB_DIALECT_LOX_IR_CONSTANTOPVERIFY_H
