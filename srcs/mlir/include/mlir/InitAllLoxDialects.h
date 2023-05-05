//
// License: MIT
//

#ifndef LOX_SRCS_MLIR_INCLUDE_INITALLDIALECTS_H
#define LOX_SRCS_MLIR_INCLUDE_INITALLDIALECTS_H

#include "mlir/Dialect/Lox/IR/LoxDialect.h"

namespace mlir::lox {

/// Add all the MLIR dialects to the provided registry.
inline void registerAllDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<LoxDialect>();
  // clang-format on
}

/// Append all the MLIR dialects to the registry contained in the given context.
inline void registerAllDialects(MLIRContext &context) {
  DialectRegistry registry;
  lox::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace mlir::lox

#endif // LOX_SRCS_MLIR_INCLUDE_INITALLDIALECTS_H
