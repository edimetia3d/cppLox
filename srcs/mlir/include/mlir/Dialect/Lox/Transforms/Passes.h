
#ifndef MLIR_LOX_PASSES_H
#define MLIR_LOX_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace lox {
std::unique_ptr<Pass> createShapeInferencePass();
} // namespace lox
} // end namespace mlir

#endif // MLIR_LOX_PASSES_H
