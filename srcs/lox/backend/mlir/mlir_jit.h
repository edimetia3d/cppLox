//
// LICENSE: MIT
//

#ifndef LOX_MLIR_JIT_H
#define LOX_MLIR_JIT_H

#include <memory>

#include "lox/backend/backend.h"

namespace lox::mlir_jit {
class MLIRJITImpl;
class MLIRJIT : public BackEnd {
public:
  MLIRJIT();
  void Run(Scanner &scanner) override;

private:
  std::shared_ptr<MLIRJITImpl> impl_;
};
}; // namespace lox::mlir_jit

#endif // LOX_MLIR_JIT_H
