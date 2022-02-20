//
// LICENSE: MIT
//

#ifndef LOX_MLIR_JIT_H
#define LOX_MLIR_JIT_H

#include <memory>

#include "lox/backend/backend.h"

namespace lox::jit {
class MLIRJITImpl;
class MLIRJIT : public BackEnd {
 public:
  MLIRJIT();
  void Run(Scanner& scanner) override;

 private:
  std::shared_ptr<MLIRJITImpl> impl_;
};
};  // namespace lox::jit

#endif  // LOX_MLIR_JIT_H
