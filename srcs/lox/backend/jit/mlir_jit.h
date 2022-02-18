//
// LICENSE: MIT
//

#ifndef LOX_MLIR_JIT_H
#define LOX_MLIR_JIT_H

#include "lox/backend/backend.h"

namespace lox::jit {
  class MLIRJIT : public BackEnd {
   public:
    void Run(Scanner& scanner) override;
  };
};

#endif  // LOX_MLIR_JIT_H
