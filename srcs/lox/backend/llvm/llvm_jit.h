//
// LICENSE: MIT
//

#ifndef LOX_LLVM_JIT_H
#define LOX_LLVM_JIT_H

#include <memory>

#include "lox/backend/backend.h"

namespace lox::llvm_jit {
class LLVMJITImpl;
class LLVMJIT : public BackEnd {
public:
  LLVMJIT();
  void Run(Scanner &scanner) override;

private:
  std::shared_ptr<LLVMJITImpl> impl_;
};
}; // namespace lox::llvm_jit

#endif // LOX_LLVM_JIT_H
