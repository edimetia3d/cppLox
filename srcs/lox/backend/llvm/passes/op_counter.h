
#ifndef LOX_SRCS_LOX_BACKEND_LLVM_PASSES_OP_COUNTER_H
#define LOX_SRCS_LOX_BACKEND_LLVM_PASSES_OP_COUNTER_H

#include "llvm/IR/PassManager.h"

namespace lox::llvm_jit {
class HaltAnalyzer : public llvm::PassInfoMixin<HaltAnalyzer> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

 private:
  llvm::SmallVector<llvm::Instruction *, 2> findHaltCalls(llvm::Function &F);
};
}  // namespace lox::llvm_jit

#endif  // LOX_SRCS_LOX_BACKEND_LLVM_PASSES_OP_COUNTER_H
