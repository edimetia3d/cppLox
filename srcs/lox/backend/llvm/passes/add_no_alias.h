#ifndef LOX_BACKEND_LLVM_PASSES_STRICT_OPT
#define LOX_BACKEND_LLVM_PASSES_STRICT_OPT
#include "llvm/IR/PassManager.h"

namespace lox::llvm_jit {
/**
 * A function pass to add no alias attribute to pointer args
 */
struct AddNoAlias : public llvm::PassInfoMixin<AddNoAlias> {
  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
  // PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) // this will be a module pass

  // PreservedAnalyses run(Loop &LP, LoopAnalysisManager &LAM,LoopStandardAnalysisResults &LAR,LPMUpdater &U); // this will be a loop pass
  // LoopStandardAnalysisResults is a proxy to FunctionAnalysisManager, so we can get things liek DominateTree, LoopInfo from it

//  The last argument is used for notifying PassManager of any newly added loops so that it
//can put those new loops into the queue before processing them later. It can also tell the
//PassManager to put the current loop into the queue again
};
}  // namespace lox::llvm_jit
#endif