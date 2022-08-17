
#include "lox/backend/llvm/passes/add_no_alias.h"

#include "llvm/Analysis/AliasAnalysis.h"

using namespace llvm;
namespace lox::llvm_jit {

PreservedAnalyses AddNoAlias::run(Function &F, FunctionAnalysisManager &FAM) {
  bool Modified = false;
  for (auto &Arg : F.args()) {
    if (Arg.getType()->isPointerTy() && !Arg.hasAttribute(Attribute::NoAlias)) {
      Arg.addAttr(Attribute::NoAlias);
      Modified |= true;
    }
  }

  auto PA = PreservedAnalyses::all();
  // this transformation mostly affects alias analysis only, invalidating
  // it when the arguments were modified.
  if (Modified) PA.abandon<AAManager>();

  return PA;
}

}  // namespace lox::llvm_jit