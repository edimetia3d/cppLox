
#include "lox/backend/llvm/passes/op_counter.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>
using namespace llvm;

static constexpr const char *HaltFuncName = "my_halt";

namespace lox::llvm_jit {
PreservedAnalyses HaltAnalyzer::run(Function &F, FunctionAnalysisManager &FAM) {
  auto preserv_all = PreservedAnalyses::all();

  auto calls = findHaltCalls(F);
  if (calls.empty()) return preserv_all;

  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);

  for (auto *I : calls) {
    auto *BB = I->getParent();
    SmallVector<BasicBlock *, 4> DomBBs;
    DT.getDescendants(BB, DomBBs);

    for (auto *DomBB : DomBBs) {
      // exclude self
      if (DomBB != BB) {
        DomBB->printAsOperand(errs() << "Unreachable: ");
        errs() << "\n";
      }
    }
  }

  return preserv_all;
}

SmallVector<llvm::Instruction *, 2> HaltAnalyzer::findHaltCalls(Function &F) {
  SmallVector<llvm::Instruction *, 2> calls;
  for (auto &I : instructions(F)) {
    if (auto *CI = dyn_cast<CallInst>(&I)) {
      if (CI->getCalledFunction()->getName() == HaltFuncName) calls.push_back(&I);
    }
  }
  return calls;
}
}  // namespace lox::llvm_jit
