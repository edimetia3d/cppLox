//
// LICENSE: MIT
//

#include "lox/backend/llvm/llvm_jit.h"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/InstSimplifyPass.h>
#include <llvm/Transforms/Utils.h>

#include <map>

#include "lox/ast/ast_printer/ast_printer.h"
#include "lox/backend/llvm/translation/ast_to_llvm.h"
#include "lox/common/global_setting.h"
#include "lox/frontend/parser.h"

namespace lox::llvm_jit {

class LLVMJITImpl : public BackEnd {
 public:
  LLVMJITImpl() {}
  void Run(Scanner &scanner) override;

  void OptimiazeFn(llvm::Module *module, llvm::Function *function) {
    if (!FPM_.contains(module)) {
      auto FPM = std::make_unique<llvm::legacy::FunctionPassManager>(module);
      // Do simple "peephole" optimizations and bit-twiddling optzns.
      FPM->add(llvm::createInstructionCombiningPass());
      // Reassociate expressions.
      FPM->add(llvm::createReassociatePass());
      // Eliminate Common SubExpressions.
      FPM->add(llvm::createGVNPass());
      // Simplify the control flow graph (deleting unreachable blocks, etc).
      FPM->add(llvm::createCFGSimplificationPass());
      // Promote allocas to registers.
      FPM->add(llvm::createPromoteMemoryToRegisterPass());
      // Do simple "peephole" optimizations and bit-twiddling optzns.
      FPM->add(llvm::createInstructionCombiningPass());
      // Reassociate expressions.
      FPM->add(llvm::createReassociatePass());

      FPM->doInitialization();
      FPM_.insert(std::make_pair(module, std::move(FPM)));
    }

    FPM_[module]->run(*function);
  }

 private:
  std::map<llvm::Module *, std::unique_ptr<llvm::legacy::FunctionPassManager>> FPM_;
};

LLVMJIT::LLVMJIT() { impl_ = std::make_shared<LLVMJITImpl>(); }
void LLVMJIT::Run(Scanner &scanner) { impl_->Run(scanner); }

void LLVMJITImpl::Run(Scanner &scanner) {
  auto lox_module = BuildASTModule(scanner);

  auto context = llvm::LLVMContext();

  auto ll_module = ConvertASTToLLVM(context, lox_module.get());
  if (!ll_module) {
    throw ParserError("Translation failed");
  }
  if (lox::GlobalSetting().opt_level > 0) {
    for (auto &fn : ll_module->functions()) {
      OptimiazeFn(ll_module.get(), &fn);
    }
  }
  ll_module->print(llvm::outs(), nullptr);  // todo: remove debug print later
}

}  // namespace lox::llvm_jit
