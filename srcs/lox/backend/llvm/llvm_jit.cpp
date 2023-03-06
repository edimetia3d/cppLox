//
// LICENSE: MIT
//

#include "lox/backend/llvm/llvm_jit.h"

#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>

#include <map>

#include "lox/passes/ast_printer/ast_printer.h"
#include "lox/backend/llvm/builtins/builtin.h"
#include "lox/backend/llvm/translation/ast_to_llvm.h"
#include "lox/common/global_setting.h"
#include "lox/frontend/parser.h"

namespace lox::llvm_jit {

class RuntimeError : public LoxErrorWithExitCode<EX_SOFTWARE> {
  using LoxErrorWithExitCode<EX_SOFTWARE>::LoxErrorWithExitCode;
};

class LLVMJITImpl : public BackEnd {
 public:
  LLVMJITImpl() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    auto expectJIT = llvm::orc::LLJITBuilder().create();
    if (!expectJIT) throw RuntimeError(toString(expectJIT.takeError()));
    JIT_ = std::move(expectJIT.get());
    // export symbols in current process
    auto &DL = JIT_->getDataLayout();
    auto DLSG = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(DL.getGlobalPrefix());
    if (!DLSG) throw RuntimeError(toString(DLSG.takeError()));
    JIT_->getMainJITDylib().addGenerator(std::move(*DLSG));

    // register builtin functions
    TSCtx_ = llvm::orc::ThreadSafeContext(std::make_unique<llvm::LLVMContext>());
    RegBuiltin(*TSCtx_.getContext(), &known_global_symbol_);
  }
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

  void InvokeInitAndDiscard(std::unique_ptr<llvm::Module> init_module);

 private:
  std::map<llvm::Module *, std::unique_ptr<llvm::legacy::FunctionPassManager>> FPM_;
  std::unique_ptr<llvm::orc::LLJIT> JIT_;
  llvm::orc::ThreadSafeContext TSCtx_;
  KnownGlobalSymbol known_global_symbol_;
  void AddModule(std::unique_ptr<llvm::Module> ll_module, llvm::orc::ResourceTrackerSP RT = nullptr) {
    if (RT) {
      if (auto Err = this->JIT_->addIRModule(RT, llvm::orc::ThreadSafeModule(std::move(ll_module), TSCtx_)))
        throw RuntimeError(toString(std::move(Err)));
    } else {
      if (auto Err = this->JIT_->addIRModule(llvm::orc::ThreadSafeModule(std::move(ll_module), TSCtx_)))
        throw RuntimeError(toString(std::move(Err)));
    }
  }

  template <class RetT>
  RetT Invoke(const std::string &name) {
    // look up
    auto EntrySym = JIT_->lookup(name);
    if (!EntrySym) throw RuntimeError(toString(EntrySym.takeError()));
    // call
    auto *Entry = EntrySym->template toPtr<RetT (*)()>();
    return Entry();
  }
};

LLVMJIT::LLVMJIT() { impl_ = std::make_shared<LLVMJITImpl>(); }
void LLVMJIT::Run(Scanner &scanner) { impl_->Run(scanner); }

void LLVMJITImpl::Run(Scanner &scanner) {
  auto lox_module = BuildASTModule(scanner);

  auto converted_module = ConvertASTToLLVM(*TSCtx_.getContext(), lox_module.get(), &known_global_symbol_);
  if (!converted_module.init_module || !converted_module.def_module) {
    throw ParserError("Translation failed");
  }
  if (lox::GlobalSetting().opt_level > 0) {
    for (auto &fn : converted_module.def_module->functions()) {
      OptimiazeFn(converted_module.def_module.get(), &fn);
    }
    for (auto &fn : converted_module.init_module->functions()) {
      OptimiazeFn(converted_module.init_module.get(), &fn);
    }
  }
  if (lox::GlobalSetting().debug) {
    printf("===========================DEF MODULE IR==========================\n");
    converted_module.def_module->print(llvm::outs(), nullptr);
    printf("===========================INIT MODULE IR==========================\n");
    converted_module.init_module->print(llvm::outs(), nullptr);
    printf("============================================================\n");
    printf("Press Enter to start/continue execution.\n");
    getchar();
  }
  auto main_fn = converted_module.def_module->getFunction("main");
  bool ret_void = true;
  if (main_fn) {
    ret_void = main_fn->getReturnType()->isVoidTy();
  }
  AddModule(std::move(converted_module.def_module));
  InvokeInitAndDiscard(std::move(converted_module.init_module));
  if (!GlobalSetting().interactive_mode && main_fn) {
    if (ret_void) {
      Invoke<void>("main");
    } else {
      Invoke<double>("main");
    }
  }
}
void LLVMJITImpl::InvokeInitAndDiscard(std::unique_ptr<llvm::Module> init_module) {
  auto RT = JIT_->getMainJITDylib().createResourceTracker();
  AddModule(std::move(init_module), RT);
  Invoke<void>("__lox_init_module");
  if (auto err = RT->remove()) {
    throw RuntimeError(toString(std::move(err)));
  }
}

}  // namespace lox::llvm_jit
