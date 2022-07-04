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

#include "lox/ast/ast_printer/ast_printer.h"
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

  void InvokeMain(std::unique_ptr<llvm::Module> ll_module, std::unique_ptr<llvm::LLVMContext> context);

 private:
  std::map<llvm::Module *, std::unique_ptr<llvm::legacy::FunctionPassManager>> FPM_;
};

LLVMJIT::LLVMJIT() { impl_ = std::make_shared<LLVMJITImpl>(); }
void LLVMJIT::Run(Scanner &scanner) { impl_->Run(scanner); }

void LLVMJITImpl::Run(Scanner &scanner) {
  auto lox_module = BuildASTModule(scanner);

  auto context = std::make_unique<llvm::LLVMContext>();

  auto ll_module = ConvertASTToLLVM(*context, lox_module.get());
  if (!ll_module) {
    throw ParserError("Translation failed");
  }
  if (lox::GlobalSetting().opt_level > 0) {
    for (auto &fn : ll_module->functions()) {
      OptimiazeFn(ll_module.get(), &fn);
    }
  }
  if (lox::GlobalSetting().debug) {
    printf("===========================LLVM IR==========================\n");
    ll_module->print(llvm::outs(), nullptr);  // todo: remove debug print later
    printf("============================================================\n");
    printf("Press Enter to start/continue execution.\n");
    getchar();
  }
  InvokeMain(std::move(ll_module), std::move(context));
}
void LLVMJITImpl::InvokeMain(std::unique_ptr<llvm::Module> ll_module, std::unique_ptr<llvm::LLVMContext> context) {
  auto ret_t = ll_module->getFunction("main")->getReturnType();
  auto expectJIT = llvm::orc::LLJITBuilder().create();
  if (!expectJIT) throw RuntimeError(toString(expectJIT.takeError()));
  auto JIT = std::move(expectJIT.get());

  if (auto Err = JIT->addIRModule(llvm::orc::ThreadSafeModule(std::move(ll_module), std::move(context))))
    throw RuntimeError("Compilation failed");

  // export symbols in current process
  auto &DL = JIT->getDataLayout();
  auto DLSG = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(DL.getGlobalPrefix());
  if (!DLSG) throw RuntimeError(toString(DLSG.takeError()));
  JIT->getMainJITDylib().addGenerator(std::move(*DLSG));

  // look up
  auto EntrySym = JIT->lookup("main");
  if (!EntrySym) throw RuntimeError(toString(EntrySym.takeError()));

  // call
  if (ret_t->isVoidTy()) {
    auto *Entry = (void (*)())EntrySym->getAddress();
    Entry();
    printf("\nLox main exited\n");
  } else {
    auto *Entry = (double (*)())EntrySym->getAddress();
    double ret = Entry();
    printf("\nLox main exited with code %g\n", ret);
  }
}

}  // namespace lox::llvm_jit
