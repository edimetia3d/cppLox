//
// LICENSE: MIT
//

#include "lox/backend/llvm/llvm_jit.h"

#include "lox/ast/ast_printer/ast_printer.h"
#include "lox/backend/llvm/translation/ast_to_llvm.h"
#include "lox/frontend/parser.h"

namespace lox::llvm_jit {

class LLVMJITImpl : public BackEnd {
 public:
  LLVMJITImpl() {}
  void Run(Scanner &scanner) override;
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
  ll_module->print(llvm::outs(), nullptr);  // todo: remove debug print later
}

}  // namespace lox::llvm_jit
