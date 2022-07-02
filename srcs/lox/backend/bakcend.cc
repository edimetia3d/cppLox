//
// LICENSE: MIT
//

#include "lox/ast/ast_printer/ast_printer.h"
#include "lox/backend/backend.h"
#include "lox/backend/llvm/llvm_jit.h"
#include "lox/backend/mlir/mlir_jit.h"
#include "lox/backend/tree_walker/tree_walker.h"
#include "lox/backend/virtual_machine/virtual_machine.h"
#include "lox/common/global_setting.h"
#include "lox/common/lox_error.h"
#include "lox/frontend/parser.h"
#include "lox/passes/pass_runner.h"
#include "lox/passes/semantic_check/semantic_check.h"

namespace lox {

static void LoadBuiltinBackEnd(BackEndRegistry* registry) {
  registry->Register("TreeWalker", []() { return std::shared_ptr<BackEnd>(new twalker::TreeWalker()); });
  registry->Register("VirtualMachine", []() { return std::shared_ptr<BackEnd>(new vm::VirtualMachine()); });
#ifdef ENABLE_MLIR_JIT_BACKEND
  registry->Register("MLIRJIT", []() { return std::shared_ptr<BackEnd>(new mlir_jit::MLIRJIT()); });
  registry->Register("LLVMJIT", []() { return std::shared_ptr<BackEnd>(new llvm_jit::LLVMJIT()); });
#endif
}

BackEndRegistry& BackEndRegistry::Instance() {
  static BackEndRegistry instance;
  return instance;
}
BackEndRegistry::BackEndRegistry() { LoadBuiltinBackEnd(this); }
void BackEndRegistry::Register(const std::string& name, BackEndRegistry::BackEndCreateFn fn) {
  reg_[name] = std::move(fn);
}
BackEndRegistry::BackEndCreateFn BackEndRegistry::Get(const std::string& name) {
  if (!reg_.contains(name)) {
    throw LoxError("Backend not found: " + name);
  }
  return reg_[name];
}

std::unique_ptr<Module> BackEnd::BuildASTModule(Scanner& scanner) {
  auto parser = Parser::Make(GlobalSetting().parser, &scanner);
  auto lox_module = parser->Parse();
  if (!lox_module) {
    throw ParserError("Parse failed");
  }
#ifndef NDEBUG
  if (lox_module && GlobalSetting().debug) {
    AstPrinter printer;
    std::cout << printer.Print(lox_module->ViewAsBlock()) << std::endl;
  }
#endif
  PassRunner pass_runner;
  pass_runner.SetPass({std::make_shared<SemanticCheck>()});
  pass_runner.Run(lox_module.get());
  return lox_module;
}
}  // namespace lox
