//
// LICENSE: MIT
//

#include "lox/backend/backend.h"
#include "lox/backend/tree_walker/tree_walker.h"
#include "lox/backend/virtual_machine/virtual_machine.h"
#include "lox/common/lox_error.h"

namespace lox {

static void LoadBuiltinBackEnd(BackEndRegistry* registry) {
  registry->Register("TreeWalker", []() { return std::shared_ptr<BackEnd>(new twalker::TreeWalker()); });
  registry->Register("VirtualMachine", []() { return std::shared_ptr<BackEnd>(new vm::VirtualMachine()); });
}

BackEndRegistry& BackEndRegistry::Instance() {
  static BackEndRegistry instance;
  return instance;
}
BackEndRegistry::BackEndRegistry() { LoadBuiltinBackEnd(this); }
void BackEndRegistry::Register(const std::string name, BackEndRegistry::BackEndCreateFn fn) { reg_[name] = fn; }
BackEndRegistry::BackEndCreateFn BackEndRegistry::Get(const std::string& name) {
  if (!reg_.contains(name)) {
    throw LoxError("Backend not found: " + name);
  }
  return reg_[name];
}
}  // namespace lox
