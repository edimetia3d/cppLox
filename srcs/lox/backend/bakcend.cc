//
// LICENSE: MIT
//

#include "lox/backend/backend.h"
#include "lox/backend/tree_walker/tree_walker.h"
#include "lox/backend/virtual_machine/virtual_machine.h"
namespace lox {
std::map<std::string, BackEnd::BackEndCreateFn>& BackEnd::GetRegistry() {
  static auto ret = std::map<std::string, BackEndCreateFn>();
  return ret;
}
void BackEnd::LoadBuiltinBackEnd() {
  static auto reg0 = RegisterBackend<TreeWalker>("TreeWalker");
  static auto reg1 = RegisterBackend<vm::VirtualMachine>("VirtualMachine");
}
}  // namespace lox
