//
// LICENSE: MIT
//

#include "lox/backend/backend.h"
#include "lox/backend/tree_walker/tree_walker.h"
namespace lox {
std::map<std::string, BackEnd::BackEndCreateFn>& BackEnd::GetRegistry() {
  static auto ret = std::map<std::string, BackEndCreateFn>();
  return ret;
}
void BackEnd::LoadBuiltinBackEnd() { static auto reg = RegisterBackend<TreeWalker>("TreeWalker"); }
}  // namespace lox
