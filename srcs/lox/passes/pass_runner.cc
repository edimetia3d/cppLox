//
// LICENSE: MIT
//

#include "pass_runner.h"
namespace lox {

lox::Pass::IsModified PassRunner::RunPass(ASTNode* node, Pass* pass) {
  assert(node);
  assert(pass);
  bool is_modified = false;
  is_modified |= Pass::IsModified::YES == pass->PreNode(node);
  for (auto& child : node->Children()) {
    is_modified |= Pass::IsModified::YES == RunPass(child, pass);
  }
  is_modified |= Pass::IsModified::YES == pass->PostNode(node);
  return is_modified ? Pass::IsModified::YES : Pass::IsModified::NO;
}

lox::Pass::IsModified PassRunner::Run(ASTNode* node) {
  assert(node);
  bool is_modified = false;
  for (auto pass : passes_) {
    is_modified |= Pass::IsModified::YES == RunPass(node, pass.get());
  }
  return is_modified ? Pass::IsModified::YES : Pass::IsModified::NO;
}

}  // namespace lox
