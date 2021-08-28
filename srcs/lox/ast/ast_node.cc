//
// LICENSE: MIT
//

#include "ast_node.h"
namespace lox {
void AstNode::Walk(std::function<void(AstNode *)> fn) {
  std::vector<AstNode *> stk;
  stk.push_back(this);
  while (!stk.empty()) {
    auto back = stk.back();
    stk.pop_back();
    fn(back);
    for (auto child : back->Children()) {
      stk.push_back(child);
    }
  }
}

void AstNode::UpdateChild(std::shared_ptr<AstNode> old_child, std::shared_ptr<AstNode> new_child) {
  remove_child(old_child);
  add_child(new_child);
}
void AstNode::add_child(std::shared_ptr<AstNode> &new_child) {
  if (new_child) {
    new_child->parent_ = this;
  }
}
void AstNode::remove_child(std::shared_ptr<AstNode> &old_child) {
  if (old_child && old_child->parent_ == this) {
    old_child->parent_ = nullptr;
  }
}
void AstNode::UpdateChild(std::vector<std::shared_ptr<AstNode>> old_child,
                          std::vector<std::shared_ptr<AstNode>> new_child) {
  for (auto &old_one : old_child) {
    remove_child(old_one);
  }
  for (auto &new_one : new_child) {
    add_child(new_one);
  }
}
}  // namespace lox
