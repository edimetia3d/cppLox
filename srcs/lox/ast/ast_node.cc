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
    for (auto child : back->children_) {
      stk.push_back(child);
    }
  }
}
void AstNode::SetParent(AstNode *parent) {
  assert(parent_ == nullptr || parent_ == parent);  // every element could only has one parent
  parent_ = parent;
}
void AstNode::UpdateChild(std::shared_ptr<AstNode> old_child, std::shared_ptr<AstNode> new_child) {
  if (old_child) {
    children_.erase(old_child.get());
  }
  if (new_child) {
    children_.insert(new_child.get());
  }
}
void AstNode::UpdateChild(std::vector<std::shared_ptr<AstNode>> old_child,
                          std::vector<std::shared_ptr<AstNode>> new_child) {
  for (auto &old_one : old_child) {
    if (old_one) {
      children_.erase(old_one.get());
    }
  }
  for (auto &new_one : new_child) {
    if (new_one) {
      children_.insert(new_one.get());
    }
  }
}
}  // namespace lox
