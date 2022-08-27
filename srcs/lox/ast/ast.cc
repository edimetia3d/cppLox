//
// LICENSE: MIT
//

#include "ast.h"

namespace lox {
ASTNode *ASTNode::Parent() { return parent_; }
std::vector<ASTNode *> ASTNode::Children() {
  std::vector<ASTNode *> ret;
  for (auto p : children_) {
    if (p->get()) {
      ret.push_back(p->get());
    }
  }
  return ret;
}
bool ASTNode::UpdateChild(const std::unique_ptr<ASTNode> &old, std::unique_ptr<ASTNode> &&replace) {
  for (auto &p : children_) {
    if (p->get() == old.get()) {
      replace->parent_ = this;
      *p = std::move(replace);
      return true;
    }
  }
  return false;
}
bool ASTNode::UpdateChild(int child_index, ASTNodePtr &&replace) {
  if (child_index < 0 || child_index >= children_.size()) {
    return false;
  }
  replace->parent_ = this;
  *(children_[child_index]) = std::move(replace);
  return true;
}

void ASTNode::AddChild(std::unique_ptr<ASTNode> *child) {
  children_.push_back(child);
  if ((*child)) {
    (*child)->parent_ = this;
  }
}

void ASTNode::AddChild(std::vector<std::unique_ptr<ASTNode>> *child) {
  for (auto &c : *child) {
    AddChild(&c);
  }
}

} // namespace lox
