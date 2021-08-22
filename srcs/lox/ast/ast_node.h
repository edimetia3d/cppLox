//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_NODE_H_
#define CPPLOX_SRCS_LOX_AST_AST_NODE_H_
#include <cassert>

namespace lox {
class AstNode {
 public:
  explicit AstNode(AstNode* parent = nullptr) : parent_(parent){};
  virtual ~AstNode(){};
  AstNode* Parent() { return parent_; }
  void SetParent(AstNode* parent) {
    assert(parent_ == nullptr);  // every node could only has one parent
    parent_ = parent;
  }
  void ResetParent() { parent_ = nullptr; }

 private:
  // Every node will hold a shared_ptr to it's sub node, so parent_ is just a weak reference
  AstNode* parent_ = nullptr;
};
template <class T>
concept SubclassOfAstNode = std::is_base_of<AstNode, T>::value;
template <SubclassOfAstNode Type, SubclassOfAstNode... Rest>
bool MatchAnyType(AstNode* p) {
  if (dynamic_cast<Type*>(p)) {
    return true;
  }
  if constexpr (sizeof...(Rest) > 0) {
    return MatchAnyType<Rest...>();
  }
  return false;
}

template <SubclassOfAstNode Type>
Type* CastTo(AstNode* p) {
  return dynamic_cast<Type*>(p);
}
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_AST_AST_NODE_H_
