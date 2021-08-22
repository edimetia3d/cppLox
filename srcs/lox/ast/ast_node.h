//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_NODE_H_
#define CPPLOX_SRCS_LOX_AST_AST_NODE_H_
#include <cassert>
#include <memory>
#include <type_traits>

namespace lox {
class AstNode;
template <class T>
concept SubclassOfAstNode = std::is_base_of<AstNode, T>::value;

class AstNode : public std::enable_shared_from_this<AstNode> {
 public:
  template <SubclassOfAstNode SubT, class... Args>
  static std::shared_ptr<SubT> Make(Args... args) {
    return std::shared_ptr<SubT>(new SubT(nullptr, args...));
  }

  template <SubclassOfAstNode T>
  T* DownCast() {
    return dynamic_cast<T*>(this);
  }

  AstNode* Parent() { return parent_; }
  void SetParent(AstNode* parent) {
    assert(parent_ == nullptr || parent_ == parent);  // every node could only has one parent
    parent_ = parent;
  }
  void ResetParent() { parent_ = nullptr; }
  virtual ~AstNode() = default;

 protected:
  explicit AstNode(AstNode* parent = nullptr) : parent_(parent){};

 private:
  // Every node will hold a shared_ptr to it's sub node, so parent_ is just a weak reference
  AstNode* parent_ = nullptr;
};

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
