//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_NODE_H_
#define CPPLOX_SRCS_LOX_AST_AST_NODE_H_
#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>

#include "lox/lox_object/lox_object.h"
namespace lox {
class AstNode;
template <class T>
concept SubclassOfAstNode = std::is_base_of<AstNode, T>::value;

class AstNodeVisitor;

class AstNode : public std::enable_shared_from_this<AstNode> {
 public:
  template <SubclassOfAstNode SubT, class... Args>
  static std::shared_ptr<SubT> Make(Args... args) {
    return std::shared_ptr<SubT>(new SubT(nullptr, args...));
  }

  virtual object::LoxObject Accept(AstNodeVisitor* visitor) = 0;

  virtual bool IsModified() = 0;

  virtual void ResetModify() = 0;

  template <SubclassOfAstNode T>
  T* DownCast() {
    return dynamic_cast<T*>(this);
  }

  virtual ~AstNode() = default;

  AstNode* Parent() { return parent_; }
  void SetParent(AstNode* parent) {
    assert(parent_ == nullptr || parent_ == parent);  // every element could only has one parent
    parent_ = parent;
  }
  void ResetParent() { parent_ = nullptr; }

 protected:
  // current element is always "owned" by parent through shared_ptr, to avoid cycle reference, just use a weak pointer
  // here
  AstNode* parent_ = nullptr;

  bool is_modified_ = false;

  explicit AstNode(AstNode* parent = nullptr) { SetParent(parent); };
};

static inline bool IsValid(const std::shared_ptr<AstNode>& node) { return node.get(); }

template <SubclassOfAstNode T>
static inline void BindParent(std::shared_ptr<T> element, AstNode* parent) {
  if (element) {
    element->SetParent(parent);
  }
}
template <SubclassOfAstNode T>
static inline void BindParent(const std::vector<std::shared_ptr<T>>& elements, AstNode* parent) {
  for (auto& element : elements) {
    BindParent(element, parent);
  }
}

template <SubclassOfAstNode T>
static inline bool IsModified(std::shared_ptr<T> element) {
  if (IsValid(element)) {
    return element->IsModified();
  }
  return false;
}
template <SubclassOfAstNode T>
static inline bool IsModified(const std::vector<std::shared_ptr<T>>& elements) {
  bool ret = false;
  for (auto& element : elements) {
    ret = ret || IsModified(element);
  }
  return ret;
}

template <SubclassOfAstNode T>
static inline void ResetModify(std::shared_ptr<T> element) {
  if (IsValid(element)) {
    element->ResetModify();
  }
}
template <SubclassOfAstNode T>
static inline void ResetModify(const std::vector<std::shared_ptr<T>>& elements) {
  for (auto& element : elements) {
    ResetModify(element);
  }
}

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

using Expr = std::shared_ptr<AstNode>;
using Stmt = std::shared_ptr<AstNode>;
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_AST_AST_NODE_H_
