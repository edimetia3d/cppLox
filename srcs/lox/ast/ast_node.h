//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_NODE_H_
#define CPPLOX_SRCS_LOX_AST_AST_NODE_H_
namespace lox {
class AstNode {
 public:
  virtual ~AstNode(){};
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
