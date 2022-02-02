//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_NODE_H_
#define CPPLOX_SRCS_LOX_AST_AST_NODE_H_

#include <memory>
#include <type_traits>
#include <vector>

namespace lox {

/**
 * A forward decl to make IASTNodeVisitor valid
 */
class IASTNodeVisitor;

/**
 * An ASTNode could have any extra data as it's attribute, anything that is not a child node should be stored in it's
 * attribute.
 */
struct ASTNodeAttr {};

class ASTNode;
template <class T>
concept ConcreteASTNode = std::is_base_of<ASTNode, T>::value;
using ASTNodePtr = std::unique_ptr<ASTNode>;

/**
 * Expr and Stmt are both just an name alias of ASTNode, to make things clear.
 *
 * All ASTNode are created as unique_ptr. The parent node owns all its child,
 * an ASTNode will and will only have one parent node.
 */
class ASTNode {
 public:
  ASTNode() = default;

  ASTNode(const ASTNode&) = delete;
  ASTNode(ASTNode&&) = delete;

  template <ConcreteASTNode SubT, class... Args>
  static std::unique_ptr<SubT> Make(Args&&... args) {
    return std::unique_ptr<SubT>(new SubT(std::forward<Args>(args)...));
  }

  virtual void Accept(IASTNodeVisitor* visitor) = 0;

  template <ConcreteASTNode SubT>
  SubT* As() {
    return static_cast<SubT*>(this);
  }

  template <ConcreteASTNode SubT>
  SubT* DynAs() {
    return dynamic_cast<SubT*>(this);
  }

  ASTNode* Parent();

  std::vector<ASTNode*> Children();

  bool UpdateChild(const ASTNodePtr& old, ASTNodePtr&& replace);

  bool UpdateChild(int child_index, ASTNodePtr&& replace);

  virtual ~ASTNode() = default;

 protected:
  ASTNode* parent_;
  void AddChild(ASTNodePtr* child);

  void AddChild(std::vector<ASTNodePtr>* child);
  std::vector<ASTNodePtr*> children_;

 private:
  void* operator new(size_t size) { return ::operator new(size); }
};

using Expr = ASTNode;
using ExprPtr = std::unique_ptr<Expr>;

using Stmt = ASTNode;
using StmtPtr = std::unique_ptr<Stmt>;

}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_AST_AST_NODE_H_
