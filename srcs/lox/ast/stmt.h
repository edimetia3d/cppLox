//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_STMT_H_
#define CPPLOX_SRCS_LOX_AST_STMT_H_

#include <type_traits>
#include <vector>

#include "lox/ast/ast_node.h"
#include "lox/ast/expr.h"

namespace lox {
class StmtBase;
using Stmt = std::shared_ptr<StmtBase>;
template <class T>
concept SubclassOfStmt = std::is_base_of<StmtBase, T>::value;

class StmtVisitor;
class StmtBase : public AstNode {
 public:
  ~StmtBase() {}

  virtual object::LoxObject Accept(StmtVisitor* visitor) = 0;

 protected:
  StmtBase(StmtBase* parent = nullptr) : AstNode(parent){};
};

static inline bool IsValid(const Stmt& stmt) { return stmt.get(); }

template <SubclassOfStmt SubT, class... Args>
std::shared_ptr<SubT> MakeStmt(Args... args) {
  return AstNode::Make<SubT, Args...>(args...);
}
}  // namespace lox

#ifdef DYNAMIC_GEN_DECL
#include "lox/ast/stmt_decl_dynamic.h.inc"
#else
#include "lox/ast/stmt_decl.h.inc"
#endif

#endif  // CPPLOX_SRCS_LOX_AST_STMT_H_
