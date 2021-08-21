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
class StmtBase : public std::enable_shared_from_this<StmtBase>, public AstNode {
 public:
  template <SubclassOfStmt SubT, class... Args>
  static std::shared_ptr<SubT> Make(Args... args) {
    return std::shared_ptr<SubT>(new SubT(args...));
  }
  ~StmtBase() {}

  virtual object::LoxObject Accept(StmtVisitor* visitor) = 0;

  template <SubclassOfStmt T>
  T* DownCast() {
    return dynamic_cast<T*>(this);
  }
};

static inline bool IsValid(const Stmt& stmt) { return stmt.get(); }

template <SubclassOfStmt SubT, class... Args>
std::shared_ptr<SubT> MakeStmt(Args... args) {
  return StmtBase::Make<SubT, Args...>(args...);
}
}  // namespace lox

#ifdef DYNAMIC_GEN_DECL
#include "lox/ast/stmt_decl_dynamic.h.inc"
#else
#include "lox/ast/stmt_decl.h.inc"
#endif

#endif  // CPPLOX_SRCS_LOX_AST_STMT_H_
