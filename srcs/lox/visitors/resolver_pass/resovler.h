//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOVLER_H_
#define CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOVLER_H_

#include <cassert>
#include <map>
#include <string>
#include <vector>

#include "lox/ast/expr.h"
#include "lox/ast/stmt.h"
#include "lox/lox_object/lox_object.h"
#include "lox/visitors/evaluator/evaluator.h"

namespace lox {

class Resovler : public StmtVisitor, public ExprVisitor {
 public:
  Resovler(Evaluator* ev) { evaluator_ = ev; }
  using Scope = std::map<std::string, bool>;
  void Resolve(Stmt stmt) {
    assert(stmt.IsValid());
    stmt.Accept(this);
  }

  void Resolve(Expr expr) {
    assert(expr.IsValid());
    expr.Accept(this);
  }

 protected:
  void BeginScope() { scopes.push_back(Scope()); }
  void EndScope() { scopes.erase(scopes.end() - 1); }
  void Declare(Token token);
  void Define(Token token);
  void ResolveLocal(ExprState* state, Token name);
  void ResolveFunction(FunctionStmtState* state);
  object::LoxObject Visit(LogicalState* state) override;
  object::LoxObject Visit(BinaryState* state) override;
  object::LoxObject Visit(GroupingState* state) override;
  object::LoxObject Visit(LiteralState* state) override;
  object::LoxObject Visit(UnaryState* state) override;
  object::LoxObject Visit(VariableState* state) override;
  object::LoxObject Visit(AssignState* state) override;
  object::LoxObject Visit(CallState* state) override;

  object::LoxObject Visit(PrintStmtState* state) override;
  object::LoxObject Visit(ReturnStmtState* state) override;
  object::LoxObject Visit(WhileStmtState* state) override;
  object::LoxObject Visit(BreakStmtState* state) override;
  object::LoxObject Visit(ExprStmtState* state) override;
  object::LoxObject Visit(VarDeclStmtState* state) override;
  object::LoxObject Visit(FunctionStmtState* state) override;
  object::LoxObject Visit(BlockStmtState* state) override;
  object::LoxObject Visit(IfStmtState* state) override;

  std::vector<Scope> scopes{Scope()};
  Evaluator* evaluator_;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOVLER_H_
