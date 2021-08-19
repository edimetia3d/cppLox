//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOVLER_H_
#define CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOVLER_H_

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lox/ast/expr.h"
#include "lox/ast/stmt.h"
#include "lox/lox_object/lox_object.h"
#include "lox/visitors/evaluator/evaluator.h"
#include "lox/visitors/resolver_pass/resolve_map.h"
namespace lox {

enum class FunctionType {
  NONE,
  FUNCTION,
  METHOD,
};

class Resovler : public StmtVisitor, public ExprVisitor {
 public:
  Resovler(std::shared_ptr<EnvResolveMap> map) : map_(std::move(map)) {}
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
  void ResolveFunction(FunctionStmtState* state, FunctionType type);
  object::LoxObject Visit(LogicalState* state) override;
  object::LoxObject Visit(BinaryState* state) override;
  object::LoxObject Visit(GroupingState* state) override;
  object::LoxObject Visit(LiteralState* state) override;
  object::LoxObject Visit(UnaryState* state) override;
  object::LoxObject Visit(VariableState* state) override;
  object::LoxObject Visit(AssignState* state) override;
  object::LoxObject Visit(CallState* state) override;
  object::LoxObject Visit(GetAttrState* state) override;
  object::LoxObject Visit(SetAttrState* state) override;

  object::LoxObject Visit(PrintStmtState* state) override;
  object::LoxObject Visit(ReturnStmtState* state) override;
  object::LoxObject Visit(WhileStmtState* state) override;
  object::LoxObject Visit(BreakStmtState* state) override;
  object::LoxObject Visit(ExprStmtState* state) override;
  object::LoxObject Visit(VarDeclStmtState* state) override;
  object::LoxObject Visit(FunctionStmtState* state) override;
  object::LoxObject Visit(ClassStmtState* state) override;
  object::LoxObject Visit(BlockStmtState* state) override;
  object::LoxObject Visit(IfStmtState* state) override;

  std::vector<Scope> scopes{Scope()};
  std::shared_ptr<EnvResolveMap> map_;
  FunctionType current_function_type = FunctionType::NONE;
  int while_loop_level = 0;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOVLER_H_
