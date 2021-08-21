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
#include "lox/passes/env_resolve_pass/resolve_map.h"
#include "lox/visitors/evaluator/evaluator.h"
namespace lox {

enum class FunctionType {
  NONE,
  FUNCTION,
  INITIALIZER,
  METHOD,
};

class Resovler : public StmtVisitor, public ExprVisitor {
 public:
  Resovler(std::shared_ptr<EnvResolveMap> map) : map_(std::move(map)) {}
  using Scope = std::map<std::string, bool>;
  void Resolve(Stmt stmt) {
    assert(IsValid(stmt));
    stmt->Accept(this);
  }

  void Resolve(Expr expr) {
    assert(IsValid(expr));
    expr->Accept(this);
  }

 protected:
  void BeginScope() { scopes.push_back(Scope()); }
  void EndScope() { scopes.erase(scopes.end() - 1); }
  void Declare(Token token);
  void Define(Token token);
  void ResolveLocal(ExprBase* state, Token name);
  void ResolveFunction(FunctionStmt* state, FunctionType type);
  object::LoxObject Visit(LogicalExpr* state) override;
  object::LoxObject Visit(BinaryExpr* state) override;
  object::LoxObject Visit(GroupingExpr* state) override;
  object::LoxObject Visit(LiteralExpr* state) override;
  object::LoxObject Visit(UnaryExpr* state) override;
  object::LoxObject Visit(VariableExpr* state) override;
  object::LoxObject Visit(AssignExpr* state) override;
  object::LoxObject Visit(CallExpr* state) override;
  object::LoxObject Visit(GetAttrExpr* state) override;
  object::LoxObject Visit(SetAttrExpr* state) override;

  object::LoxObject Visit(PrintStmt* state) override;
  object::LoxObject Visit(ReturnStmt* state) override;
  object::LoxObject Visit(WhileStmt* state) override;
  object::LoxObject Visit(BreakStmt* state) override;
  object::LoxObject Visit(ExprStmt* state) override;
  object::LoxObject Visit(VarDeclStmt* state) override;
  object::LoxObject Visit(FunctionStmt* state) override;
  object::LoxObject Visit(ClassStmt* state) override;
  object::LoxObject Visit(BlockStmt* state) override;
  object::LoxObject Visit(IfStmt* state) override;

  std::vector<Scope> scopes{Scope()};
  std::shared_ptr<EnvResolveMap> map_;
  FunctionType current_function_type = FunctionType::NONE;
  int while_loop_level = 0;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOVLER_H_
