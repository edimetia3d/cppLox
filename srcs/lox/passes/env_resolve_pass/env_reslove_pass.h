//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOVLER_H_
#define CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOVLER_H_

#include "lox/ast/ast.h"
#include "lox/passes/env_resolve_pass/resolve_map.h"
#include "lox/passes/pass.h"
namespace lox {

enum class FunctionType {
  NONE,
  FUNCTION,
  INITIALIZER,
  METHOD,
};
enum class ScopeType {
  NONE,
  CLASS,
  BLOCK,
  FUNCTION,
};
/**
 * EnvResovlePass MUST be synced with evaluator, everytime we create a new scope
 * we should always begin a new scope.
 */
class EnvResovlePass : public Pass {
 public:
  explicit EnvResovlePass(std::shared_ptr<EnvResolveMap> map) : map_(std::move(map)){};
  using Scope = std::map<std::string, bool>;
  void PreNode(AstNode* ast_node) override;
  void PostNode(AstNode* ast_node) override;

 protected:
  void BeginScope(ScopeType type) {
    scopes.push_back(Scope());
    previous_scope_type.push_back(current_scope_type);
    current_scope_type = type;
  }
  void EndScope() {
    scopes.erase(scopes.end() - 1);
    current_scope_type = previous_scope_type.back();
    previous_scope_type.pop_back();
  }
  bool IsInClassScope() {
    auto scope = current_scope_type;
    auto iter = previous_scope_type.rbegin();
    while (iter != previous_scope_type.rend() && scope != ScopeType::CLASS) {
      scope = *iter;
      ++iter;
    }
    return scope == ScopeType::CLASS;
  }
  void Declare(Token token);
  void Define(Token token);
  void ResolveName(ExprBase* state, Token name);
  std::vector<Scope> scopes{Scope()};
  std::shared_ptr<EnvResolveMap> map_;
  std::vector<FunctionType> previous_function_type = {};
  FunctionType current_function_type = FunctionType::NONE;
  std::vector<ScopeType> previous_scope_type = {};
  ScopeType current_scope_type = ScopeType::NONE;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_VISITORS_RESOLVER_PASS_RESOVLER_H_
