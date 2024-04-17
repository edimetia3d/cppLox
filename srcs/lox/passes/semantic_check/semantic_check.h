//
// LICENSE: MIT
//
#ifndef CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_
#define CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "lox/common/lox_error.h"
#include "lox/passes/pass.h"

namespace lox {

class SemanticError : public LoxErrorWithExitCode<EX_DATAERR> {
public:
  using LoxErrorWithExitCode<EX_DATAERR>::LoxErrorWithExitCode;
};

struct ClassInfo {
  std::string name;
  std::shared_ptr<ClassInfo> superclass;
};

struct LoopInfo {};

enum class FunctionType {
  NONE,
  FUNCTION,
  METHOD,
  INITIALIZER,
};

enum class ScopeType {
  NONE,
  FUNCTION,
  BLOCK,
  CLASS,
  GLOBAL,
};
struct ScopeInfo {
  ScopeType type;
  std::set<std::string> locals;
};

struct FunctionInfo {
  FunctionInfo(FunctionType type, std::string name) : type(type), name(name) {
    if (name == "<script>") {
      scopes.push_back(ScopeInfo{.type = ScopeType::GLOBAL});
    } else {
      scopes.push_back(ScopeInfo{.type = lox::ScopeType::FUNCTION});
    }
  }
  FunctionType type;
  std::string name;
  std::vector<ScopeInfo> scopes;
};

class SemanticCheck : public Pass {
public:
  SemanticCheck() { function_infos.emplace_back(FunctionInfo(FunctionType::FUNCTION, "<script>")); };
  IsModified PreNode(ASTNode *ast_node) override;
  IsModified PostNode(ASTNode *ast_node) override;

protected:
  std::vector<LoopInfo> loop_infos;
  std::vector<FunctionInfo> function_infos;
  std::map<std::string, std::shared_ptr<ClassInfo>> all_classes;
  std::vector<std::shared_ptr<ClassInfo>> class_infos;
  void AddNamedValue(const std::string &name);
};
} // namespace lox

#endif // CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_
