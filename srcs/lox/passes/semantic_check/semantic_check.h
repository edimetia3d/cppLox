//
// LICENSE: MIT
//
#ifndef CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_
#define CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_

#include <map>
#include <memory>
#include <vector>

#include "lox/passes/pass.h"
#include "lox/common/lox_error.h"

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

struct FunctionInfo {
  int semantic_depth = 0;
};

class SemanticCheck : public Pass {
 public:
  explicit SemanticCheck() = default;
  IsModified PreNode(ASTNode* ast_node) override;
  IsModified PostNode(ASTNode* ast_node) override;

 protected:
  std::vector<LoopInfo> loop_infos;
  std::vector<FunctionInfo> function_infos;
  std::map<std::string, std::shared_ptr<ClassInfo>> all_classes;
  std::vector<std::shared_ptr<ClassInfo>> class_infos;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_
