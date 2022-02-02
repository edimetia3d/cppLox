//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_TREE_WALKER_EVALUATOR_RUNTIME_ERROR_H
#define LOX_SRCS_LOX_BACKEND_TREE_WALKER_EVALUATOR_RUNTIME_ERROR_H
#include "lox/common/lox_error.h"

namespace lox::twalker {
class RuntimeError : public LoxErrorWithExitCode<EX_SOFTWARE> {
 public:
  using LoxErrorWithExitCode<EX_SOFTWARE>::LoxErrorWithExitCode;
};
}  // namespace lox::twalker
#endif  // LOX_SRCS_LOX_BACKEND_TREE_WALKER_EVALUATOR_RUNTIME_ERROR_H
