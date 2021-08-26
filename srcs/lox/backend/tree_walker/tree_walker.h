//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_BACKEND_TREE_WALKER_TREE_WALKER_H_
#define CPPLOX_SRCS_LOX_BACKEND_TREE_WALKER_TREE_WALKER_H_
#include <memory>

#include "lox/backend/backend.h"
#include "lox/frontend/scanner.h"

namespace lox {
class Evaluator;
class Environment;
class EnvResolveMap;
class TreeWalker : public BackEnd {
 public:
  TreeWalker();
  BackEndErrCode Run(Scanner& scanner) override;

 private:
  std::shared_ptr<Evaluator> evaluator_;
  std::shared_ptr<EnvResolveMap> resolve_map_;
  std::shared_ptr<Environment> global_env_;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_BACKEND_TREE_WALKER_TREE_WALKER_H_
