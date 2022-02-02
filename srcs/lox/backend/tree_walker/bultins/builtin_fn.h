//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_TREE_WALKER_BULTINS_BUILTIN_FN_H
#define LOX_SRCS_LOX_BACKEND_TREE_WALKER_BULTINS_BUILTIN_FN_H

#include <map>
#include <string>

#include "lox/backend/tree_walker/evaluator/runtime_object.h"

namespace lox::twalker {
std::map<std::string, ObjectPtr> BuiltinCallables();

}  // namespace lox::twalker

#endif  // LOX_SRCS_LOX_BACKEND_TREE_WALKER_BULTINS_BUILTIN_FN_H
