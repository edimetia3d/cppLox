//
// LICENSE: MIT
//

#include "lox/backend/tree_walker/evaluator/environment.h"

#include "lox/backend/tree_walker/evaluator/runtime_error.h"

namespace lox::twalker {

void Environment::Define(std::string var_name, ObjectPtr value) {
  map[var_name] = value;  // redefine existing var is permitted
}

bool Environment::Set(std::string var_name, ObjectPtr value) {
  if (!map.contains(var_name)) {
    if (parent_) {
      return parent_->Set(var_name, value);
    }
    return false;
  }
  map[var_name] = value;
  return true;
}
ObjectPtr Environment::Get(std::string var_name) {
  if (!map.contains(var_name)) {
    if (parent_) {
      return parent_->Get(var_name);
    }
    return NullObject();
  }
  return map[var_name];
}
}  // namespace lox::twalker
