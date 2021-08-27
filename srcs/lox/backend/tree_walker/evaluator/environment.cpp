//
// LICENSE: MIT
//

#include "lox/backend/tree_walker/evaluator/environment.h"

#include "lox/backend/tree_walker/error.h"
namespace lox {

void Environment::Define(std::string var_name, object::LoxObject value) {
  map[var_name] = value;  // redefine existing var is permitted
}
void Environment::Remove(std::string var_name) { map.erase(var_name); }
void Environment::Set(std::string var_name, object::LoxObject value) {
  if (map.count(var_name) == 0) {
    if (parent_) {
      return parent_->Set(var_name, value);
    }
    throw "Var Not found";
  }
  map[var_name] = value;
}
object::LoxObject Environment::Get(std::string var_name) {
  if (map.count(var_name) == 0) {
    if (parent_) {
      return parent_->Get(var_name);
    }
    return object::VoidObject();
  }
  return map[var_name];
}
}  // namespace lox
