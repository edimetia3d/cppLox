//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/environment.h"

#include "lox/error.h"
namespace lox {

void Environment::Define(std::string var_name, object::LoxObject value) {
  map[var_name] = value;  // redefine existing var is permitted
}
void Environment::Remove(std::string var_name) { map.erase(var_name); }
void Environment::Set(std::string var_name, object::LoxObject value) {
  if (map.count(var_name) == 0) {
    if (enclosing_) {
      return enclosing_->Set(var_name, value);
    }
    throw "Var Not found";
  }
  map[var_name] = value;
}
object::LoxObject Environment::Get(std::string var_name) {
  if (map.count(var_name) == 0) {
    if (enclosing_) {
      return enclosing_->Get(var_name);
    }
    throw "Var Not found";
  }
  return map[var_name];
}
}  // namespace lox
