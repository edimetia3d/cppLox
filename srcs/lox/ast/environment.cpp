//
// LICENSE: MIT
//

#include "lox/ast/environment.h"

#include "lox/error.h"
namespace lox {

void Environment::Define(std::string var_name, object::LoxObject value) {
  map[var_name] = value;  // redefine existing var is permitted
}
void Environment::Remove(std::string var_name) { map.erase(var_name); }
void Environment::Set(std::string var_name, object::LoxObject value) {
  if (map.count(var_name) == 0) {
    throw "Var Not found";
  }
  map[var_name] = value;
}
object::LoxObject Environment::Get(std::string var_name) {
  if (map.count(var_name) == 0) {
    throw "Var Not found";
  }
  return map[var_name];
}
}  // namespace lox
