//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_ENVIRONMENT_H_
#define CPPLOX_SRCS_LOX_ENVIRONMENT_H_
#include <map>
#include <string>

#include "lox/lox_object/lox_object.h"
namespace lox {

class Environment {
 public:
  void Define(std::string var_name, object::LoxObject value);
  void Remove(std::string var_name);
  void Set(std::string var_name, object::LoxObject value);
  object::LoxObject Get(std::string var_name);

 private:
  std::map<std::string, object::LoxObject> map;
};

}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_ENVIRONMENT_H_
