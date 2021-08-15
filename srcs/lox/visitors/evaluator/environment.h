//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_ENVIRONMENT_H_
#define CPPLOX_SRCS_LOX_ENVIRONMENT_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lox/lox_object/lox_object.h"
namespace lox {

class Environment {
 public:
  static std::shared_ptr<Environment> Make() { return std::shared_ptr<Environment>(new Environment); };
  static std::shared_ptr<Environment> Make(std::shared_ptr<Environment> parent) {
    return std::shared_ptr<Environment>(new Environment(parent));
  };
  void Define(std::string var_name, object::LoxObject value);
  void Remove(std::string var_name);
  void Set(std::string var_name, object::LoxObject value);
  object::LoxObject Get(std::string var_name);

 private:
  Environment() {}
  Environment(std::shared_ptr<Environment> parent) : parent_(parent) {}
  std::shared_ptr<Environment> parent_;
  std::map<std::string, object::LoxObject> map;
};

}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_ENVIRONMENT_H_
