//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_ENVIRONMENT_H_
#define CPPLOX_SRCS_LOX_ENVIRONMENT_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lox/object/object.h"
namespace lox::twalker {
class Environment;
using EnvPtr = std::shared_ptr<Environment>;

class Environment {
public:
  static EnvPtr Make() { return EnvPtr(new Environment); };
  static EnvPtr Make(EnvPtr parent) { return EnvPtr(new Environment(parent)); };

  void Define(std::string var_name, ObjectPtr value);
  bool Set(std::string var_name, ObjectPtr value);
  ObjectPtr Get(std::string var_name);
  EnvPtr Parent() { return parent_; }

private:
  Environment() = default;
  explicit Environment(EnvPtr parent) : parent_(std::move(parent)) {}
  EnvPtr parent_;
  std::map<std::string, ObjectPtr> map;
};

} // namespace lox::twalker

#endif // CPPLOX_SRCS_LOX_ENVIRONMENT_H_
