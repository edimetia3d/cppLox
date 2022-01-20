//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_BACKEND_BACKEND_H_
#define CPPLOX_SRCS_LOX_BACKEND_BACKEND_H_
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "lox/frontend/scanner.h"
#include "lox/lox_error.h"
namespace lox {
class BackEnd {
 public:
  using BackEndCreateFn = std::function<std::shared_ptr<BackEnd>()>;
  static std::shared_ptr<BackEnd> CreateBackEnd(const std::string &name) {
    LoadBuiltinBackEnd();
    return GetRegistry()[name]();
  }

  static void Register(const std::string name, BackEndCreateFn fn) { GetRegistry()[name] = fn; }

  virtual void Run(Scanner &scanner) = 0;

  static void LoadBuiltinBackEnd();

 private:
  static std::map<std::string, BackEndCreateFn> &GetRegistry();
};

template <class ConcreteBackend>
struct RegisterBackend {
  RegisterBackend(const char *name) {
    BackEnd::Register(name, []() { return std::make_shared<ConcreteBackend>(); });
  }
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_BACKEND_BACKEND_H_
