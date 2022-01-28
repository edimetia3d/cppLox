//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_BACKEND_BACKEND_H_
#define CPPLOX_SRCS_LOX_BACKEND_BACKEND_H_
#include <functional>
#include <map>
#include <memory>

#include "lox/frontend/scanner.h"

namespace lox {
class BackEnd;
struct BackEndRegistry {
  using BackEndCreateFn = std::function<std::shared_ptr<BackEnd>()>;

  static BackEndRegistry &Instance();

  void Register(const std::string name, BackEndCreateFn fn);

  BackEndCreateFn Get(const std::string &name);

 private:
  BackEndRegistry();

  std::map<std::string, BackEndCreateFn> reg_;
};

class BackEnd {
 public:
  virtual void Run(Scanner &scanner) = 0;
};

}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_BACKEND_BACKEND_H_
