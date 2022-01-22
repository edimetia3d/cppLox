//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_COMMON_FINALLY_H
#define LOX_SRCS_LOX_COMMON_FINALLY_H

#include <functional>
namespace lox {
class Finally {
 public:
  explicit Finally(std::function<void()> fn) : cleaner(std::move(fn)) {}
  ~Finally() { cleaner(); }

 private:
  std::function<void()> cleaner;
};
}  // namespace lox
#endif  // LOX_SRCS_LOX_COMMON_FINALLY_H
