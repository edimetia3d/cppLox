//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_STATE_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_STATE_H_
#include <memory>
#include <string>

#include "lox/token.h"

namespace lox {

namespace object {

class LoxObjectState;
using LoxObjectStatePtr = std::shared_ptr<LoxObjectState>;
struct LoxObjectState {
  template <class RealT>
  LoxObjectState(const RealT &v) : raw_value(new RealT{v}) {}
  std::shared_ptr<void> raw_value;
  virtual LoxObjectStatePtr operator-() { throw "Not supported"; }
  virtual bool IsTrue() { return raw_value.get(); };
  virtual std::string ToString() {
    return std::string("LoxObjectState at ") + std::to_string((uint64_t)raw_value.get());
  };

  template <class T>
  T &AsNative() {
    return *static_cast<T *>(raw_value.get());
  }

  template <class T>
  const T &AsNative() const {
    return *static_cast<T *>(raw_value.get());
  }

  virtual ~LoxObjectState() = default;
};

}  // namespace object
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_STATE_H_
