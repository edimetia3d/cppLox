//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
#include <memory>
#include <string>

#include "lox/token.h"

namespace lox {

namespace object {

struct LoxObjectState;
using LoxObjectStatePtr = std::shared_ptr<LoxObjectState>;

struct LoxObjectState {
  template <class RealT>
  LoxObjectState(const RealT& v) : raw_value(new RealT{v}) {}
  std::shared_ptr<void> raw_value;
  virtual LoxObjectStatePtr operator-() = 0;
  virtual LoxObjectStatePtr operator!() = 0;
  virtual bool IsTrue() { return raw_value.get(); };
  virtual std::string ToString() {
    return std::string("LoxObjectState at") + std::to_string((uint64_t)raw_value.get());
  };

  template <class T>
  T& AsNative() {
    return *static_cast<T*>(raw_value.get());
  }

  template <class T>
  const T& AsNative() const {
    return *static_cast<T*>(raw_value.get());
  }

  virtual ~LoxObjectState() = default;
};
class LoxObject {
 public:
  LoxObject() = default;
  LoxObject(LoxObjectStatePtr ptr) : lox_object_state_(ptr) {}
  explicit LoxObject(bool);
  explicit LoxObject(double);
  explicit LoxObject(char* v) : LoxObject(std::string(v)){};
  explicit LoxObject(const std::string&);
  // Uary
  LoxObject operator-();
  LoxObject operator!();

  // Binary
  LoxObject operator-(LoxObject& rhs);
  LoxObject operator+(LoxObject& rhs);
  LoxObject operator*(LoxObject& rhs);
  LoxObject operator/(LoxObject& rhs);
  LoxObject operator==(LoxObject& rhs);
  LoxObject operator!=(LoxObject& rhs);
  LoxObject operator<(LoxObject& rhs);
  LoxObject operator>(LoxObject& rhs);
  LoxObject operator<=(LoxObject& rhs);
  LoxObject operator>=(LoxObject& rhs);
  std::string ToString();
  operator bool() const;

  template <class T>
  T& AsNative() {
    return lox_object_state_->template AsNative<T>();
  }

  template <class T>
  const T& AsNative() const {
    return lox_object_state_->template AsNative<T>();
  }

  LoxObjectStatePtr lox_object_state_;
};
}  // namespace object
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
