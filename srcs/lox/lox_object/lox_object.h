//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
#include <memory>
#include <string>
#include <type_traits>

#include "lox/lox_object/lox_object_state.h"
#include "lox/token.h"
namespace lox {

namespace object {

class LoxObjectBase;
template <class T>
concept SubclassOfLoxObject = std::is_base_of<LoxObjectBase, T>::value;

using LoxObject = std::shared_ptr<LoxObjectBase>;
struct LoxObjectBase : std::enable_shared_from_this<LoxObjectBase> {
  template <class RealT>
  LoxObjectBase(const RealT& v) : raw_value(new RealT{v}) {}

  LoxObjectBase() : raw_value(nullptr){};
  std::shared_ptr<void> raw_value;
  virtual LoxObject operator-() { throw "Not supported"; }
  virtual bool IsTrue() { return raw_value.get(); };
  virtual std::string ToString() {
    return std::string("LoxObjectBase at ") + std::to_string((uint64_t)raw_value.get());
  };

  template <class T>
  T& AsNative() {
    return *static_cast<T*>(raw_value.get());
  }

  template <class T>
  const T& AsNative() const {
    return *static_cast<T*>(raw_value.get());
  }

  template <SubclassOfLoxObject T>
  T* DownCastState() {
    return dynamic_cast<T*>(this);
  }

  virtual ~LoxObjectBase() = default;
};

bool IsValid(const LoxObject& obj) { return obj.get(); }
LoxObject MakeLoxObject(bool);
LoxObject MakeLoxObject(double);
LoxObject MakeLoxObject(char* v);
LoxObject MakeLoxObject(const std::string&);

// Uary
LoxObject operator-(const LoxObject& self);

// Binary
LoxObject operator-(const LoxObject& lhs, const LoxObject& rhs);
LoxObject operator+(const LoxObject& lhs, const LoxObject& rhs);
LoxObject operator*(const LoxObject& lhs, const LoxObject& rhs);
LoxObject operator/(const LoxObject& lhs, const LoxObject& rhs);
LoxObject operator==(const LoxObject& lhs, const LoxObject& rhs);
LoxObject operator!=(const LoxObject& lhs, const LoxObject& rhs);
LoxObject operator<(const LoxObject& lhs, const LoxObject& rhs);
LoxObject operator>(const LoxObject& lhs, const LoxObject& rhs);
LoxObject operator<=(const LoxObject& lhs, const LoxObject& rhs);
LoxObject operator>=(const LoxObject& lhs, const LoxObject& rhs);

}  // namespace object
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
