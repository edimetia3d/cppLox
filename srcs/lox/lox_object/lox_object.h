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
class LoxObjectBase : std::enable_shared_from_this<LoxObjectBase> {
 public:
  template <SubclassOfLoxObject SubT>
  static std::shared_ptr<SubT> Make(typename SubT::RealT v) {
    static_assert(sizeof(SubT) == sizeof(LoxObjectBase));  // Only Base are allowed to hold data
    auto ret = std::shared_ptr<SubT>(static_cast<SubT*>(new LoxObjectBase(v)));
    ret->Init();  // subclass could init extra data here
    return ret;
  }

  virtual LoxObject operator-() { throw "Not supported"; }
  virtual void Init(){};
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

 protected:
  std::shared_ptr<void> raw_value;
  std::shared_ptr<void> extra_data;

 private:
  template <class RealT>
  explicit LoxObjectBase(const RealT& v) : raw_value(new RealT{v}) {}  // disable subclass creation
};

bool IsValid(const LoxObject& obj) { return obj.get(); }
template <SubclassOfLoxObject SubT>
static std::shared_ptr<SubT> MakeLoxObject(typename SubT::RealT v) {
  return LoxObjectBase::Make<SubT>(v);
}

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
