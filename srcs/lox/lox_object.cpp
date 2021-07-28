//
// LICENSE: MIT
//

#include "lox_object.h"

namespace lox {
namespace object {

class LoxObjectOperator : public std::enable_shared_from_this<LoxObjectOperator> {
 public:
  virtual ~LoxObjectOperator(){};

  virtual LoxObject operator-(LoxObject *) = 0;
  virtual std::string ToString(const LoxObject *) = 0;
  virtual bool IsTrue(const LoxObject *obj) const { return obj != nullptr; }

  std::shared_ptr<LoxObjectOperator> share_this() { return shared_from_this(); }
};

/**
 * Always return a new LoxObject
 */
template <class RealType>
class OutplaceOperator : public LoxObjectOperator {
 public:
  using RealT = RealType;

  LoxObject operator-(LoxObject *obj) override { return LoxObject(share_this(), -1 * obj->template AsNative<RealT>()); }
  std::string ToString(const LoxObject *obj) override { return std::to_string(obj->AsNative<RealT>()); }

  bool IsTrue(const LoxObject *obj) const override { return static_cast<bool>(obj->AsNative<RealT>()); }
};

/**
 * Always return the raw LoxObject
 */
class InplaceOperator : public LoxObjectOperator {};

using Number = OutplaceOperator<double>;
using Bool = OutplaceOperator<bool>;

template <>
LoxObject Bool::operator-(LoxObject *obj) {
  return LoxObject(new Number(), static_cast<double>(-1 * obj->AsNative<RealT>()));
}

using String = OutplaceOperator<std::string>;

template <>
LoxObject String::operator-(LoxObject *obj) {
  throw "Not supported";
}
template <>
std::string String::ToString(const LoxObject *obj) {
  return obj->AsNative<std::string>();
}
template <>
bool String::IsTrue(const LoxObject *obj) const {
  return LoxObjectOperator::IsTrue(obj);
}

LoxObject LoxObject::operator-() { return lox_operator->operator-(this); }
std::string LoxObject::ToString() { return lox_operator->ToString(this); }
LoxObject::operator bool() const { return lox_operator->IsTrue(this); }
LoxObject::LoxObject(bool v) : LoxObject(new Bool(), v) {}
LoxObject::LoxObject(double v) : LoxObject(new Number(), v) {}
LoxObject::LoxObject(const std::string &v) : LoxObject(new String(), v) {}
LoxObject LoxObject::operator!() { return LoxObject(static_cast<bool>(*this)); }
}  // namespace object
}  // namespace lox
