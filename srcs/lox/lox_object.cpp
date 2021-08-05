//
// LICENSE: MIT
//

#include "lox/lox_object.h"

namespace lox {
namespace object {

class LoxObjectOperator : public std::enable_shared_from_this<LoxObjectOperator> {
 public:
  virtual ~LoxObjectOperator(){};

  virtual LoxObject UaryMinus(LoxObject *) { throw "Not supported"; }
  virtual LoxObject BinaryMinus(LoxObject *lhs, LoxObject *rhs) { throw "Not supported"; }
  virtual LoxObject BinaryPlus(LoxObject *lhs, LoxObject *rhs) { throw "Not supported"; }
  virtual LoxObject BinaryMul(LoxObject *lhs, LoxObject *rhs) { throw "Not supported"; }
  virtual LoxObject BinaryDiv(LoxObject *lhs, LoxObject *rhs) { throw "Not supported"; }
  virtual LoxObject BinaryEQ(LoxObject *lhs, LoxObject *rhs) { throw "Not supported"; }
  virtual LoxObject BinaryNE(LoxObject *lhs, LoxObject *rhs) { throw "Not supported"; }
  virtual LoxObject BinaryLT(LoxObject *lhs, LoxObject *rhs) { throw "Not supported"; }
  virtual LoxObject BinaryGT(LoxObject *lhs, LoxObject *rhs) { throw "Not supported"; }
  virtual LoxObject BinaryLE(LoxObject *lhs, LoxObject *rhs) { throw "Not supported"; }
  virtual LoxObject BinaryGE(LoxObject *lhs, LoxObject *rhs) { throw "Not supported"; }
  virtual std::string ToString(const LoxObject *) { throw "Not supported"; }
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

  LoxObject UaryMinus(LoxObject *obj) override { return LoxObject(share_this(), -1 * obj->template AsNative<RealT>()); }
  LoxObject BinaryMinus(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(share_this(), lhs->template AsNative<RealT>() - rhs->template AsNative<RealT>());
  }
  LoxObject BinaryPlus(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(share_this(), lhs->template AsNative<RealT>() + rhs->template AsNative<RealT>());
  }
  LoxObject BinaryMul(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(share_this(), lhs->template AsNative<RealT>() * rhs->template AsNative<RealT>());
  }
  LoxObject BinaryDiv(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(share_this(), lhs->template AsNative<RealT>() / rhs->template AsNative<RealT>());
  }
  LoxObject BinaryEQ(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(lhs->template AsNative<RealT>() == rhs->template AsNative<RealT>());
  }
  LoxObject BinaryNE(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(lhs->template AsNative<RealT>() != rhs->template AsNative<RealT>());
  }
  LoxObject BinaryLT(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(lhs->template AsNative<RealT>() < rhs->template AsNative<RealT>());
  }
  LoxObject BinaryGT(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(lhs->template AsNative<RealT>() > rhs->template AsNative<RealT>());
  }
  LoxObject BinaryLE(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(lhs->template AsNative<RealT>() <= rhs->template AsNative<RealT>());
  }
  LoxObject BinaryGE(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(lhs->template AsNative<RealT>() >= rhs->template AsNative<RealT>());
  }
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
LoxObject Bool::UaryMinus(LoxObject *obj) {
  return LoxObject(new Number(), static_cast<double>(-1 * obj->AsNative<RealT>()));
}
template <>
std::string Bool::ToString(const LoxObject *obj) {
  return (obj->AsNative<RealT>() ? "true" : "false");
}

class String : public LoxObjectOperator {
  LoxObject BinaryPlus(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(share_this(), lhs->AsNative<std::string>() + rhs->AsNative<std::string>());
  }
  LoxObject BinaryEQ(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(lhs->template AsNative<std::string>() == rhs->template AsNative<std::string>());
  }
  LoxObject BinaryNE(LoxObject *lhs, LoxObject *rhs) override {
    return LoxObject(lhs->template AsNative<std::string>() != rhs->template AsNative<std::string>());
  }
  std::string ToString(const LoxObject *obj) override { return obj->AsNative<std::string>(); }
};

std::string LoxObject::ToString() { return lox_operator->ToString(this); }
LoxObject::operator bool() const { return lox_operator->IsTrue(this); }
LoxObject::LoxObject(bool v) : LoxObject(new Bool(), v) {}
LoxObject::LoxObject(double v) : LoxObject(new Number(), v) {}
LoxObject::LoxObject(const std::string &v) : LoxObject(new String(), v) {}
LoxObject LoxObject::operator-() { return lox_operator->UaryMinus(this); }
LoxObject LoxObject::operator!() { return LoxObject(!static_cast<bool>(*this)); }
LoxObject LoxObject::operator-(LoxObject &rhs) {
  BinaryTypeCheck(this, &rhs);
  return lox_operator->BinaryMinus(this, &rhs);
}
LoxObject LoxObject::operator+(LoxObject &rhs) {
  BinaryTypeCheck(this, &rhs);
  return lox_operator->BinaryPlus(this, &rhs);
}
LoxObject LoxObject::operator*(LoxObject &rhs) {
  BinaryTypeCheck(this, &rhs);
  return lox_operator->BinaryMul(this, &rhs);
}
LoxObject LoxObject::operator/(LoxObject &rhs) {
  BinaryTypeCheck(this, &rhs);
  return lox_operator->BinaryDiv(this, &rhs);
}
LoxObject LoxObject::operator==(LoxObject &rhs) {
  BinaryTypeCheck(this, &rhs);
  return lox_operator->BinaryEQ(this, &rhs);
}
LoxObject LoxObject::operator!=(LoxObject &rhs) {
  BinaryTypeCheck(this, &rhs);
  return lox_operator->BinaryNE(this, &rhs);
}
LoxObject LoxObject::operator<(LoxObject &rhs) {
  BinaryTypeCheck(this, &rhs);
  return lox_operator->BinaryLT(this, &rhs);
}
LoxObject LoxObject::operator>(LoxObject &rhs) {
  BinaryTypeCheck(this, &rhs);
  return lox_operator->BinaryGT(this, &rhs);
}
LoxObject LoxObject::operator<=(LoxObject &rhs) {
  BinaryTypeCheck(this, &rhs);
  return lox_operator->BinaryLE(this, &rhs);
}
LoxObject LoxObject::operator>=(LoxObject &rhs) {
  BinaryTypeCheck(this, &rhs);
  return lox_operator->BinaryGE(this, &rhs);
}
void LoxObject::BinaryTypeCheck(LoxObject *lhs, LoxObject *rhs) {
  if (typeid(*(lhs->lox_operator.get())) != typeid(*(rhs->lox_operator.get()))) {
    throw "Not same type";
  }
}
}  // namespace object
}  // namespace lox
