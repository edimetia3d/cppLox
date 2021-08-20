//
// LICENSE: MIT
//

#include "lox/lox_object/lox_object.h"

#include "lox/lox_object/binary_dispatcher.h"

namespace lox {
namespace object {

struct Bool : public LoxObjectBase {
  using RealT = bool;
  explicit Bool(const RealT &v) : LoxObjectBase(v) {}
  LoxObject operator-() override;
  bool IsTrue() override { return AsNative<RealT>(); }
  std::string ToString() override { return (AsNative<RealT>() ? "true" : "false"); }
};
struct Number : public LoxObjectBase {
  using RealT = double;
  explicit Number(const RealT &v) : LoxObjectBase(v) {}
  bool IsTrue() override { return static_cast<bool>(AsNative<RealT>()); }
  std::string ToString() override { return std::to_string(AsNative<RealT>()); }
  LoxObject operator-() override { return LoxObject(new Number(-AsNative<RealT>())); }
};

LoxObject Bool::operator-() { return LoxObject(new Number(-AsNative<RealT>())); }

struct String : public LoxObjectBase {
  using RealT = std::string;
  explicit String(const RealT &v) : LoxObjectBase(v) {}
  std::string ToString() override { return std::string("\"") + AsNative<RealT>() + "\""; }
  LoxObject operator-() override { throw "`!` is not supported on String"; }
};

LoxObject BinaryEQ(Bool *lhs, Bool *rhs) {
  return LoxObject(new Bool(lhs->AsNative<Bool::RealT>() == rhs->AsNative<Bool::RealT>()));
}

LoxObject BinaryNE(Bool *lhs, Bool *rhs) {
  return LoxObject(new Bool(lhs->AsNative<Bool::RealT>() != rhs->AsNative<Bool::RealT>()));
}

LoxObject BinaryPlus(String *lhs, String *rhs) {
  return LoxObject(new String(lhs->AsNative<String::RealT>() + rhs->AsNative<String::RealT>()));
}

LoxObject BinaryMul(String *lhs, Number *rhs) {
  int loop = rhs->AsNative<Number::RealT>();
  std::string ret = "";
  for (int i = 0; i < loop; ++i) {
    ret += lhs->AsNative<std::string>();
  }
  return LoxObject(new String(ret));
}
LoxObject BinaryMul(Number *lhs, String *rhs) { return BinaryMul(rhs, lhs); }

LoxObject BinaryEQ(String *lhs, String *rhs) {
  return LoxObject(new Bool(lhs->AsNative<String::RealT>() == rhs->AsNative<String::RealT>()));
}

LoxObject BinaryNE(String *lhs, String *rhs) {
  return LoxObject(new Bool(lhs->AsNative<String::RealT>() != rhs->AsNative<String::RealT>()));
}

LoxObject BinaryPlus(Number *lhs, Number *rhs) {
  return LoxObject(new Number(lhs->AsNative<Number::RealT>() + rhs->AsNative<Number::RealT>()));
}
LoxObject BinaryMinus(Number *lhs, Number *rhs) {
  return LoxObject(new Number(lhs->AsNative<Number::RealT>() - rhs->AsNative<Number::RealT>()));
}
LoxObject BinaryMul(Number *lhs, Number *rhs) {
  return LoxObject(new Number(lhs->AsNative<Number::RealT>() * rhs->AsNative<Number::RealT>()));
}
LoxObject BinaryDiv(Number *lhs, Number *rhs) {
  return LoxObject(new Number(lhs->AsNative<Number::RealT>() / rhs->AsNative<Number::RealT>()));
}

LoxObject BinaryEQ(Number *lhs, Number *rhs) {
  return LoxObject(new Bool(lhs->AsNative<Number::RealT>() == rhs->AsNative<Number::RealT>()));
}

LoxObject BinaryNE(Number *lhs, Number *rhs) {
  return LoxObject(new Bool(lhs->AsNative<Number::RealT>() != rhs->AsNative<Number::RealT>()));
}

LoxObject BinaryLT(Number *lhs, Number *rhs) {
  return LoxObject(new Bool(lhs->AsNative<Number::RealT>() < rhs->AsNative<Number::RealT>()));
}

LoxObject BinaryGT(Number *lhs, Number *rhs) {
  return LoxObject(new Bool(lhs->AsNative<Number::RealT>() > rhs->AsNative<Number::RealT>()));
}

LoxObject BinaryLE(Number *lhs, Number *rhs) {
  return LoxObject(new Bool(lhs->AsNative<Number::RealT>() <= rhs->AsNative<Number::RealT>()));
}

LoxObject BinaryGE(Number *lhs, Number *rhs) {
  return LoxObject(new Bool(lhs->AsNative<Number::RealT>() >= rhs->AsNative<Number::RealT>()));
}

LoxObject::LoxObject(bool v) : lox_object_state_(new Bool(v)) {}
LoxObject::LoxObject(double v) : lox_object_state_(new Number(v)) {}
LoxObject::LoxObject(const std::string &v) : lox_object_state_(new String(v)) {}
LoxObject LoxObject::VoidObject() { return LoxObject(LoxObject(nullptr)); }
LoxObject LoxObject::operator-() { return LoxObject(-(*lox_object_state_)); }
std::string LoxObject::ToString() { return lox_object_state_->ToString(); }
bool LoxObject::IsValueTrue() {
  if (!IsValid()) {
    throw "Not a valid LoxObject";
  }
  return lox_object_state_->IsTrue();
};
void *LoxObject::RawObjPtr() { return lox_object_state_->raw_value.get(); }
void *LoxObject::RawObjPtr() const { return (const_cast<LoxObject *>(this))->RawObjPtr(); }

#define QUICK_DEF_BINARY_OP(OPNAME, SYMBOL)                                                                      \
  DEF_DISPATCHER(OPNAME)                                                                                         \
  LoxObject LoxObject::SYMBOL(LoxObject &rhs) {                                                                  \
    LoxObject ret = LoxObject();                                                                                 \
    DISPATCHTER(OPNAME)<Number, Bool, String>::In<Number, Bool, String>::Run(lox_object_state_.get(),            \
                                                                             rhs.lox_object_state_.get(), &ret); \
    if (!ret.lox_object_state_) {                                                                                \
      throw "No overload found";                                                                                 \
    }                                                                                                            \
    return ret;                                                                                                  \
  }
QUICK_DEF_BINARY_OP(BinaryMinus, operator-)
QUICK_DEF_BINARY_OP(BinaryPlus, operator+)
QUICK_DEF_BINARY_OP(BinaryMul, operator*)
QUICK_DEF_BINARY_OP(BinaryDiv, operator/)
QUICK_DEF_BINARY_OP(BinaryEQ, operator==)
QUICK_DEF_BINARY_OP(BinaryNE, operator!=)
QUICK_DEF_BINARY_OP(BinaryLT, operator<)
QUICK_DEF_BINARY_OP(BinaryGT, operator>)
QUICK_DEF_BINARY_OP(BinaryLE, operator<=)
QUICK_DEF_BINARY_OP(BinaryGE, operator>=)

}  // namespace object
}  // namespace lox
