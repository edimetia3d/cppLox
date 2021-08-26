//
// LICENSE: MIT
//

#include "lox_object.h"

#include "lox/backend/tree_walker/lox_object/binary_dispatcher.h"

namespace lox {
namespace object {

LoxObject operator-(const LoxObject &self) { return self->operator-(); }
LoxObject Bool::operator-() const { return MakeLoxObject<Number>(-RawValue<RawValueT>()); }
LoxObject Number::operator-() const { return MakeLoxObject<Number>(-RawValue<RawValueT>()); }

LoxObject BinaryEQ(const Bool *lhs, const Bool *rhs) {
  return (MakeLoxObject<Bool>(lhs->RawValue<Bool::RawValueT>() == rhs->RawValue<Bool::RawValueT>()));
}

LoxObject BinaryNE(const Bool *lhs, const Bool *rhs) {
  return (MakeLoxObject<Bool>(lhs->RawValue<Bool::RawValueT>() != rhs->RawValue<Bool::RawValueT>()));
}

LoxObject BinaryPlus(const String *lhs, const String *rhs) {
  return (MakeLoxObject<String>(lhs->RawValue<String::RawValueT>() + rhs->RawValue<String::RawValueT>()));
}

LoxObject BinaryMul(const String *lhs, const Number *rhs) {
  int loop = rhs->RawValue<Number::RawValueT>();
  std::string ret = "";
  for (int i = 0; i < loop; ++i) {
    ret += lhs->RawValue<std::string>();
  }
  return (MakeLoxObject<String>(ret));
}
LoxObject BinaryMul(const Number *lhs, const String *rhs) { return BinaryMul(rhs, lhs); }

LoxObject BinaryEQ(const String *lhs, const String *rhs) {
  return (MakeLoxObject<Bool>(lhs->RawValue<String::RawValueT>() == rhs->RawValue<String::RawValueT>()));
}

LoxObject BinaryNE(const String *lhs, const String *rhs) {
  return (MakeLoxObject<Bool>(lhs->RawValue<String::RawValueT>() != rhs->RawValue<String::RawValueT>()));
}

LoxObject BinaryPlus(const Number *lhs, const Number *rhs) {
  return (MakeLoxObject<Number>(lhs->RawValue<Number::RawValueT>() + rhs->RawValue<Number::RawValueT>()));
}
LoxObject BinaryMinus(const Number *lhs, const Number *rhs) {
  return (MakeLoxObject<Number>(lhs->RawValue<Number::RawValueT>() - rhs->RawValue<Number::RawValueT>()));
}
LoxObject BinaryMul(const Number *lhs, const Number *rhs) {
  return (MakeLoxObject<Number>(lhs->RawValue<Number::RawValueT>() * rhs->RawValue<Number::RawValueT>()));
}
LoxObject BinaryDiv(const Number *lhs, const Number *rhs) {
  return (MakeLoxObject<Number>(lhs->RawValue<Number::RawValueT>() / rhs->RawValue<Number::RawValueT>()));
}

LoxObject BinaryEQ(const Number *lhs, const Number *rhs) {
  return (MakeLoxObject<Bool>(lhs->RawValue<Number::RawValueT>() == rhs->RawValue<Number::RawValueT>()));
}

LoxObject BinaryNE(const Number *lhs, const Number *rhs) {
  return (MakeLoxObject<Bool>(lhs->RawValue<Number::RawValueT>() != rhs->RawValue<Number::RawValueT>()));
}

LoxObject BinaryLT(const Number *lhs, const Number *rhs) {
  return (MakeLoxObject<Bool>(lhs->RawValue<Number::RawValueT>() < rhs->RawValue<Number::RawValueT>()));
}

LoxObject BinaryGT(const Number *lhs, const Number *rhs) {
  return (MakeLoxObject<Bool>(lhs->RawValue<Number::RawValueT>() > rhs->RawValue<Number::RawValueT>()));
}

LoxObject BinaryLE(const Number *lhs, const Number *rhs) {
  return (MakeLoxObject<Bool>(lhs->RawValue<Number::RawValueT>() <= rhs->RawValue<Number::RawValueT>()));
}

LoxObject BinaryGE(const Number *lhs, const Number *rhs) {
  return (MakeLoxObject<Bool>(lhs->RawValue<Number::RawValueT>() >= rhs->RawValue<Number::RawValueT>()));
}

#define QUICK_DEF_BINARY_OP(OPNAME, SYMBOL)                                                               \
  DEF_DISPATCHER(OPNAME)                                                                                  \
  LoxObject SYMBOL(const LoxObject &lhs, const LoxObject &rhs) {                                          \
    LoxObject ret = LoxObject();                                                                          \
    DISPATCHTER(OPNAME)<Number, Bool, String>::In<Number, Bool, String>::Run(lhs.get(), rhs.get(), &ret); \
    if (!IsValid(ret)) {                                                                                  \
      throw "No overload found";                                                                          \
    }                                                                                                     \
    return ret;                                                                                           \
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
