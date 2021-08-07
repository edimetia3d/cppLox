//
// LICENSE: MIT
//

#include "lox/lox_object/lox_object.h"

#include "lox/lox_object/binary_dispatcher.h"

namespace lox {
namespace object {

struct LoxObjectState {
  template <class RealT>
  LoxObjectState(const RealT &v) : raw_value(new RealT{v}) {}
  std::shared_ptr<void> raw_value;
  virtual LoxObjectStatePtr operator-() = 0;
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

struct Bool : public LoxObjectState {
  using RealT = bool;
  explicit Bool(const RealT &v) : LoxObjectState(v) {}
  LoxObjectStatePtr operator-() override;
  bool IsTrue() override { return AsNative<RealT>(); }
  std::string ToString() override { return (AsNative<RealT>() ? "true" : "false"); }
};
struct Number : public LoxObjectState {
  using RealT = double;
  explicit Number(const RealT &v) : LoxObjectState(v) {}
  bool IsTrue() override { return static_cast<bool>(AsNative<RealT>()); }
  std::string ToString() override { return std::to_string(AsNative<RealT>()); }
  LoxObjectStatePtr operator-() override { return LoxObjectStatePtr(new Number(-AsNative<RealT>())); }
};

LoxObjectStatePtr Bool::operator-() { return LoxObjectStatePtr(new Number(-AsNative<RealT>())); }

struct String : public LoxObjectState {
  using RealT = std::string;
  explicit String(const RealT &v) : LoxObjectState(v) {}
  std::string ToString() override { return std::string("\"") + AsNative<RealT>() + "\""; }
  LoxObjectStatePtr operator-() override { throw "`!` is not supported on String"; }
};

LoxObjectStatePtr BinaryEQ(Bool *lhs, Bool *rhs) {
  return LoxObjectStatePtr(new Bool(lhs->AsNative<Bool::RealT>() == rhs->AsNative<Bool::RealT>()));
}

LoxObjectStatePtr BinaryNE(Bool *lhs, Bool *rhs) {
  return LoxObjectStatePtr(new Bool(lhs->AsNative<Bool::RealT>() != rhs->AsNative<Bool::RealT>()));
}

LoxObjectStatePtr BinaryPlus(String *lhs, String *rhs) {
  return LoxObjectStatePtr(new String(lhs->AsNative<String::RealT>() + rhs->AsNative<String::RealT>()));
}

LoxObjectStatePtr BinaryMul(String *lhs, Number *rhs) {
  int loop = rhs->AsNative<Number::RealT>();
  std::string ret = "";
  for (int i = 0; i < loop; ++i) {
    ret += lhs->AsNative<std::string>();
  }
  return LoxObjectStatePtr(new String(ret));
}
LoxObjectStatePtr BinaryMul(Number *lhs, String *rhs) { return BinaryMul(rhs, lhs); }

LoxObjectStatePtr BinaryEQ(String *lhs, String *rhs) {
  return LoxObjectStatePtr(new Bool(lhs->AsNative<String::RealT>() == rhs->AsNative<String::RealT>()));
}

LoxObjectStatePtr BinaryNE(String *lhs, String *rhs) {
  return LoxObjectStatePtr(new Bool(lhs->AsNative<String::RealT>() != rhs->AsNative<String::RealT>()));
}

LoxObjectStatePtr BinaryPlus(Number *lhs, Number *rhs) {
  return LoxObjectStatePtr(new Number(lhs->AsNative<Number::RealT>() + rhs->AsNative<Number::RealT>()));
}
LoxObjectStatePtr BinaryMinus(Number *lhs, Number *rhs) {
  return LoxObjectStatePtr(new Number(lhs->AsNative<Number::RealT>() - rhs->AsNative<Number::RealT>()));
}
LoxObjectStatePtr BinaryMul(Number *lhs, Number *rhs) {
  return LoxObjectStatePtr(new Number(lhs->AsNative<Number::RealT>() * rhs->AsNative<Number::RealT>()));
}
LoxObjectStatePtr BinaryDiv(Number *lhs, Number *rhs) {
  return LoxObjectStatePtr(new Number(lhs->AsNative<Number::RealT>() / rhs->AsNative<Number::RealT>()));
}

LoxObjectStatePtr BinaryEQ(Number *lhs, Number *rhs) {
  return LoxObjectStatePtr(new Bool(lhs->AsNative<Number::RealT>() == rhs->AsNative<Number::RealT>()));
}

LoxObjectStatePtr BinaryNE(Number *lhs, Number *rhs) {
  return LoxObjectStatePtr(new Bool(lhs->AsNative<Number::RealT>() != rhs->AsNative<Number::RealT>()));
}

LoxObjectStatePtr BinaryLT(Number *lhs, Number *rhs) {
  return LoxObjectStatePtr(new Bool(lhs->AsNative<Number::RealT>() < rhs->AsNative<Number::RealT>()));
}

LoxObjectStatePtr BinaryGT(Number *lhs, Number *rhs) {
  return LoxObjectStatePtr(new Bool(lhs->AsNative<Number::RealT>() > rhs->AsNative<Number::RealT>()));
}

LoxObjectStatePtr BinaryLE(Number *lhs, Number *rhs) {
  return LoxObjectStatePtr(new Bool(lhs->AsNative<Number::RealT>() <= rhs->AsNative<Number::RealT>()));
}

LoxObjectStatePtr BinaryGE(Number *lhs, Number *rhs) {
  return LoxObjectStatePtr(new Bool(lhs->AsNative<Number::RealT>() >= rhs->AsNative<Number::RealT>()));
}

LoxObject::LoxObject(bool v) : lox_object_state_(new Bool(v)) {}
LoxObject::LoxObject(double v) : lox_object_state_(new Number(v)) {}
LoxObject::LoxObject(const std::string &v) : lox_object_state_(new String(v)) {}
LoxObject LoxObject::VoidObject() { return LoxObject(LoxObjectStatePtr(nullptr)); }
LoxObject LoxObject::operator-() { return -(*lox_object_state_); }
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
