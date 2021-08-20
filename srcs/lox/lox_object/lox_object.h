//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
#include <map>
#include <memory>
#include <string>
#include <type_traits>

#include "lox/token.h"
namespace lox {

namespace object {

class LoxObjectBase;
template <class T>
concept SubclassOfLoxObject = std::is_base_of<LoxObjectBase, T>::value;

#define LOX_OBJECT_CTOR_SHARED_PTR_ONLY(CLASS_NAME) \
 protected:                                         \
  CLASS_NAME() = default;                           \
  friend class LoxObjectBase

using LoxObject = std::shared_ptr<LoxObjectBase>;
static inline LoxObject VoidObject() { return LoxObject(nullptr); }

class LoxObjectBase : public std::enable_shared_from_this<LoxObjectBase> {
 public:
  template <SubclassOfLoxObject SubT>
  static std::shared_ptr<SubT> Make(const typename SubT::RawValueT& v) {
    static_assert(sizeof(SubT) == sizeof(LoxObjectBase));  // Only Base are allowed to hold data
    SubT* obj = new SubT();
    LoxObjectSet(obj, v);
    auto ret = std::shared_ptr<SubT>(obj);
    ret->Init();  // subclass could init extra data here
    return ret;
  }

  virtual LoxObject operator-() const { throw "Not supported"; }
  virtual void Init() const {};
  virtual bool IsTrue() const { return raw_value.get(); };
  virtual std::string ToString() const {
    return std::string("LoxObjectBase at ") + std::to_string((uint64_t)raw_value.get());
  };

  virtual LoxObject GetAttr(const std::string& name) {
    if (dict.contains(name)) {
      return dict[name];
    }
    return VoidObject();
  }

  virtual void SetAttr(std::string name, LoxObject obj) { dict[name] = obj; }

  template <class T>
  T& AsNative() {
    return *static_cast<T*>(raw_value.get());
  }

  template <class T>
  const T& AsNative() const {
    return *static_cast<T*>(raw_value.get());
  }

  template <SubclassOfLoxObject T>
  T* DownCast() {
    return dynamic_cast<T*>(this);
  }

  virtual ~LoxObjectBase() = default;

 protected:
  std::shared_ptr<void> raw_value;
  std::map<std::string, object::LoxObject> dict;
  template <class RawValueT>
  static void TypeDeleter(RawValueT* p) {
    delete p;
  }

 private:
  template <class RawValueT>
  static void LoxObjectSet(LoxObjectBase* target, const RawValueT& v) {
    target->raw_value.reset(new RawValueT{v}, TypeDeleter<RawValueT>);
  }

 protected:
  LoxObjectBase() = default;
};

class Bool : public LoxObjectBase {
 public:
  using RawValueT = bool;
  LoxObject operator-() const override;
  bool IsTrue() const override { return AsNative<RawValueT>(); }
  std::string ToString() const override { return (AsNative<RawValueT>() ? "true" : "false"); }
  LOX_OBJECT_CTOR_SHARED_PTR_ONLY(Bool);
};
class Number : public LoxObjectBase {
 public:
  using RawValueT = double;
  bool IsTrue() const override { return static_cast<bool>(AsNative<RawValueT>()); }
  std::string ToString() const override { return std::to_string(AsNative<RawValueT>()); }
  LoxObject operator-() const override;
  LOX_OBJECT_CTOR_SHARED_PTR_ONLY(Number);
};

class String : public LoxObjectBase {
 public:
  using RawValueT = std::string;
  std::string ToString() const override { return std::string("\"") + AsNative<RawValueT>() + "\""; }
  LoxObject operator-() const override { throw "`!` is not supported on String"; }
  LOX_OBJECT_CTOR_SHARED_PTR_ONLY(String);
};

static inline bool IsValid(const LoxObject& obj) { return obj.get(); }

static inline bool IsValueTrue(const LoxObject& obj) {
  if (!IsValid(obj)) {
    throw "Not a valid LoxObject";
  }
  return obj->IsTrue();
};

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

template <SubclassOfLoxObject SubT>
static inline std::shared_ptr<SubT> MakeLoxObject(typename SubT::RawValueT v) {
  return LoxObjectBase::Make<SubT>(v);
}

static inline LoxObject MakeLoxObject(bool v) { return LoxObjectBase::Make<Bool>(v); }

static inline LoxObject MakeLoxObject(const std::string& v) { return LoxObjectBase::Make<String>(v); }

static inline LoxObject MakeLoxObject(double v) { return LoxObjectBase::Make<Number>(v); }

}  // namespace object
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_LOX_OBJECT_H_
