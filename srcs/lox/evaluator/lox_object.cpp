//
// LICENSE: MIT
//

#include "lox_object.h"

namespace lox {
namespace object {

class LoxObjectImpl {
 public:
  virtual ~LoxObjectImpl(){};

  virtual LoxObjectImplPointer operator-() = 0;
  virtual std::string ToString() = 0;
  explicit virtual operator bool() const { return this != nullptr; }
};

class Number : public LoxObjectImpl {
 public:
  explicit Number(double value) : value(value) {}

  LoxObjectImplPointer operator-() override { return LoxObjectImplPointer(new Number(-value)); }

  explicit operator bool() const override { return static_cast<bool>(value); }

  std::string ToString() override { return std::to_string(value); }

  double value;
};

class Bool : public LoxObjectImpl {
 public:
  explicit Bool(bool value) : value(value) {}

  LoxObjectImplPointer operator-() override { return LoxObjectImplPointer(new Number(-value)); }

  explicit operator bool() const override { return value; }

  std::string ToString() override { return std::to_string(value); }

  bool value;
};

class String : public LoxObjectImpl {
 public:
  explicit String(const std::string value) : value(value) {}
  LoxObjectImplPointer operator-() override { throw "Not supported"; }
  std::string ToString() override { return value; }
  std::string value;
};

LoxObject LoxObject::operator-() { return LoxObject(-(*impl)); }
std::string LoxObject::ToString() { return impl->ToString(); }
LoxObject::operator bool() const { return static_cast<bool>(*impl); }
LoxObject::LoxObject(bool v) { impl = LoxObjectImplPointer(new Bool(v)); }
LoxObject::LoxObject(double v) { impl = LoxObjectImplPointer(new Number(v)); }
LoxObject::LoxObject(const std::string& v) { impl = LoxObjectImplPointer(new String(v)); }
LoxObject LoxObject::operator!() { return LoxObject(static_cast<bool>(*this)); }

}  // namespace object
}  // namespace lox
