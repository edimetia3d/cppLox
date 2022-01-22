//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_OBJECT_OBJECT_H
#define LOX_SRCS_LOX_OBJECT_OBJECT_H

#include <set>
#include <string>
#include <vector>

namespace lox {
struct Object {
  Object();
  virtual ~Object();

  template <class T>
  T* DynAs() {
    return dynamic_cast<T*>(this);
  }

  template <class T>
  const T* DynAs() const {
    return dynamic_cast<const T*>(this);
  }

  template <class T>
  T* As() {
    return static_cast<T*>(this);
  }

  template <class T>
  const T* As() const {
    return static_cast<const T*>(this);
  }

  virtual bool Equal(const Object* rhs) const { return rhs == this; }
  [[nodiscard]] virtual std::string Str() const = 0;
  virtual std::vector<Object*> References() = 0;

  static std::set<Object*>& AllCreatedObj();
};
}  // namespace lox
#endif  // LOX_SRCS_LOX_OBJECT_OBJECT_H
