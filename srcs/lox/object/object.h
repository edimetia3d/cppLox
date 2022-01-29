//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_OBJECT_OBJECT_H
#define LOX_SRCS_LOX_OBJECT_OBJECT_H

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace lox {
struct Object {
  Object();

  /**
   * Make a object managed by the sweeper
   */
  template <class T, class... Args>
  static T* Make(Args&&... args) {
    return new T(std::forward<Args>(args)...);
  }

  /**
   * Make a object managed by shared_ptr, most gc will be handled by reference count, partial will be handled by
   * the sweeper.
   */
  template <class T, class... Args>
  static std::shared_ptr<T> MakeShared(Args&&... args) {
    return new T(std::forward<Args>(args)...);
  }

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

  /**
   * Get the objects that this object referenced directly , so that these object will not be garbage collected.
   * @return
   */
  virtual std::vector<Object*> References() = 0;

  static std::set<Object*>& AllCreatedObj();

 private:
  void* operator new(size_t size);
};
}  // namespace lox
#endif  // LOX_SRCS_LOX_OBJECT_OBJECT_H
