//
// License: MIT
//

#ifndef CPPLOX_VERSION_VERSION_H_
#define CPPLOX_VERSION_VERSION_H_
#include <string>
namespace lox {
namespace version {
std::string GitSha1();
std::string GitDate();
} // namespace version
} // namespace lox
#endif // CPPLOX_VERSION_VERSION_H_
