/**
 * @file supported_functions.hpp
 * @brief The set of callables a component makes available to expressions.
 *
 * dexpr owns the grammar; a component owns the vocabulary. The parser accepts
 * any call syntactically -- `foo(a, b=c)` parses the same whether or not `foo`
 * exists -- and a component declares what it can actually evaluate by filling a
 * FunctionRegistry at init and running validate_calls() over the AST.
 *
 * That split is deliberate: parsing stays independent of who is asking, so the
 * same expression can be checked against different components' vocabularies,
 * and a component adds a callable without editing anything in this library.
 *
 * Every spec carries an `example` of how the call is written, and
 * validate_registry() proves each example parses, validates against the very
 * spec that declared it, and really calls that function. So a vocabulary is
 * checked against itself: get the arity or the keywords wrong and the function
 * you just added fails immediately, rather than the first time a user writes it.
 */
#ifndef DEXPR_SUPPORTED_FUNCTIONS_HPP
#define DEXPR_SUPPORTED_FUNCTIONS_HPP

#include <dexpr/ast.hpp>

#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dexpr {

// A keyword argument a function accepts, e.g. the `dims` of
// `derivative(dx, dims=['col'])`.
struct ParamSpec {
  std::string name;
  bool required = false;
};

struct FunctionSpec {
  std::string name;
  std::string desc;

  // Positional arity, not counting keyword arguments. For a method call `x.f()`
  // the receiver is not positional, so `x.mean('lev')` has one.
  int min_positional = 0;
  int max_positional = 0;

  std::vector<ParamSpec> keywords;

  // One way the call is actually written, e.g. "T_mid.mean('lev')". Doubles as
  // the documentation and as what validate_registry() checks the spec against,
  // so the two cannot drift apart.
  std::string example;

  // "name(arg, kw=..)\n--- desc\n--- e.g. example", the `dexpr` tool listing.
  std::string to_string() const;
};

inline std::ostream& operator<<(std::ostream& os, const FunctionSpec& function) {
  return os << function.to_string() << '\n';
}

// The callables one component makes available. Ordered, so listings and error
// messages are stable rather than hash-order.
class FunctionRegistry {
public:
  // Throws std::invalid_argument if the spec is malformed (no name, impossible
  // arity, a nameless or repeated keyword) or if the name is already
  // registered: two specs for one name means one of them is silently dead.
  void add(FunctionSpec spec);

  const FunctionSpec* find(std::string_view name) const;
  bool contains(std::string_view name) const { return find(name) != nullptr; }

  std::vector<std::string> names() const;

  auto begin() const { return fns_.begin(); }
  auto end() const { return fns_.end(); }

private:
  // std::less<> so lookups take a string_view without allocating.
  std::map<std::string, FunctionSpec, std::less<>> fns_;
};

// The generic callables that used to be hard-coded here. A component may seed
// its registry from this or start empty; nothing consults it implicitly.
const FunctionRegistry& builtin_functions();

class ValidationError : public std::runtime_error {
public:
  explicit ValidationError(const std::vector<std::string>& errors);
};

// Checks every call in the tree against `reg`: that the callee is a plain name,
// that the name is registered, that positional arity fits, that each keyword
// argument names a declared parameter, and that required keywords are present.
// Collects every problem and throws once.
//
// Kept out of the parser on purpose -- see the file comment.
void validate_calls(const ast::Expression& root, const FunctionRegistry& reg);

// Checks a whole vocabulary against itself: every spec must carry an example,
// and that example must parse, pass validate_calls() against `reg`, and call
// the function that declared it. Collects every problem and throws once.
//
// This is the check to run after registering a new function -- from a unit
// test, or straight from the code that builds the registry.
void validate_registry(const FunctionRegistry& reg);

} // namespace dexpr

#endif
