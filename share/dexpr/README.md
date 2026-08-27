# dexpr

A lexer and parser for the diagnostics expression language: the small DSL used
to describe derived diagnostics as expressions over model fields, rather than
as hand-written code per diagnostic.

Given a string like

```text
x*y.derivative(dx=dy,['col']).where(x>0)
```

which will be grouped as

```text
(x*y.derivative((dx=dy), ['col']).where((x>0)))
```

`dexpr` produces an abstract syntax tree. It does not evaluate anything, and it
knows nothing about fields, grids or timesteps -- turning an AST into an actual
diagnostic is the caller's job. the Parser::parse function returns a pointer to the
root node of the AST.

Example Usage:

```c++
parser::Parser parser{Lexer{string_input}};
const auto expr = parser.parse(); // expr is a std::unique_ptr
auto string_representation = ast::to_string(*expr); //to_string implmented as a Vistor
```

## Layout

| Path | Contents |
| --- | --- |
| `include/dexpr/` | Public headers |
| `src/` | Lexer, parser, AST printing, token and precedence tables |
| `tests/` | Catch2 unit tests |
| `tools/` | `dexpr`, a small command line helper |

The grammar is handled by a hand-written lexer feeding a Pratt (top-down operator precedence) parser.
AST nodes are represented by a `std::variant`, visited through
`Expression::visit`, which serves as a generic wrapper around `std::visit`.
This design means the nodes should be stable and transformations acting on the AST can be easily added.

Callable functions are not baked in. `supported_functions.hpp` defines a
`FunctionRegistry` a component fills in at init; see
[Component-supplied functions](#component-supplied-functions) below. Nothing
else in the library is diagnostics-specific.

### Parser terminology

The expression parser uses a hand-written
[Pratt parser](https://tdop.github.io/),
also known as *top-down operator-precedence parsing*. Pratt parsing associates
parsing behavior with tokens and uses operator precedence to determine how an
expression is grouped.

Some terminology used throughout the implementation:

- **Prefix expression** — an expression beginning with an operator
  and consumes an expression to its right, e.g. `-x` or `!x`.
  Literals and identifers are considered Prefix expressions for parser-function dispatch
  but are stored in the AST as a more specific node.

- **Infix expression** — an expression that contains operator located **in**-between a left- and right-hand
  expression, e.g. `x + y` or `x < y`.

- **Precedence / binding power** — determines how tightly an operator binds
  relative to surrounding operators. For example, multiplication has higher
  precedence than addition, so `x + y * z` is parsed as `x + (y * z)`.
  Exponentiation is right-associative: `x ** y ** z` is parsed as
  `x ** (y ** z)`. This is represented by conditionally modifying its Precedence to
  reflect right-binding power.

- **Prefix parse function** — parses an expression beginning with a token that can begin an expression,
  such as an identifier, literal, unary operator, or opening parenthesis.

- **Infix parse function** — extends an expression based on the `Precedence` of the following token.
  It receives the expression to its left and parses the required expression(s) to its right.
  Example tokens include: aritmetic operators, logical operators, opening parenthesis, etc...

## Component-supplied functions

dexpr owns the grammar; a component owns the vocabulary. The parser accepts any
call syntactically -- `foo(a, b=c)` parses the same whether or not `foo` exists
-- and a component declares what it can actually evaluate by filling a
`FunctionRegistry` and running `validate_calls` over the AST:

```c++
dexpr::FunctionRegistry reg;
reg.add({.name = "mean",
         .desc = "average over a dimension",
         .min_positional = 1,
         .max_positional = 1,           // -1 for variadic
         .keywords = {{"weights", false}},
         .form = dexpr::CallForm::Method});   // Free, Method or Any

parser::Parser parser{Lexer{input}};
const auto expr = parser.parse();
dexpr::validate_calls(*expr, reg);      // throws ValidationError
```

`validate_calls` checks that each callee is a plain name, that the name is
registered, that positional arity fits, that every keyword argument names a
declared parameter and appears once, that required keywords are present, and
that the call form matches. It collects every problem in the expression and
throws once, so a user fixes them all in one pass rather than one build at a
time. Unknown names come back with the registered set listed.

Keeping this out of the parser is deliberate: the same expression can be
checked against different components' vocabularies, and adding a callable
requires no edit to this library.

`builtin_functions()` holds the four generic callables (`where`, `sum`,
`derivative`, `tend`) as an optional seed. Nothing consults it implicitly -- a
component may seed from it or start empty.

## Building

`dexpr` is deliberately standalone. It has no Kokkos, MPI, netCDF or EKAT
dependency, it is not part of the CIME or `csm_share` build, and it requires
only CMake 3.14 and a C++20 compiler:

```shell
./run_tests.sh
```

which is shorthand for

```shell
cmake -S . -B build -DDEXPR_ENABLE_TESTS=ON -DDEXPR_ENABLE_TOOL=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure --parallel
```

Catch2 v3 is used for the tests. An installed copy is used if there is one;
otherwise CMake fetches a pinned version.

### Compiler requirement

C++20, and in practice GCC 11 or newer: numeric literals are parsed and
printed with floating-point `<charconv>`, which is the binding constraint.

### Options

| Option | Default | Effect |
| --- | --- | --- |
| `DEXPR_ENABLE_TESTS` | `ON` | Build the unit tests |
| `DEXPR_ENABLE_TOOL` | `ON` | Build the `dexpr` command line tool |
| `DEXPR_WERROR` | `OFF` | Treat compiler warnings as errors; CI sets this |

## Testing

Every `TEST_CASE` is registered with ctest individually, so a failure names
itself. The same commands run in CI, across gcc and clang in both Debug and
Release, on every pull request touching this directory.

## Planned work

Not implemented here, recorded so it is not rediscovered:

- **Operator syntax is fixed.** A registry would let a component add functions
  but not operators. A new operator means editing the token enum, the lexer,
  the precedence table and the parser's dispatch tables.

## Provenance

Vendored from [peterdschwartz/e3sm_diags_parser](https://github.com/peterdschwartz/e3sm_diags_parser)
at `d680d50`, MIT licensed; see `LICENSE`. The upstream name, and its `edp`
abbreviation, were dropped in favour of `dexpr` once the code moved here.
