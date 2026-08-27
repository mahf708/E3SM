#include <iostream>
#include <string_view>

#include <dexpr/supported_functions.hpp>

namespace {

void print_functions() {
    // The builtin seed only. A component registers its own vocabulary on top of
    // this at init, so what a given model accepts is a superset of this list.
    std::cout << "Builtin functions\n\n";

    for (const auto& entry : dexpr::builtin_functions()) {
        std::cout << "  " << entry.second << '\n';
    }
}

void print_help() {
    std::cout <<
R"(Usage:
    dexpr functions
    dexpr help
)";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc == 1) {
        print_help();
        return 0;
    }

    std::string_view command{argv[1]};

    if (command == "functions") {
        print_functions();
        return 0;
    }

    if (command == "help") {
        print_help();
        return 0;
    }

    std::cerr << "Unknown command: " << command << '\n';
    return 1;
}
