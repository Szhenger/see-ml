#ifndef SEECPP_COMPILER_DIAGNOSTICS_LOGGER_H_
#define SEECPP_COMPILER_DIAGNOSTICS_LOGGER_H_

#include <cstdint>
#include <source_location>
#include <string_view>

namespace seecpp::utility {

// Google Style: Enumerators should be named like constants (kEnumName).
enum class LogLevel : uint8_t { 
    kDebug = 0, 
    kInfo = 1, 
    kWarn = 2, 
    kError = 3 
};

class Logger {
 public:
    // --- Configuration ---
    static void SetLevel(LogLevel min_level);
    static LogLevel Level();

    // --- Public API ---
    // Google Style: Function names use PascalCase.
    static void Debug(std::string_view msg,
                      const std::source_location loc = std::source_location::current());
    static void Info(std::string_view msg,
                     const std::source_location loc = std::source_location::current());
    static void Warn(std::string_view msg,
                     const std::source_location loc = std::source_location::current());
    static void Error(std::string_view msg,
                      const std::source_location loc = std::source_location::current());

 private:
    static void Log(LogLevel level, std::string_view msg, const std::source_location& loc);
};

}  // namespace seecpp::utility

#endif  // SEECPP_COMPILER_DIAGNOSTICS_LOGGER_H_
