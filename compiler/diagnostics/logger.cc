#include "compiler/diagnostics/logger.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace seecpp::utility {

namespace {
// Google Style: Hide internal static state in an anonymous namespace 
// inside the implementation file rather than using static class members.
// Atomic: SetLevel may race with Log's filter check on another thread.
std::atomic<LogLevel> g_min_level{LogLevel::kInfo};
std::mutex g_log_mutex;
}  // namespace

void Logger::SetLevel(LogLevel min_level) {
    g_min_level.store(min_level, std::memory_order_relaxed);
}

LogLevel Logger::Level() {
    return g_min_level.load(std::memory_order_relaxed);
}

void Logger::Debug(std::string_view msg, const std::source_location loc) {
    Log(LogLevel::kDebug, msg, loc);
}

void Logger::Info(std::string_view msg, const std::source_location loc) {
    Log(LogLevel::kInfo, msg, loc);
}

void Logger::Warn(std::string_view msg, const std::source_location loc) {
    Log(LogLevel::kWarn, msg, loc);
}

void Logger::Error(std::string_view msg, const std::source_location loc) {
    Log(LogLevel::kError, msg, loc);
}

void Logger::Log(LogLevel level, std::string_view msg, const std::source_location& loc) {
    if (level < g_min_level.load(std::memory_order_relaxed)) return;

    // Timestamp — thread-safe via localtime_r / localtime_s
    const auto now = std::chrono::system_clock::now();
    const std::time_t time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};

#if defined(_WIN32)
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif

    const char* color;
    const char* label;
    std::ostream* out;

    switch (level) {
        case LogLevel::kDebug:
            color = "\033[36m"; label = "[DEBUG]"; out = &std::cout; break;
        case LogLevel::kInfo:
            color = "\033[32m"; label = "[INFO] "; out = &std::cout; break;
        case LogLevel::kWarn:
            color = "\033[33m"; label = "[WARN] "; out = &std::cerr; break;
        case LogLevel::kError:
            color = "\033[31m"; label = "[ERROR]"; out = &std::cerr; break;
        default:
            color = "\033[0m";  label = "[?????]"; out = &std::cerr; break;
    }

    // Mutex guard locks only the actual formatting and stream output
    std::lock_guard<std::mutex> lock(g_log_mutex);

    *out << color
         << label << ' '
         << std::put_time(&tm_buf, "%H:%M:%S") << ' '
         << loc.file_name() << ':' << loc.line()
         << " — " << msg
         << "\033[0m" << '\n';  // '\n' avoids per-line flush bottleneck

    if (level == LogLevel::kError) {
        out->flush();
    }
}

}  // namespace seecpp::utility
