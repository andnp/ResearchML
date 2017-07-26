#pragma once
#include <string>
#include <iostream>
#include <map>

namespace GPUCompute {
    enum class Output {stdout, file, none};

    std::string csvHeader(const json &js);

    class Logger {
    public:
        static Logger& instance();
        static void info(std::string);
        static void log(std::string);
        static void warn(std::string);
        static void aux(std::string file, std::string message);
        static void setWarn(Output);
        static void setLog(Output);
        static void setInfo(Output);
        static void setAux(Output);
        static void setFilepath(std::string);
        static void addFilepath(std::string);
        static std::string getFilepath();
        static Logger& out();
        static Logger& warn();
        static Logger& info();
        static Logger& aux(std::string file);
        template <typename T>
        friend Logger& operator<<(Logger& os, const T val);
        friend Logger& operator<<(Logger&, std::ostream& (*os)(std::ostream&));
        friend void message(Output dest, std::string str, std::string file);

    private:
        static void registerFile(std::string file, std::ofstream* outfile);
        static std::string file;
        static Output warn_type;
        static Output log_type;
        static Output info_type;
        static Output aux_type;
        static int current_stream;
        static std::string aux_file;
        static std::map<std::string, std::ofstream*> open_files;
        static int MAX_OPEN_FILES;
        Logger();
    };

    template <typename T>
    Logger& operator<<(Logger& os, T val) {
        std::stringstream stream;
        stream << val;
        std::string s = stream.str();
        if (Logger::current_stream == 0) {
            Logger::log(s);
        } else if (Logger::current_stream == 1) {
            Logger::info(s);
        } else if(Logger::current_stream == 2) {
            Logger::warn(s);
        } else {
            Logger::aux(Logger::getFilepath() + Logger::aux_file, s);
        }
        return os;
    }

    Logger& operator<<(Logger& l, std::ostream& (*os)(std::ostream&));
}
