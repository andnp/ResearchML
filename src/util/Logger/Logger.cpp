#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <map>
#include "util/Random/rand.hpp"
#include "util/cdash.hpp"
#include "util/json.hpp"
#include "util/Files/files.hpp"

#include "Logger.hpp"

namespace GPUCompute {
    std::string csvHeader(const json &js) {
        std::string out = "";
        for (const auto &j : json::iterator_wrapper(js)) {
            if (j.value().is_object()) {
            // This only goes one level deep. Need to replace with recursive function
            // in the future.
            for (const auto &j2 : json::iterator_wrapper(j.value())) {
                out += j2.key() + ", ";
            }
            } else {
            out += j.key() + ", ";
            }
        }
        return out;
    }

Logger &Logger::instance() {
  static Logger l;
  return l;
    }

    Logger& Logger::out() {
        Logger::current_stream = 0;
        return Logger::instance();
    }
    Logger& Logger::warn() {
        Logger::current_stream = 2;
        return Logger::instance();
    }
    Logger& Logger::info() {
        Logger::current_stream = 1;
        return Logger::instance();
    }
    Logger& Logger::aux(std::string f) {
        Logger::current_stream = 3;
        Logger::aux_file = f;
        return Logger::instance();
    }

    void Logger::registerFile(std::string file, std::ofstream* outfile) {
        if (Logger::open_files.size() < Logger::MAX_OPEN_FILES) {
            Logger::open_files[file] = outfile;
        } else {
            std::map<std::string, std::ofstream*>::iterator item = Logger::open_files.begin();
            std::advance(item, Random::uniform(0, Logger::open_files.size()));
            Logger::open_files[item->first]->close();
            Logger::open_files.erase(item);
            Logger::open_files[file] = outfile;
        }
    }

    void message(Output dest, std::string str, std::string fname) {
        switch (dest) {
            case Output::stdout : std::cout << str; break;
            case Output::none : break;
            case Output::file :
                std::ofstream *outfile;
                if (Logger::open_files.find(fname) == Logger::open_files.end()) {
                // not found
                outfile = new std::ofstream(fname, std::ios::app);
                Logger::registerFile(fname, outfile);
                } else {
                // found
                outfile = Logger::open_files[fname];
                }
                *outfile << str;
        }
    }

    void Logger::info(std::string str) {
        message(info_type, str, file + "info.txt");
    }
    void Logger::log(std::string str) {
        message(log_type, str, file + "log.txt");
    }
    void Logger::warn(std::string str) {
        message(warn_type, str, file + "warn.txt");
    }
    void Logger::aux(std::string f, std::string str) {
        message(aux_type, str, f);
    }
    void Logger::setWarn(Output dest) {
        if (dest == Output::file)
            createFile(file);
        warn_type = dest;
    }
    void Logger::setLog(Output dest) {
        if (dest == Output::file)
            createFile(file);
        log_type = dest;
    }
    void Logger::setInfo(Output dest) {
        if (dest == Output::file)
            createFile(file);
        info_type = dest;
    }
    void Logger::setAux(Output dest) {
        if (dest == Output::file)
            createFile(file);
        aux_type = dest;
    }
    void Logger::setFilepath(std::string str) {
        file = str;
    }
    void Logger::addFilepath(std::string str) {
        file += str;
    }
    // int Logger::currentStream() {
    //     return Logger::current_stream;
    // }

    std::string Logger::getFilepath() {
        if (file == "") {
            return "";
        }
        std::vector<std::string> x = _::split(file, '/');
        std::string f = "";
        int offset = 1;
        if (file.back() == '/') offset = 0;
        for (int i = 0; i < x.size() - offset; ++i) {
            f += x[i] + "/";
        }
        return f;
    }

    Logger& operator<<(Logger& l, std::ostream& (*os)(std::ostream&)) {
        Output out_type;
        if (Logger::current_stream == 0) {
            out_type = Logger::log_type;
        } else if (Logger::current_stream == 1) {
            out_type = Logger::info_type;
        } else if (Logger::current_stream == 2) {
            out_type = Logger::warn_type;
        } else {
            out_type = Logger::aux_type;
        }
        switch (out_type) {
            case Output::stdout : std::cout << std::endl; break;
            case Output::none : break;
            case Output::file :
                std::ofstream *outfile;
                std::string fname;
                if (Logger::current_stream == 0) {
                    fname = Logger::file + "log.txt";
                } else if (Logger::current_stream == 1) {
                    fname = Logger::file + "info.txt";
                } else if (Logger::current_stream == 2) {
                    fname = Logger::file + "warn.txt";
                } else {
                    fname = Logger::getFilepath() + Logger::aux_file;
                }
                if (Logger::open_files.find(fname) == Logger::open_files.end()) {
                // not found
                outfile = new std::ofstream(fname, std::ios::app);
                Logger::registerFile(fname, outfile);
                } else {
                // found
                outfile = Logger::open_files[fname];
                }
                *outfile << std::endl;
                break;
        }
        return l;
    }

    Logger::Logger() {}
    Output Logger::log_type = Output::stdout;
    Output Logger::warn_type = Output::stdout;
    Output Logger::info_type = Output::none;
    Output Logger::aux_type = Output::none;
    int Logger::current_stream = 0;
    std::string Logger::aux_file = "";
    std::string Logger::file = "";
    int Logger::MAX_OPEN_FILES = 25;
    std::map<std::string, std::ofstream*> Logger::open_files = {};
}

