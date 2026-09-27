// Minimal JSON reader for stage graphs (objects, arrays, strings, numbers, bools, null).
#pragma once

#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rjllm {

class Json {
public:
    enum Kind { Null, Bool, Number, String, Array, Object };

    Json() : kind_(Null) {}

    static Json parse(const std::string &text) {
        size_t at = 0;
        Json value = parseValue(text, at);
        skip(text, at);
        if (at != text.size()) fail("trailing characters", at);
        return value;
    }

    Kind kind() const { return kind_; }
    bool isNull() const { return kind_ == Null; }
    bool isObject() const { return kind_ == Object; }

    bool has(const std::string &key) const {
        return kind_ == Object && object_.count(key) && !object_.at(key).isNull();
    }

    const Json &operator[](const std::string &key) const {
        if (kind_ != Object) throw std::runtime_error("JSON: '" + key + "' looked up in a non-object");
        auto it = object_.find(key);
        if (it == object_.end()) throw std::runtime_error("JSON: missing key '" + key + "'");
        return it->second;
    }

    const Json &operator[](size_t index) const {
        if (kind_ != Array || index >= array_.size()) throw std::runtime_error("JSON: bad array index");
        return array_[index];
    }

    size_t size() const { return kind_ == Array ? array_.size() : object_.size(); }
    const std::map<std::string, Json> &items() const { return object_; }

    double number() const {
        if (kind_ != Number) throw std::runtime_error("JSON: expected a number");
        return number_;
    }
    long long integer() const { return (long long)number(); }
    bool boolean() const {
        if (kind_ != Bool) throw std::runtime_error("JSON: expected a boolean");
        return bool_;
    }
    const std::string &string() const {
        if (kind_ != String) throw std::runtime_error("JSON: expected a string");
        return string_;
    }

    double get(const std::string &key, double fallback) const { return has(key) ? (*this)[key].number() : fallback; }
    bool flag(const std::string &key, bool fallback) const { return has(key) ? (*this)[key].boolean() : fallback; }
    std::string text(const std::string &key, const std::string &fallback) const {
        return has(key) ? (*this)[key].string() : fallback;
    }

private:
    Kind kind_;
    bool bool_ = false;
    double number_ = 0;
    std::string string_;
    std::vector<Json> array_;
    std::map<std::string, Json> object_;

    [[noreturn]] static void fail(const char *what, size_t at) {
        throw std::runtime_error(std::string("JSON parse error: ") + what + " at offset " + std::to_string(at));
    }

    static void skip(const std::string &s, size_t &at) {
        while (at < s.size() && (s[at] == ' ' || s[at] == '\n' || s[at] == '\r' || s[at] == '\t')) at++;
    }

    static Json parseValue(const std::string &s, size_t &at) {
        skip(s, at);
        if (at >= s.size()) fail("unexpected end", at);
        char c = s[at];
        Json v;
        if (c == '{') {
            v.kind_ = Object;
            at++;
            skip(s, at);
            if (at < s.size() && s[at] == '}') { at++; return v; }
            while (true) {
                skip(s, at);
                if (at >= s.size() || s[at] != '"') fail("expected key", at);
                std::string key = parseString(s, at);
                skip(s, at);
                if (at >= s.size() || s[at] != ':') fail("expected ':'", at);
                at++;
                v.object_[key] = parseValue(s, at);
                skip(s, at);
                if (at < s.size() && s[at] == ',') { at++; continue; }
                if (at < s.size() && s[at] == '}') { at++; return v; }
                fail("expected ',' or '}'", at);
            }
        }
        if (c == '[') {
            v.kind_ = Array;
            at++;
            skip(s, at);
            if (at < s.size() && s[at] == ']') { at++; return v; }
            while (true) {
                v.array_.push_back(parseValue(s, at));
                skip(s, at);
                if (at < s.size() && s[at] == ',') { at++; continue; }
                if (at < s.size() && s[at] == ']') { at++; return v; }
                fail("expected ',' or ']'", at);
            }
        }
        if (c == '"') {
            v.kind_ = String;
            v.string_ = parseString(s, at);
            return v;
        }
        if (s.compare(at, 4, "true") == 0) { v.kind_ = Bool; v.bool_ = true; at += 4; return v; }
        if (s.compare(at, 5, "false") == 0) { v.kind_ = Bool; at += 5; return v; }
        if (s.compare(at, 4, "null") == 0) { at += 4; return v; }
        const char *start = s.c_str() + at;
        char *end = nullptr;
        v.number_ = std::strtod(start, &end);
        if (end == start) fail("unexpected character", at);
        v.kind_ = Number;
        at += (size_t)(end - start);
        return v;
    }

    static void appendUtf8(std::string &out, unsigned cp) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 63)); }
        else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 63)); out += (char)(0x80 | (cp & 63));
        } else {
            out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 63));
            out += (char)(0x80 | ((cp >> 6) & 63)); out += (char)(0x80 | (cp & 63));
        }
    }

    static unsigned hex4(const std::string &s, size_t at) {
        if (at + 4 > s.size()) fail("short \\u escape", at);
        return (unsigned)std::stoul(s.substr(at, 4), nullptr, 16);
    }

    static std::string parseString(const std::string &s, size_t &at) {
        std::string out;
        at++;  // opening quote
        while (at < s.size()) {
            char c = s[at++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (at >= s.size()) break;
            char e = s[at++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4(s, at);
                    at += 4;
                    if (cp >= 0xD800 && cp < 0xDC00 && at + 6 <= s.size() && s[at] == '\\' && s[at + 1] == 'u') {
                        unsigned low = hex4(s, at + 2);
                        at += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default: fail("bad escape", at);
            }
        }
        fail("unterminated string", at);
    }
};

}  // namespace rjllm
