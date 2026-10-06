/* Readers for the specification's conformance files: a parser for the JSON they use, which has
 * no escapes in strings, and SHA-256 for their stream hashes. */
#pragma once

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    std::string text; /* a string's characters or a number's literal */
    bool flag = false;
    std::vector<Json> items;        /* an array's elements or an object's values */
    std::vector<std::string> names; /* an object's names, one per value */

    const Json *find(const char *name) const {
        for (size_t i = 0; i < names.size(); i++)
            if (names[i] == name)
                return &items[i];
        return nullptr;
    }
    const Json &operator[](const char *name) const {
        if (const Json *v = find(name))
            return *v;
        throw std::runtime_error(std::string("conformance file: no field ") + name);
    }
    uint64_t u64() const { return std::strtoull(text.c_str(), nullptr, 10); }
    double number() const { return std::strtod(text.c_str(), nullptr); }
    uint64_t hex() const { return std::strtoull(text.c_str(), nullptr, 16); }
};

class JsonParser {
  public:
    explicit JsonParser(const std::string &s) : p(s.data()), end(s.data() + s.size()) {}

    Json value() {
        Json v;
        switch (peek()) {
        case '{':
            p++;
            v.type = Json::Object;
            if (peek() == '}')
                return p++, v;
            do {
                v.names.push_back(string());
                expect(':');
                v.items.push_back(value());
            } while (accept(','));
            expect('}');
            return v;
        case '[':
            p++;
            v.type = Json::Array;
            if (peek() == ']')
                return p++, v;
            do
                v.items.push_back(value());
            while (accept(','));
            expect(']');
            return v;
        case '"':
            v.type = Json::String;
            v.text = string();
            return v;
        case 't':
            word("true");
            v.type = Json::Bool;
            v.flag = true;
            return v;
        case 'f':
            word("false");
            v.type = Json::Bool;
            return v;
        case 'n':
            word("null");
            return v;
        default: {
            const char *s = p;
            while (p < end && std::strchr("+-.0123456789eE", *p))
                p++;
            if (p == s)
                fail();
            v.type = Json::Number;
            v.text.assign(s, p);
            return v;
        }
        }
    }

  private:
    const char *p, *end;

    [[noreturn]] void fail() { throw std::runtime_error("conformance file: bad JSON"); }
    char peek() {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t'))
            p++;
        return p < end ? *p : '\0';
    }
    bool accept(char c) { return peek() == c && (p++, true); }
    void expect(char c) {
        if (!accept(c))
            fail();
    }
    void word(const char *w) {
        size_t n = std::strlen(w);
        if ((size_t)(end - p) < n || std::memcmp(p, w, n) != 0)
            fail();
        p += n;
    }
    std::string string() {
        expect('"');
        const char *s = p;
        while (p < end && *p != '"')
            if (*p++ == '\\')
                fail();
        if (p == end)
            fail();
        return std::string(s, p++);
    }
};

inline Json read_json(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("cannot open " + path);
    std::string s;
    char buf[1 << 16];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;)
        s.append(buf, n);
    std::fclose(f);
    JsonParser parser(s);
    return parser.value();
}

/* FIPS 180-4, as lowercase hexadecimal. */
inline std::string sha256_hex(const void *data, size_t len) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const uint8_t *b = static_cast<const uint8_t *>(data);
    std::vector<uint8_t> m(b, b + len);
    m.push_back(0x80);
    while (m.size() % 64 != 56)
        m.push_back(0);
    for (int i = 7; i >= 0; i--)
        m.push_back((uint8_t)((uint64_t)len * 8 >> (8 * i)));
    for (size_t o = 0; o < m.size(); o += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)m[o + 4 * i] << 24 | (uint32_t)m[o + 4 * i + 1] << 16 |
                   (uint32_t)m[o + 4 * i + 2] << 8 | m[o + 4 * i + 3];
        for (int i = 16; i < 64; i++)
            w[i] = w[i - 16] + w[i - 7] +
                   (std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)) +
                   (std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10));
        uint32_t s[8];
        std::memcpy(s, h, sizeof h);
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = s[7] + (std::rotr(s[4], 6) ^ std::rotr(s[4], 11) ^ std::rotr(s[4], 25)) +
                          ((s[4] & s[5]) ^ (~s[4] & s[6])) + k[i] + w[i];
            uint32_t t2 = (std::rotr(s[0], 2) ^ std::rotr(s[0], 13) ^ std::rotr(s[0], 22)) +
                          ((s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]));
            std::memmove(s + 1, s, 7 * sizeof(uint32_t));
            s[4] += t1;
            s[0] = t1 + t2;
        }
        for (int i = 0; i < 8; i++)
            h[i] += s[i];
    }
    std::string out;
    char hex[9];
    for (uint32_t x : h) {
        std::snprintf(hex, sizeof hex, "%08x", x);
        out += hex;
    }
    return out;
}
