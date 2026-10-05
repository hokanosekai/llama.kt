// UTF-8 <-> UTF-16 conversion for the JNI boundary (TEN-70).
//
// JNI's own UTF-8 entry points (GetStringUTFChars / NewStringUTF) speak
// *modified* UTF-8, not UTF-8: a character outside the BMP (an emoji) is
// encoded as two 3-byte surrogate sequences (CESU-8, 6 bytes) instead of one
// 4-byte sequence. llama.cpp wants real UTF-8, so tensai_jni.cpp goes through
// the UTF-16 entry points (GetStringChars / NewString) and converts here.
//
// Header-only and JNI-free so a host-side test can include it as-is.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tensai_utf {

// UTF-16 -> UTF-8. A valid surrogate pair becomes one 4-byte sequence; a lone
// surrogate (high without a low after it, or a low on its own) becomes U+FFFD.
inline std::string utf16_to_utf8(const char16_t* s, size_t n) {
    std::string out;
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        uint32_t cp = s[i];
        if (cp >= 0xD800 && cp <= 0xDFFF) {
            const bool high = cp <= 0xDBFF;
            if (high && i + 1 < n && s[i + 1] >= 0xDC00 && s[i + 1] <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (static_cast<uint32_t>(s[i + 1]) - 0xDC00);
                i++;
            } else {
                cp = 0xFFFD;
            }
        }
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

// UTF-8 -> UTF-16. A 4-byte sequence becomes a surrogate pair. Anything that
// is not well-formed UTF-8 (stray continuation byte, C0/C1/F5..FF lead byte,
// overlong form, encoded surrogate i.e. CESU-8, code point above U+10FFFF,
// sequence cut short) becomes U+FFFD, one per maximal ill-formed subpart: the
// lead byte plus however many continuation bytes were valid for it are
// consumed, and decoding resumes at the first byte that broke the sequence.
// Same policy as the WHATWG decoder and ICU.
inline std::u16string utf8_to_utf16(const char* str, size_t n) {
    const unsigned char* s = reinterpret_cast<const unsigned char*>(str);
    std::u16string out;
    out.reserve(n);
    size_t i = 0;
    while (i < n) {
        const unsigned char c = s[i];
        if (c < 0x80) {
            out.push_back(static_cast<char16_t>(c));
            i++;
            continue;
        }
        size_t need;
        uint32_t cp;
        // Allowed range of the first continuation byte; it is what rules out
        // overlong forms (E0, F0), surrogates (ED) and > U+10FFFF (F4).
        unsigned char lo = 0x80, hi = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) {
            need = 1; cp = c & 0x1F;
        } else if (c >= 0xE0 && c <= 0xEF) {
            need = 2; cp = c & 0x0F;
            if (c == 0xE0) lo = 0xA0;
            else if (c == 0xED) hi = 0x9F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            need = 3; cp = c & 0x07;
            if (c == 0xF0) lo = 0x90;
            else if (c == 0xF4) hi = 0x8F;
        } else {
            out.push_back(u'�');  // continuation byte or invalid lead
            i++;
            continue;
        }
        size_t j = i + 1;
        bool ok = true;
        for (size_t k = 0; k < need; k++, j++) {
            if (j >= n || s[j] < lo || s[j] > hi) { ok = false; break; }
            cp = (cp << 6) | (s[j] & 0x3F);
            lo = 0x80; hi = 0xBF;
        }
        i = j;
        if (!ok) {
            out.push_back(u'�');
            continue;
        }
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out.push_back(static_cast<char16_t>(0xD800 + (cp >> 10)));
            out.push_back(static_cast<char16_t>(0xDC00 + (cp & 0x3FF)));
        } else {
            out.push_back(static_cast<char16_t>(cp));
        }
    }
    return out;
}

}  // namespace tensai_utf
