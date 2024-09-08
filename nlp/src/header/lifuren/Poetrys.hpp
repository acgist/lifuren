/**
 * 诗词工具
 */
#ifndef LFR_HEADER_NLP_POETRYS_HPP
#define LFR_HEADER_NLP_POETRYS_HPP

#include <string>
#include <vector>

namespace lifuren {
namespace poetrys {

extern std::vector<std::string> toChars(const std::string& poetry);

extern std::vector<std::string> toWords(const std::string& poetry);

extern std::vector<std::string> toSegments(const std::string& poetry);

extern std::string replaceSymbol(const std::string& poetry);

} // END OF poetrys
} // END OF lifuren

#endif // LFR_HEADER_NLP_POETRYS_HPP