/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 乐谱
 * 
 * https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/
 * https://www.w3.org/2021/06/musicxml40/musicxml-reference/examples/
 * https://www.w3.org/2021/06/musicxml40/musicxml-reference/element-tree/
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_BOOT_MUSIC_SCORE_HPP
#define LFR_HEADER_BOOT_MUSIC_SCORE_HPP

#include <map>
#include <string>
#include <vector>

namespace lifuren::music {

/**
 * 音符
 */
class Note {

// TODO: 音高、符号、指法fingering...

};

/**
 * 小节
 */
class Measure {

public:
    std::vector<Note> noteList; // 音符列表
    
};

/**
 * 乐谱
 */
class Score {

public:
    std::string name;   // 名称
    std::string author; // 作者
    std::map<std::string, std::vector<Measure>> measureMap; // 多声部小节列表

public:
    /**
     * @return 是否为空
     */
    bool empty();

};

/**
 * @param file 文件路径
 * 
 * @return 乐谱
 */
extern Score load_xml(const std::string& file);

/**
 * @param file  文件路径
 * @param score 乐谱
 * 
 * @return 是否成功
 */
extern bool save_xml(const std::string& file, const Score& score);

} // END OF lifuren::music

#endif // END OF LFR_HEADER_BOOT_MUSIC_SCORE_HPP
