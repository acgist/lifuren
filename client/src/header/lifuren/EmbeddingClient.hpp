/**
 * 服务终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CLIENT_EMBEDDING_CLIENT_HPP
#define LFR_HEADER_CLIENT_EMBEDDING_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

/**
 * 词嵌入终端
 */
class EmbeddingClient : public Client {

public:
    static std::unique_ptr<lifuren::EmbeddingClient> getClient(const std::string& embedding);

};

class OllamaEmbeddingClient : public EmbeddingClient {

};

/**
 * Chinese-Word-Vectors词嵌入终端
 * 
 * 项目地址：https://github.com/Embedding/Chinese-Word-Vectors
 */
class ChineseWordVectorsEmbeddingClient : public EmbeddingClient {

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CLIENT_EMBEDDING_CLIENT_HPP
