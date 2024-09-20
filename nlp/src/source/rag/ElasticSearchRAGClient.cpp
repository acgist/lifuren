/**
 * https://www.elastic.co/guide/en/elasticsearch/reference/current/rest-apis.html
 */
#include "lifuren/RAG.hpp"

// TODO: GCC/G++ 13+
// #include <format>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

static bool indexExists(const size_t& id, std::shared_ptr<lifuren::RestClient> client);
static bool indexCreate(const size_t& id, std::shared_ptr<lifuren::RestClient> client, size_t dims);
static bool indexDelete(const size_t& id, std::shared_ptr<lifuren::RestClient> client);
static bool index(const size_t& id, const std::string& content, const std::vector<float>& vector, std::shared_ptr<lifuren::RestClient> client);
static std::vector<std::string> search(const size_t& id, const std::vector<float>& vector, const int& size, std::shared_ptr<lifuren::RestClient> client);

lifuren::ElasticSearchRAGClient::ElasticSearchRAGClient(const std::string& path, const std::string& embedding) :RAGClient(path, embedding) {
    const auto& elasticsearchConfig = lifuren::config::CONFIG.elasticsearch;
    this->restClient = std::make_shared<lifuren::RestClient>(elasticsearchConfig.api);
    this->restClient->auth(elasticsearchConfig);
}

lifuren::ElasticSearchRAGClient::~ElasticSearchRAGClient() {
}

std::vector<float> lifuren::ElasticSearchRAGClient::index(const std::string& content) {
    auto&& vector = this->embeddingClient->getVector(content);
    ::index(this->id, content, vector, this->restClient);
    return vector;
}

std::vector<std::string> lifuren::ElasticSearchRAGClient::search(const std::vector<float>& prompt, const int size) {
    return ::search(this->id, prompt, size, this->restClient);
}

void lifuren::ElasticSearchRAGClient::loadIndex() {
    lifuren::RAGClient::loadIndex();
    if(indexExists(this->id, this->restClient)) {
        return;
    }
    indexCreate(this->id, this->restClient, this->embeddingClient->getDims());
}

void lifuren::ElasticSearchRAGClient::truncateIndex() {
    indexDelete(this->id, this->restClient);
    lifuren::RAGClient::truncateIndex();
}

static bool indexExists(const size_t& id, std::shared_ptr<lifuren::RestClient> client) {
    SPDLOG_INFO("检查ElasticSearch索引：{}", id);
    return client->head("/" + std::to_string(id));
}

static bool indexCreate(const size_t& id, std::shared_ptr<lifuren::RestClient> client, size_t dims) {
    SPDLOG_DEBUG("创建ElasticSearch索引：{}", id);
    const nlohmann::json body = {
        { "mappings", {
            { "properties", {
                { "vector", {
                    { "type"      , "dense_vector" },
                    { "dims"      , dims           },
                    { "index"     , true           },
                    { "similarity", "l2_norm"      }
                } },
                { "content", {
                    { "type" , "text" },
                    { "index", false  },
                    { "store", true   }
                } }
            } }
        } }
    };
    return client->putJson("/" + std::to_string(id), body.dump());
}

static bool indexDelete(const size_t& id, std::shared_ptr<lifuren::RestClient> client) {
    return client->deletePath("/" + std::to_string(id));
}

static bool index(const size_t& id, const std::string& content, const std::vector<float>& vector, std::shared_ptr<lifuren::RestClient> client) {
    // TODO: 是否需要重复验证
    const nlohmann::json body = {
        { "vector" , vector  },
        { "content", content }
    };
    return client->postJson("/" + std::to_string(id) + "/_doc", body.dump());
}

static std::vector<std::string> search(const size_t& id, const std::vector<float>& vector, const int& size, std::shared_ptr<lifuren::RestClient> client) {
    const nlohmann::json body = {
        { "knn", {
            { "k"             , size     },
            { "field"         , "vector" },
            { "query_vector"  , vector   },
            { "num_candidates", 100      },
        } }
    };
    auto&& response = client->postJson("/" + std::to_string(id) + "/_search", body.dump());
    if(!response) {
        return {};
    }
    nlohmann::json data = nlohmann::json::parse(response.body);
    if(data.find("hits") == data.end()) {
        return {};
    }
    nlohmann::json& hits = data["hits"];
    std::vector<std::string> ret;
    ret.reserve(hits["total"]["value"].get<int>());
    nlohmann::json& docs = hits["hits"];
    for(const auto& doc : docs) {
        ret.push_back(doc["_source"]["content"].get<std::string>());
    }
    return ret;
}
