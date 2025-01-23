#include "lifuren/RAGClient.hpp"

// TODO: GCC/G++ 13+
// #include <format>

#include <unordered_map>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/Client.hpp"
#include "lifuren/EmbeddingClient.hpp"

// 检测索引
static bool indexExists(const size_t& id, std::shared_ptr<lifuren::RestClient> client);
// 创建索引
static bool indexCreate(const size_t& id, std::shared_ptr<lifuren::RestClient> client, size_t dims);
// 删除索引
static bool indexDelete(const size_t& id, std::shared_ptr<lifuren::RestClient> client);

lifuren::ElasticSearchRAGClient::ElasticSearchRAGClient(const std::string& path, const std::string& embedding) :RAGClient(path, embedding) {
    const auto& elasticsearchConfig = lifuren::config::CONFIG.elasticsearch;
    this->restClient = std::make_shared<lifuren::RestClient>(elasticsearchConfig.api);
    this->restClient->auth(elasticsearchConfig);
    if(
        indexExists(this->id, this->restClient) &&
        indexCreate(this->id, this->restClient, this->embeddingClient->getDims())
    ) {
        // -
    }
}

lifuren::ElasticSearchRAGClient::~ElasticSearchRAGClient() {
}

std::vector<float> lifuren::ElasticSearchRAGClient::index(const std::string& prompt) {
    const auto vector = std::move(this->embeddingClient->getVector(prompt));
    if(this->donePromptEmplace(prompt)) {
        return vector;
    }
    if(vector.empty()) {
        return vector;
    }
    const nlohmann::json body = {
        { "vector" , vector },
        { "content", prompt }
    };
    this->restClient->postJson("/" + std::to_string(this->id) + "/_doc", body.dump());
    return vector;
}

std::vector<std::string> lifuren::ElasticSearchRAGClient::search(const std::vector<float>& prompt, const uint8_t size) const {
    const nlohmann::json body = {
        { "knn", {
            { "k"             ,  size    },
            { "field"         , "vector" },
            { "query_vector"  ,  prompt  },
            { "num_candidates",  100     },
        } }
    };
    const auto response = std::move(this->restClient->postJson("/" + std::to_string(this->id) + "/_search", body.dump()));
    if(!response) {
        return {};
    }
    const nlohmann::json data = std::move(nlohmann::json::parse(response.body));
    if(data.find("hits") == data.end()) {
        return {};
    }
    const nlohmann::json& hits = data["hits"];
    std::vector<std::string> ret;
    ret.reserve(hits["total"]["value"].get<int>());
    const nlohmann::json& docs = hits["hits"];
    for(const auto& doc : docs) {
        ret.push_back(doc["_source"]["content"].get<std::string>());
    }
    return ret;
}

static bool indexExists(const size_t& id, std::shared_ptr<lifuren::RestClient> client) {
    SPDLOG_DEBUG("检查ElasticSearch索引：{}", id);
    return client->head("/" + std::to_string(id));
}

static bool indexCreate(const size_t& id, std::shared_ptr<lifuren::RestClient> client, size_t dims) {
    SPDLOG_DEBUG("创建ElasticSearch索引：{}", id);
    const nlohmann::json body = {
        { "mappings", {
            { "properties", {
                { "vector", {
                    { "type",       "dense_vector" },
                    { "dims",        dims          },
                    { "index",       true          },
                    { "similarity", "l2_norm"      }
                } },
                { "content", {
                    { "type" , "text" },
                    { "index",  false },
                    { "store",  true  }
                } }
            } }
        } }
    };
    return client->putJson("/" + std::to_string(id), body.dump());
}

[[maybe_unused]] static bool indexDelete(const size_t& id, std::shared_ptr<lifuren::RestClient> client) {
    SPDLOG_DEBUG("删除ElasticSearch索引：{}", id);
    return client->del("/" + std::to_string(id));
}
