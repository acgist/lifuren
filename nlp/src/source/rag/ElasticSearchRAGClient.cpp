/**
 * https://www.elastic.co/guide/en/elasticsearch/reference/current/rest-apis.html
 */
#include "lifuren/RAGClient.hpp"

#include <format>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

static bool indexExists(const size_t& id, std::shared_ptr<lifuren::RestClient> client);
static bool indexCreate(const size_t& id, std::shared_ptr<lifuren::RestClient> client, size_t dims);
static bool indexDelete(const size_t& id, std::shared_ptr<lifuren::RestClient> client);

static bool index(const size_t& id, const std::string& content, const std::vector<float> vector, std::shared_ptr<lifuren::RestClient> client);
static std::vector<std::string> search(const size_t& id, const std::vector<float>, const int& size, std::shared_ptr<lifuren::RestClient> client);

lifuren::ElasticSearchRAGClient::ElasticSearchRAGClient(const std::string& path, const std::string& embedding) :RAGClient(path, embedding) {
    const auto& elasticsearchConfig = lifuren::config::CONFIG.elasticsearch;
    this->restClient = std::make_shared<lifuren::RestClient>(elasticsearchConfig.api);
    this->restClient->auth(elasticsearchConfig);
}

lifuren::ElasticSearchRAGClient::~ElasticSearchRAGClient() {
}

std::vector<float> lifuren::ElasticSearchRAGClient::index(const std::string& content) {
    if(!this->exists) {
        this->exists = indexExists(this->id, this->restClient);
        if(!this->exists) {
            SPDLOG_DEBUG("创建ElasticSearch索引：{}", this->id);
            this->exists = indexCreate(this->id, this->restClient, this->embeddingClient->getDims());
        }
    }
    auto&& vector = this->embeddingClient->getSegmentVector(content);
    ::index(this->id, content, vector, this->restClient);
    return {};
}

std::vector<std::string> lifuren::ElasticSearchRAGClient::search(const std::string& prompt) {
    const auto& ragConfig = lifuren::config::CONFIG.rag;
    auto&& vector = this->embeddingClient->getSegmentVector(prompt);
    return ::search(this->id, vector, ragConfig.size, this->restClient);
}

bool lifuren::ElasticSearchRAGClient::deleteRAG() {
    SPDLOG_INFO("删除ElasticSearch索引：{}", this->id);
    this->truncateIndex();
    return indexDelete(this->id, this->restClient);
}

static bool indexExists(const size_t& id, std::shared_ptr<lifuren::RestClient> client) {
    return client->head("/" + std::to_string(id));
}

static bool indexCreate(const size_t& id, std::shared_ptr<lifuren::RestClient> client, size_t dims) {
    auto response = client->putJson("/" + std::to_string(id), std::format(R"(
        {{
            "mappings": {{
                "properties": {{
                    "vector": {{
                        "type": "dense_vector",
                        "dims": {:d},
                        "index": true,
                        "similarity": "l2_norm"
                    }},
                    "content": {{
                        "type": "text"
                    }}
                }}
            }}
        }}
    )", dims));
    return response;
}

static bool indexDelete(const size_t& id, std::shared_ptr<lifuren::RestClient> client) {
    return client->deletePath("/" + std::to_string(id));
}

static bool index(const size_t& id, const std::string& content, const std::vector<float> vector, std::shared_ptr<lifuren::RestClient> client) {
    nlohmann::json body = {
        { "vector",  vector  },
        { "content", content }
    };
    return client->postJson("/" + std::to_string(id) + "/_doc", body.dump());
}

static std::vector<std::string> search(const size_t& id, const std::vector<float> vector, const int& size, std::shared_ptr<lifuren::RestClient> client) {
    nlohmann::json body = {
        { "knn", {
            { "k", size },
            { "field", "vector" },
            { "query_vector", vector },
            { "num_candidates", 100 },
        } },
        {
            "fields", { "content" }
            // "fields", { "vector", "content" }
        }
    };
    auto response = client->postJson("/" + std::to_string(id) + "/_search", body.dump());
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
    for(auto& doc : docs) {
        ret.push_back(doc["_source"]["content"].get<std::string>());
    }
    return ret;
}
