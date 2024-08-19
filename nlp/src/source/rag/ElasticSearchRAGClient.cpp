/**
 * https://www.elastic.co/guide/en/elasticsearch/reference/current/rest-apis.html
 */
#include "lifuren/RAG.hpp"

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

static bool indexExists(const size_t& id, std::shared_ptr<lifuren::RestClient> client);
static bool indexCreate(const size_t& id, std::shared_ptr<lifuren::RestClient> client);
static bool indexDelete(const size_t& id, std::shared_ptr<lifuren::RestClient> client);

static bool ragIndex(const size_t& id, const std::string& content, std::shared_ptr<lifuren::RestClient> client);
static std::vector<std::string> ragSearch(const size_t& id, const std::string& prompt, const int& size, std::shared_ptr<lifuren::RestClient> client);

static bool textIndex(const size_t& id, const std::string& content, std::shared_ptr<lifuren::RestClient> client);
static std::vector<std::string> textSearch(const size_t& id, const std::string& prompt, const int& size, std::shared_ptr<lifuren::RestClient> client);

lifuren::ElasticSearchRAGClient::ElasticSearchRAGClient(const std::string& path, const std::string& embedding) :RAGClient(path, embedding) {
    const auto& elasticsearchConfig = lifuren::config::CONFIG.elasticsearch;
    this->restClient = std::make_shared<lifuren::RestClient>(elasticsearchConfig.api);
    this->restClient->auth(elasticsearchConfig);
}

lifuren::ElasticSearchRAGClient::~ElasticSearchRAGClient() {
}

std::vector<double> lifuren::ElasticSearchRAGClient::index(const std::string& content) {
    if(!this->exists) {
        this->exists = indexExists(this->id, this->restClient);
        if(!this->exists) {
            SPDLOG_DEBUG("创建ElasticSearch索引：{}", this->id);
            this->exists = indexCreate(this->id, this->restClient);
        }
    }
    const auto& elasticsearchConfig = lifuren::config::CONFIG.elasticsearch;
    if(elasticsearchConfig.embedding == "TEXT") {
        textIndex(this->id, content, this->restClient);
    } else {

    }
    return {};
}

std::vector<std::string> lifuren::ElasticSearchRAGClient::search(const std::string& prompt) {
    const auto& chatConfig = lifuren::config::CONFIG.chat;
    return textSearch(this->id, prompt, chatConfig.ragSize, this->restClient);
}

bool lifuren::ElasticSearchRAGClient::deleteRAG() {
    SPDLOG_INFO("删除ElasticSearch索引：{}", this->id);
    return indexDelete(this->id, this->restClient);
}

static bool indexExists(const size_t& id, std::shared_ptr<lifuren::RestClient> client) {
    auto response = client->head("/" + std::to_string(id));
    return response && response->status == httplib::StatusCode::OK_200;
}

static bool indexCreate(const size_t& id, std::shared_ptr<lifuren::RestClient> client) {
    auto response = client->putJson("/" + std::to_string(id), R"(
        {
            "mappings": {
                "properties": {
                    "content": { "type": "text" }
                }
            }
        }
    )");
    return response && response->status == httplib::StatusCode::OK_200;
}

static bool indexDelete(const size_t& id, std::shared_ptr<lifuren::RestClient> client) {
    auto response = client->deletePath("/" + std::to_string(id));
    return response && response->status == httplib::StatusCode::OK_200;
}

static bool textIndex(const size_t& id, const std::string& content, std::shared_ptr<lifuren::RestClient> client) {
    nlohmann::json body = {
        { "content", content }
    };
    auto response = client->postJson("/" + std::to_string(id) + "/_doc", body.dump());
    return response && response->status == httplib::StatusCode::Created_201;
}

static std::vector<std::string> textSearch(const size_t& id, const std::string& prompt, const int& size, std::shared_ptr<lifuren::RestClient> client) {
    nlohmann::json body = {
        { "size", size },
        { "query", {
            { "match",  {
                { "content", prompt }
            } }
        } }
    };
    auto response = client->postJson("/" + std::to_string(id) + "/_search", body.dump());
    if(!response) {
        return {};
    }
    nlohmann::json data = nlohmann::json::parse(response->body);
    if(data.find("hits") == data.end()) {
        return {};
    }
    nlohmann::json& hits = data["hits"];
    std::vector<std::string> vector;
    vector.reserve(hits["total"]["value"].get<int>());
    nlohmann::json& docs = hits["hits"];
    for(auto& doc : docs) {
        vector.push_back(doc["_source"]["content"].get<std::string>());
    }
    return vector;
}
