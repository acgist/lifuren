/**
 * 服务终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CLIENT_POETIZE_CLIENT_HPP
#define LFR_HEADER_CLIENT_POETIZE_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

/**
 * 诗词终端
 */
class PoetizeClient : public Client {

};

class RNNPoetizeClient : public PoetizeClient {

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CLIENT_POETIZE_CLIENT_HPP
