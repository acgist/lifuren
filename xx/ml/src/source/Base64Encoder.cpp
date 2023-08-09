#include "../header/DLibAll.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include "dlib/base64.h"
#include "dlib/compress_stream.h"

namespace lifuren {

namespace ml {

    void testBase64Encoder() {
        std::ifstream in("D:/tmp/ml.txt", std::ios::binary);
        std::istringstream sin;
        std::ostringstream sout;
        dlib::base64 encoder;
        dlib::compress_stream::kernel_1ea compressor;
        compressor.compress(in, sout);
        sin.str(sout.str());
        sout.clear();
        sout.str("");
        encoder.encode(sin, sout);
        std::string line;
        std::getline(sin, line);
        while (sin && line.size() > 0) {
            sout << line;
            std::getline(sin, line);
        }
        LOG(INFO) << "sout = " << sout.str() << std::endl;
        sin.clear();
        sin.str(sout.str());
        sout.clear();
        sout.str("");
        encoder.decode(sin, sout);
        sin.clear();
        sin.str(sout.str());
        sout.clear();
        sout.str("");
        compressor.decompress(sin, sout);
        LOG(INFO) << "sout = " << sout.str() << std::endl;
        in.close();
    }
    
}

}