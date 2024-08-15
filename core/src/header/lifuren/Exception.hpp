/**
 * 异常
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_EXCEPTION_HPP
#define LFR_HEADER_CORE_EXCEPTION_HPP

#include <string>
#include <exception>

namespace lifuren {

const std::string CODE_0000 = "0000"; // 成功
const std::string CODE_1000 = "1000"; // 系统异常
const std::string CODE_2000 = "2000"; // 外部异常（参数错误、配置错误）
const std::string CODE_9999 = "9999"; // 未知异常

/**
 * 异常
 */
class Exception : public std::exception {

public:
    // 错误编码
    std::string code;
    // 错误信息
    std::string message;

public:
    /**
     * @param code    错误编码
     * @param message 错误信息
     */
    Exception(const std::string& code = CODE_9999, const std::string& message = "未知异常");
    virtual ~Exception();

public:
    /**
     * @param code    错误编码
     * @param message 错误信息
     * 
     * @throws 异常
     */
    static void throwException(const std::string& code = CODE_9999, const std::string& message = "未知异常");
    /**
     * @param ret     条件
     * @param code    错误编码
     * @param message 错误信息
     * 
     * @throws 异常（条件正确抛出异常）
     */
    static void trueThrow(bool ret, const std::string& code = CODE_9999, const std::string& message = "未知异常");
    /**
     * @param ret     条件
     * @param code    错误编码
     * @param message 错误信息
     * 
     * @throws 异常（条件错误抛出异常）
     */
    static void falseThrow(bool ret, const std::string& code = CODE_9999, const std::string& message = "未知异常");
    /**
     * @param T 泛型
     * 
     * @param source  原始对象
     * @param target  比较对象
     * @param code    错误编码
     * @param message 错误信息
     * 
     * @throws 异常（相等抛出异常）
     */
    template<typename T>
    static void equalThrow(const T& source, const T& target, const std::string& code = CODE_9999, const std::string& message = "未知异常") {
        trueThrow(&source == &target, code, message);
    }
    /**
     * @param T 泛型
     * 
     * @param source  原始对象
     * @param target  比较对象
     * @param code    错误编码
     * @param message 错误信息
     * 
     * @throws 异常（相等抛出异常）
     */
    template<typename T>
    static void notEqualThrow(const T& source, const T& target, const std::string& code = CODE_9999, const std::string& message = "未知异常") {
        falseThrow(&source == &target, code, message);
    }
    virtual const char* what() const noexcept override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CORE_EXCEPTION_HPP
