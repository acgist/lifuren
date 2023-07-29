const dotenv = require("dotenv");

if(process.env.NODE_ENV && process.env.NODE_ENV !== "dev") {
  dotenv.config({ path: `.env.${process.env.NODE_ENV}` });
} else {
  dotenv.config({ path: `.env` });
}

/**
 * 配置
 */
module.exports = {
  env: process.env.ENV
};
