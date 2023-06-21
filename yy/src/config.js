const process = require('process');
const { app } = require('electron');

process.title = 'lifuren';

app.setting = {};
const argv  = process.argv;
for(let arg of argv) {
  if(arg.indexOf('--') === 0) {
    arg = arg.substring(2);
    const index = arg.indexOf('=');
    let key, value;
    if(index >= 0) {
      key   = arg.substring(0, index);
      value = arg.substring(index + 1);
      app.setting[key] = value;
    } else {
      app.setting[key] = key;
    }
    console.info('配置参数', key, "=", value);
  }
}
