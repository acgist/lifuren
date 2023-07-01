const fs = require('fs');
const { app, shell, ipcMain } = require('electron')

let window;

/**
 * 监听事件
 */
function listenEvent(win) {
  window = win;
  ipcMain.on('event-setting',    (event, message) => setting(event,   message));
  ipcMain.on('event-file-clean', (event, message) => fileClean(event, message));
}

/**
 * 加载配置
 */
function setting(event, message) {
  if(message) {

  } else {
    fs.promises.readFile("./src/setting.json").then(value => {
      const setting = JSON.parse(value.toString());
      window.webContents.send("setting", setting);
    });
  }
}

/**
 * 清理文件
 * 
 * @param {*} event   事件
 * @param {*} message 消息
 */
function fileClean(event, message) {
  console.info("清理文件", message);
}

module.exports = { listenEvent };
