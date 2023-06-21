const menu                   = require('./src/menu');
const config                 = require('./src/config');
const { app, BrowserWindow } = require('electron');

/**
 * 创建窗口
 */
function buildWindow() {
  const window = new BrowserWindow({
    width : 800,
    height: 600,
    webPreferences: {
    }
  });
  menu.buildMenu();
  window.loadFile('./src/index.html');
  if(app.setting.env === 'dev') {
    window.webContents.openDevTools();
  }
}

app.whenReady().then(() => {
  config.initSetting();
  buildWindow();
});

app.on('window-all-closed', () => {
  app.quit();
});
