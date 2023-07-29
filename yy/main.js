const menu    = require('./src/menu');
const event   = require('./src/event');
const config  = require('./src/config');
const process = require('process');
const {
  app,
  BrowserWindow
} = require('electron');

process.title = 'lifuren';

/**
 * 创建窗口
 */
function createWindow() {
  const window = new BrowserWindow({
    width : 800,
    height: 600,
    webPreferences: {
      nodeIntegration : true,
      contextIsolation: false,
    }
  });
  menu.createMenu(window);
  event.listenEvent(window);
  window.loadFile('./src/index.html').then(() => {
  });
  if(config.env.trim() === 'dev') {
    window.webContents.openDevTools();
  }
  return window;
}

app.whenReady().then(() => {
  createWindow();
});

app.on('window-all-closed', () => {
  app.quit();
});
