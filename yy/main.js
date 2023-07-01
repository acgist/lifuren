const process = require('process');
const { app, BrowserWindow } = require('electron');

const menu    = require('./menu');
const event   = require('./event');

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
  window.loadFile('./index.html').then(() => {
  });
  const env = process.env.NODE_ENV;
  if(env && env.trim() === 'dev') {
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
