const path                   = require('path');
const menu                   = require('./src/menu');
const config                 = require('./src/config');
const process                = require('process');
const { app, BrowserWindow } = require('electron');

function createWindow() {
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
  createWindow();
});

app.on('window-all-closed', () => {
  app.quit();
});
