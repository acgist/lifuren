const { app, Menu } = require('electron')

const template = [
  {
    label: '文件',
    submenu: [
      { label: '设置', click: setting },
      { label: '退出', click: exit }
    ]
  },
  {
    label: '图片',
    submenu: [
      { label: '采集' },
      { label: '训练' },
      { label: '预测' }
    ]
  },
  {
    label: '视频',
    submenu: [
      { label: '采集' },
      { label: '训练' },
      { label: '预测' }
    ]
  },
  {
    label: '诗词',
    submenu: [
      { label: '采集' },
      { label: '训练' },
      { label: '预测' }
    ]
  },
  {
    label: '李夫人'
  },
  {
    label: '关于',
    submenu: [
      { label: '关于' },
      { label: '帮助', click: help }
    ]
  }
];

function setting() {
}

function exit() {
  app.exit();
}

function help() {
}

function buildMenu() {
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

module.exports = { buildMenu };
