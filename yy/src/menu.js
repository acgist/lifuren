const {
  app,
  shell,
  Menu
} = require('electron')

let window;

const template = [
  {
    label: '文件',
    submenu: [
      {
        label: '设置',
        click: () => menuTabs("file", "setting")
      },
      {
        label: '诗词',
        click: () => menuTabs("file", "poetry")
      },
      {
        label: '图片',
        click: () => menuTabs("file", "image")
      },
      {
        label: '视频',
        click: () => menuTabs("file", "video")
      },
      {
        label: '清理',
        click: () => menuTabs("file", "clean")
      },
      {
        label: '退出',
        click: () => app.exit()
      }
    ]
  },
  {
    label: '训练',
    submenu: [
      { label: '诗词标记' },
      { label: '诗词训练' },
      { label: '图片标记' },
      { label: '图片训练' },
      { label: '视频标记' },
      { label: '视频训练' },
    ]
  },
  {
    label: '预测',
    submenu: [
      { label: '吟诗' },
      { label: '桃面' },
      { label: '楚腰' }
    ]
  },
  {
    label: '关于',
    submenu: [
      {
        label: '源码',
        click: async () => {
          await shell.openExternal('https://gitee.com/acgist/lifuren')
        }
      },
      {
        label: '关于',
        click: async () => {
          await shell.openExternal('https://gitee.com/acgist/lifuren')
        }
      },
      {
        label: '帮助',
        click: async () => {
          await shell.openExternal('https://gitee.com/acgist/lifuren')
        }
      },
      {
        label: '作者',
        click: async () => {
          await shell.openExternal('https://www.acgist.com')
        }
      },
    ]
  }
];

/**
 * 创建菜单
 * 
 * @param {*} _window 窗口
 */
function createMenu(_window) {
  window = _window;
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

/**
 * 菜单切换
 * 
 * @param {*} tabs  tabs
 * @param {*} label label
 */
function menuTabs(tabs, label) {
  window.webContents.send("menu-tabs", {
    tabs,
    label
  });
}

module.exports = { createMenu };
