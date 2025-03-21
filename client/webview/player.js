/**
 * 播放器
 */
class Player {

  audio_source = new Map(); // 音源
  audio_tracks = new Map(); // 音轨

  listen(selector) {
    document.querySelectorAll(selector).forEach(key => {
      key.onmouseup = () => {
        key.classList.remove("active");
      };
      key.onmousedown = () => {
        key.classList.add("active");
      };
      key.onmouseout = () => {
        key.classList.remove("active");
      };
      key.onmouseover = (e) => {
        if(e.buttons) {
          key.classList.add("active");
        }
      };
    });
  }

  play() {
  }

};