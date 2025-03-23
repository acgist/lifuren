/**
 * 播放器
 */
class Player {

  auto_playing = false; // 自动播放
  audio_ctx    = null;  // 播放器上下文
  audio_keys   = new Map(); // 键盘
  audio_source = new Map(); // 音源
  piano_keys_selector = "";
  helmholtz_pitch_notation  = new Map(); // 赫尔姆霍茨音调记号法
  scientific_pitch_notation = new Map(); // 科学音调记号法

  constructor(piano_keys_selector = "#piano_container .key") {
    this.piano_keys_selector = piano_keys_selector;
  }

  listen() {
    document.querySelectorAll(this.piano_keys_selector).forEach(key => {
      const key_code   = key.getAttribute("key-code");
      const key_code_h = key.childNodes[1].text;
      const key_code_s = key.childNodes[2].text;
      this.audio_keys.set(key_code, key);
      this.helmholtz_pitch_notation.set(key_code_h, key_code);
      this.scientific_pitch_notation.set(key_code_s, key_code);
      key.onmouseup = () => {
        if(this.auto_playing) {
          // -
        } else {
          key.classList.remove("active");
        }
      };
      key.onmouseout = () => {
        if(this.auto_playing) {
          // -
        } else {
          key.classList.remove("active");
        }
      };
      key.onmousedown = () => {
        this.play_key(key);
      };
      key.onmouseover = (e) => {
        if(e.buttons) {
          this.play_key(key);
        }
      };
    });
  }

  register(id, type, audio) {
    if(!this.audio_ctx) {
      this.audio_ctx = new AudioContext();
    }
    const rawData  = atob(audio)
    const rawArray = new Uint8Array(rawData.length);
    for (let i = 0; i < rawData.length; ++i) {
      rawArray[i] = rawData.charCodeAt(i);
    }
    let source = this.audio_source.get(type);
    if(!source) {
      source = new Map();
      this.audio_source.set(type, source);
    }
    this.audio_ctx.decodeAudioData(rawArray.buffer.slice(0)).then(data => {
      source.set(id, data);
      console.info("注册音源", id, type, data.length);
    });
  }

  play_key(key, key_code, key_code_h, key_code_s) {
    if(!key) {
      if(!key_code) {
        if(key_code_h) {
          key_code = this.helmholtz_pitch_notation.get(key_code_h);
        } else if(key_code_s) {
          key_code = this.scientific_pitch_notation.get(key_code_s);
        } {
          return;
        }
      }
      key = this.audio_keys.get(key_code);
    }
    if(!key) {
      return;
    }
    if(!key_code) {
      key_code = key.getAttribute("key-code");
    }
    key.classList.add("active");
    this.play(key_code);
  }

  play(key_code, type = "piano") {
    if(!this.audio_ctx) {
      console.warn("没有注册音频上下文");
      return;
    }
    const buffer = this.audio_source.get(type)?.get(key_code);
    if(!buffer) {
      console.warn("没有注册音频", key_code, type);
      return;
    }
    const source  = this.audio_ctx.createBufferSource();
    source.loop   = false;
    source.buffer = buffer;
    source.connect(this.audio_ctx.destination);
    source.start(0);
    // TODO: 判断是否需要释放
  }

  playList(list, next, index = 0, old_time = 0, types = null) {
    while(index < list.length) {
      const { id, note, time } = list[index];
      ++index;
      if(old_time == time) {
        this.play_key(null, note + "");
      } else {
        next();
        setTimeout(() => {
          old_time = time;
          this.playList(list, next, index, old_time, types);
        }, (time - old_time) * 1000);
        break;
      }
    }
  }

  stopPlay() {

  }
};