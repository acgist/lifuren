/**
 * 播放器
 */
class Player {

  audio_ctx      = null;      // 播放器上下文
  audio_keys     = new Map(); // 键盘
  audio_source   = new Map(); // 音源
  audio_playing  = false;     // 播放
  piano_selector = "";        // 琴键选择器
  active_keys    = [];        // 激活键盘
  helmholtz_pitch_notation  = new Map(); // 赫尔姆霍茨音调记号法
  scientific_pitch_notation = new Map(); // 科学音调记号法

  constructor(piano_selector = "#piano_player .key") {
    this.piano_selector = piano_selector;
  }

  async listen() {
    document.querySelectorAll(this.piano_selector).forEach(key => {
      const key_code   = key.getAttribute("key-code");
      const key_code_h = key.childNodes[1].text;
      const key_code_s = key.childNodes[2].text;
      this.audio_keys.set(key_code, key);
      this.helmholtz_pitch_notation.set(key_code_h, key_code);
      this.scientific_pitch_notation.set(key_code_s, key_code);
      console.debug("注册键盘", key_code, key_code_h, key_code_s);
      key.onmouseup = () => {
        if(this.audio_playing) {
          // -
        } else {
          key.classList.remove("active");
        }
      };
      key.onmouseout = () => {
        if(this.audio_playing) {
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
        } else {
          // -
        }
      };
    });
  }

  async register_audio_source(id, type, audio) {
    if(!this.audio_ctx) {
      this.audio_ctx = new AudioContext();
    }
    const rawData  = atob(audio)
    const rawArray = new Uint8Array(rawData.length);
    for (let i = 0; i < rawData.length; ++i) {
      rawArray[i] = rawData.charCodeAt(i);
    }
    let audio_source = this.audio_source.get(type);
    if(!audio_source) {
      audio_source = new Map();
      this.audio_source.set(type, audio_source);
    }
    this.audio_ctx.decodeAudioData(rawArray.buffer.slice(0)).then(audioData => {
      audio_source.set(id, audioData);
      console.debug("注册音源", id, type, audioData.length);
    });
  }

  async reset_key() {
    for(const key of this.active_keys) {
      key.classList.remove("active");
    }
    this.active_keys.length = 0;
  }

  async play_key(key, key_code, key_code_h, key_code_s, type) {
    if(!key) {
      if(!key_code) {
        if(key_code_h) {
          key_code = this.helmholtz_pitch_notation.get(key_code_h);
        } else if(key_code_s) {
          key_code = this.scientific_pitch_notation.get(key_code_s);
        } else {
          console.warn("缺少键盘编码");
          return;
        }
      }
      key = this.audio_keys.get(key_code);
    }
    if(!key) {
      console.warn("键盘匹配失败");
      return;
    }
    if(!key_code) {
      key_code = key.getAttribute("key-code");
    }
    key.classList.add("active");
    this.active_keys.push(key);
    this.play_audio(key_code, type);
  }

  async play_audio(key_code, type = "piano") {
    console.debug("播放音频", key_code);
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

  async play_list(list, play_next, play_ended, index = 0, old_time = 0.0, types = null) {
    if(index === 0) {
      this.audio_playing = true;
    }
    while(index < list.length && this.audio_playing) {
      const { id, note, time, rest } = list[index];
      if(old_time == time) {
        ++index;
        if(rest) {
          // -
        } else {
          this.play_key(null, note + "", null, null, types ? types[id] : "piano");
        }
      } else {
        setTimeout(() => {
          if(play_next) {
            play_next();
          }
          this.reset_key();
          this.play_list(list, play_next, play_ended, index, time, types);
        }, (time - old_time) * 1000);
        return;
      }
    }
    console.debug("播放完成");
    if(play_ended) {
      play_ended();
    }
    this.reset_key();
  }

  async stop_play() {
    this.audio_playing = false;
  }

};