/**
 * 李夫人
 */
class Lifuren {

  player    = null; // 播放器
  music_xml = null; // 乐谱内容
  display   = null; // 谱面渲染器
  
  a4_width  = 210; // A4宽度
  a4_height = 297; // A4高度

  score_selector = ""; // 谱面选择器
  piano_selector = ""; // 琴键选择器
  
  constructor(
    score_selector = "#score_container",
    piano_selector = "#piano_player .key"
  ) {
    this.score_selector = score_selector;
    this.piano_selector = piano_selector;
    // 初始化播放器
    this.player = new Player(piano_selector);
    this.player.listen();
    if(window.lfr_backend) {
      window.lfr_backend.postMessage("audio");
    }
    console.debug("初始化播放器");
    // 初始化渲染器
    this.display = new opensheetmusicdisplay.OpenSheetMusicDisplay(this.score_selector.substring(1));
    this.display.setOptions({
      backend   : "svg",
      drawTitle : true,
      autoResize: false,
      pageFormat: "A4_P",
      cursorsOptions: [{ type: 0, color: "#CCCC00", alpha: 0.6, follow: true }],
      pageBackgroundColor: "#FFFFFF",
    });
    console.debug("初始化渲染器");
  }
  
  async load_music_xml(music_xml) {
    this.stop_score();
    this.music_xml = music_xml;
    this.display.load(music_xml)
    .then(() => {
      this.display.render();
    });
  }
  
  async open_score() {
    const [picker] = await window.showOpenFilePicker({
      types: [{
        accept: { "text/xml": [".xml", ".musicxml"] },
        description: "乐谱"
      }]
    });
    const file = await picker?.getFile();
    if(file) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        await this.load_music_xml(e.target.result);
      };
      reader.readAsText(file);
    } else {
      console.info("用户没有选择文件");
    }
  };
  
  async play_score(play_ended) {
    const note_list = [];
    this.display.cursor.reset();
    const iterator = this.display.cursor.Iterator;
    while (!iterator.EndReached) {
      let bpm = 4;
      if(this.display.sheet.hasBPMInfo) {
        bpm = (1.0 * this.display.sheet.defaultStartTempoInBpm / 60).toFixed(2);
      }
      const voices = iterator.CurrentVoiceEntries;
      for(let i = 0; i < voices.length; i++) {
        const notes = voices[i].Notes;
        for (let j = 0; j < notes.length; j++) {
          const note = notes[j];
          note_list.push({
            "id"  : note.parentStaffEntry.parentStaff.idInMusicSheet,
            "note": note.halfTone - 12 + 3 + 1, // 钢琴键盘：十二平均律 + 只有三个键 + 从一开始
            "time": iterator.currentTimeStamp.RealValue * bpm,
            "rest": note.isRest()
          });
        }
      }
      iterator.moveToNext();
    }
    this.display.cursor.reset();
    this.display.cursor.show();
    this.player.play_list(note_list, () => {
      this.display.cursor.next();
    }, () => {
      this.display.cursor.hide();
      if(play_ended) {
        play_ended();
      }
    });
  }

  async stop_score() {
    this.player.stop_play();
  }

  async save_pdf() {
    const backends = this.display.drawer.Backends;
    let svgElement = backends[0].getSvgElement();
    let pageWidth  = this.a4_width;
    let pageHeight = this.a4_height;
    if (!this.display.rules.PageFormat?.IsUndefined) {
      pageWidth  = this.display.rules.PageFormat.width;
      pageHeight = this.display.rules.PageFormat.height;
    } else {
      pageHeight = pageWidth * svgElement.clientHeight / svgElement.clientWidth;
    }
    const pdf = new jspdf.jsPDF({
      unit  : "mm",
      format: [pageWidth, pageHeight],
      orientation: pageHeight > pageWidth ? "p" : "l"
    });
    for (let index = 0; index < backends.length; ++index) {
      if (index > 0) {
        pdf.addPage();
      }
      svgElement = backends[index].getSvgElement();
      await pdf.svg(svgElement, {
        x: 0,
        y: 0,
        width : pageWidth,
        height: pageHeight,
      })
    }
    pdf.save((this.display.sheet.FullNameString || "lifuren") + ".pdf");
  };
  
  async download_img(index, svgElement) {
    const canvas     = document.createElement("canvas");
    canvas.width     = svgElement.width.baseVal.value;
    canvas.height    = svgElement.height.baseVal.value;
    const ctx        = canvas.getContext("2d");
    const svgContent = new XMLSerializer().serializeToString(svgElement);
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0);
      const imgURL = canvas.toDataURL({ format: "image/png" });
      const dlLink = document.createElement('a');
      dlLink.href     = imgURL;
      dlLink.download = (this.display.sheet.FullNameString || "lifuren") + "-" + index + ".png";
      dlLink.dataset.downloadurl = ["image/png", dlLink.download, dlLink.href].join(':');
      document.body.appendChild(dlLink);
      dlLink.click();
      document.body.removeChild(dlLink);
    };
    let content = "";
    const chunk = 8 * 1024;
    const array = new TextEncoder().encode(svgContent);
    let i;
    for (i = 0; i < array.length / chunk; ++i) {
      content += String.fromCharCode(...array.slice(i * chunk, (i + 1) * chunk));
    }
    if(array.length % chunk != 0) {
      content += String.fromCharCode(...array.slice(i * chunk));
    }
    img.src = "data:image/svg+xml;base64," + btoa(content);
  }
  
  async save_img() {
    const backends = this.display.drawer.Backends;
    for(let i = 0; i < backends.length; ++i) {
      await this.download_img(i, backends[i].getSvgElement());
    }
  };
  
  async register_audio_source(id, type, audio) {
    if(this.player) {
      this.player.register_audio_source(id, type, audio);
    }
  }

};