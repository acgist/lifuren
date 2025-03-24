/**
 * 李夫人
 */
class Lifuren {

  show_staff = true;  // 是否显示五线谱
  
  player         = null; // 播放器
  music_xml      = null; // 乐谱内容
  display_staff  = null; // 五线谱渲染器
  display_jianpu = null; // 简谱渲染器
  
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
    // 初始化五线谱渲染器
    this.display_staff = new opensheetmusicdisplay.OpenSheetMusicDisplay(this.score_selector.substring(1));
    this.display_staff.setOptions({
      backend   : "svg",
      drawTitle : true,
      autoResize: false,
      pageFormat: "A4_P",
      cursorsOptions: [{ type: 0, color: "#CCCC00", alpha: 0.6, follow: true }],
      pageBackgroundColor: "#FFFFFF",
    });
    console.debug("初始化五线谱渲染器");
    // 初始化简谱渲染器
    this.display_jianpu = new Jianpu(this.score_selector);
    console.debug("初始化简谱渲染器");
  }
  
  async load_music_xml_staff(music_xml) {
    this.stop_score();
    this.music_xml = music_xml;
    this.display_staff.load(music_xml)
    .then(() => {
      this.display_staff.render();
    });
  }
  
  async load_music_xml_jianpu(music_xml) {
    this.stop_score();
    this.music_xml = music_xml;
    this.display_jianpu.load(music_xml)
    .then(() => {
      this.display_jianpu.render();
    });
  }
  
  async load_music_xml(music_xml) {
    if(this.show_staff) {
      await this.load_music_xml_staff(music_xml);
    } else {
      await this.load_music_xml_jianpu(music_xml);
    }
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
    if(!this.show_staff) {
      alert("该功能只支持五线谱谱面");
      return;
    }
    const note_list = [];
    this.display_staff.cursor.reset();
    const iterator = this.display_staff.cursor.Iterator;
    while (!iterator.EndReached) {
      let bpm = 4;
      if(this.display_staff.sheet.hasBPMInfo) {
        bpm = (1.0 * this.display_staff.sheet.defaultStartTempoInBpm / 60).toFixed(2);
      }
      const voices = iterator.CurrentVoiceEntries;
      for (var i = 0; i < voices.length; i++) {
        const v = voices[i];
        const notes = v.Notes;
        for (var j = 0; j < notes.length; j++) {
          const note = notes[j];
          if (note != null) {
            note_list.push({
              "id"  : note.parentStaffEntry.parentStaff.idInMusicSheet,
              "note": note.halfTone - 12 + 4,
              "time": iterator.currentTimeStamp.RealValue * bpm,
              "rest": note.isRest()
            })
          }
        }
      }
      iterator.moveToNext()
    }
    console.debug("音符列表", note_list);
    this.display_staff.cursor.reset();
    this.display_staff.cursor.show();
    this.player.play_list(note_list, () => {
      this.display_staff.cursor.next();
    }, () => {
      this.display_staff.cursor.hide();
      play_ended();
    });
  }

  async stop_score() {
    this.player.stop_play();
  }

  async swap_score() {
    this.show_staff = !this.show_staff;
    document.querySelector(this.score_selector).innerHTML = "";
    if(this.show_staff) {
      this.load_music_xml_staff(this.music_xml);
    } else {
      this.load_music_xml_jianpu(this.music_xml);
    }
  }
  
  async tone_score() {
    if(this.show_staff) {
      alert("该功能只支持简谱谱面");
      return;
    }
    this.display_jianpu.tone();
  }
  
  async save_pdf_staff() {
    const backends = this.display_staff.drawer.Backends;
    let svgElement = backends[0].getSvgElement();
    let pageWidth  = this.a4_width;
    let pageHeight = this.a4_height;
    if (!this.display_staff.rules.PageFormat?.IsUndefined) {
      pageWidth  = this.display_staff.rules.PageFormat.width;
      pageHeight = this.display_staff.rules.PageFormat.height;
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
    pdf.save((this.display_staff.sheet.FullNameString || "lifuren") + ".pdf");
  };
  
  async save_pdf_jianpu() {
  };
  
  async save_pdf() {
    if(this.show_staff) {
      await this.save_pdf_staff();
    } else {
      await this.save_pdf_jianpu();
    }
  }
  
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
      dlLink.download = (this.display_staff.sheet.FullNameString || "lifuren") + "-" + index + ".png";
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
  
  async save_img_staff() {
    const backends = this.display_staff.drawer.Backends;
    for(let i = 0; i < backends.length; ++i) {
      await this.download_img(i, backends[i].getSvgElement());
    }
  };
  
  async save_img_jianpu() {
  };
  
  async save_img() {
    if(this.show_staff) {
      await this.save_img_staff();
    } else {
      await this.save_img_jianpu();
    }
  }
  
  async register_audio(id, type, audio) {
    if(this.player) {
      this.player.register(id, type, audio);
    }
  }

};