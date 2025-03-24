/**
 * 李夫人
 */
class Lifuren {

  zoom       = 1.0;   // 谱面缩放
  show_staff = true;  // 是否显示五线谱
  show_piano = false; // 是否显示钢琴键盘
  
  player         = null; // 播放器
  music_xml      = null; // 乐谱内容
  display_staff  = null; // 五线谱渲染器
  display_jianpu = null; // 简谱渲染器
  
  a4_width  = 210; // A4宽度
  a4_height = 297; // A4高度

  score_selector      = ""; // 谱面选择器
  piano_selector      = ""; // 钢琴选择器
  piano_keys_selector = ""; // 琴键选择器
  
  constructor(
    score_selector      = "#score_container",
    piano_selector      = "#piano_player",
    piano_keys_selector = "#piano_player .key"
  ) {
    this.score_selector      = score_selector;
    this.piano_selector      = piano_selector;
    this.piano_keys_selector = piano_keys_selector;
    // 初始化播放器
    this.player = new Player(piano_keys_selector);
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
    this.music_xml = music_xml;
    this.display_staff.load(music_xml)
    .then(() => {
      this.display_staff.render();
    });
  }
  
  async load_music_xml_jianpu(music_xml) {
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
        accept: { "text/xml": ['.xml', '.musicxml'] },
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
  
  async play_score() {
    if(!this.show_staff) {
      alert("只有五线谱支持播放");
      return;
    }
    const allNotes = [];
    this.display_staff.cursor.reset();
    const iterator = this.display_staff.cursor.Iterator;
    while (!iterator.EndReached) {
      const voices = iterator.CurrentVoiceEntries;
      for (var i = 0; i < voices.length; i++) {
        const v = voices[i];
        const notes = v.Notes;
        for (var j = 0; j < notes.length; j++) {
          const note = notes[j];
          if (note != null && note.halfTone != 0 && !note.isRest()) {
            allNotes.push({
              "id"  : note.parentStaffEntry.parentStaff.idInMusicSheet,
              "note": note.halfTone ,
              "time": iterator.currentTimeStamp.RealValue * 4
            })
          }
        }
      }
      iterator.moveToNext()
    }
    console.info(allNotes);
    this.display_staff.cursor.reset();
    this.display_staff.cursor.show();
    this.player.playList(allNotes, () => {
      this.display_staff.cursor.next();
      // this.display_staff.cursor.hide();
      // this.display_staff.cursor.reset();
      // this.display_staff.cursor.previous();
    });
  }

  async stop_score() {
    this.player.stopPlay();
  }

  async swap_score() {
    this.show_staff = !this.show_staff;
    document.querySelector(this.score_selector).innerHTML  = "";
    if(this.show_staff) {
      this.load_music_xml_staff(this.music_xml);
    } else {
      this.load_music_xml_jianpu(this.music_xml);
    }
  }
  
  async rule_score() {
    if(!this.show_staff) {
      alert("只有五线谱支持指法");
      return;
    }
    if(this.show_piano) {
      document.querySelector(this.piano_selector).style = "display:none;";
    } else {
      document.querySelector(this.piano_selector).style = "display:block;";
    }
    this.show_piano = !this.show_piano;
  }
  
  async tone_score() {
    if(this.show_staff) {
      alert("只有简谱支持移调");
      return;
    }
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
    const canvas     = document.createElement('canvas');
    canvas.width     = svgElement.width.baseVal.value;
    canvas.height    = svgElement.height.baseVal.value;
    const ctx        = canvas.getContext('2d');
    const svgContent = new XMLSerializer().serializeToString(svgElement);
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0);
      const imgURL = canvas.toDataURL({ format: "image/png" });
      const dlLink = document.createElement('a');
      dlLink.href     = imgURL;
      dlLink.download = (this.display_staff.sheet.FullNameString || "lifuren") + "-" + index + ".png";
      dlLink.dataset.downloadurl = ['image/png', dlLink.download, dlLink.href].join(':');
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
      content += String.fromCharCode.apply(...array.slice(i * chunk));
    }
    img.src = 'data:image/svg+xml;base64,' + btoa(content);
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
  
  async zoom_in_staff() {
    this.display_staff.Zoom = this.zoom += 0.1;
    this.display_staff.render();
  };
  
  async zoom_in_jianput() {
  };
  
  async zoom_in() {
    if(this.show_staff) {
      await this.zoom_in_staff();
    } else {
      await this.zoom_in_jianput();
    }
  }
  
  async zoom_out_staff() {
    this.display_staff.Zoom = this.zoom -= 0.1;
    this.display_staff.render();
  };
  
  async zoom_out_jianput() {
  };
  
  async zoom_out() {
    if(this.show_staff) {
      await this.zoom_out_staff();
    } else {
      await this.zoom_out_jianput();
    }
  }
  
  async zoom_reset_staff() {
    this.display_staff.Zoom = this.zoom = 1.0;
    this.display_staff.render();
  };
  
  async zoom_reset_jianput() {
  };
  
  async zoom_reset() {
    if(this.show_staff) {
      await this.zoom_reset_staff();
    } else {
      await this.zoom_reset_jianput();
    }
  }
  
  async register_audio(id, type, audio) {
    if(this.player) {
      this.player.register(id, type, audio);
    }
  }

};