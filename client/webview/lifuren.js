/**
 * 李夫人
 */
let zoom  = 1.0;  // 缩放
let staff = true; // 五线谱

let player;
let display_staff;
let display_jianpu;
let music_xml_cache;

const a4_width  = 210;
const a4_height = 297;

async function load_music_xml(music_xml) {
  if(staff) {
    await load_music_xml_staff(music_xml);
  } else {
    await load_music_xml_jianpu(music_xml);
  }
}

async function load_music_xml_staff(music_xml) {
  music_xml_cache = music_xml;
  display_staff = new opensheetmusicdisplay.OpenSheetMusicDisplay("staff_container");
  display_staff.setOptions({
    backend   : "svg",
    drawTitle : true,
    autoResize: false,
    pageFormat: "A4_P",
    cursorsOptions: [{ type: 3, color: "#CCCC00", alpha: 0.6, follow: true }],
    pageBackgroundColor: "#FFFFFF",
  });
  display_staff.load(music_xml).then(() => {
    display_staff.render();
    // display_staff.cursor.show();
    // display_staff.cursor.hide();
    // display_staff.cursor.reset();
    // display_staff.cursor.previous();
    // display_staff.cursor.next();
  });
}

async function load_music_xml_jianpu(music_xml) {
  music_xml_cache = music_xml;
  display_jianpu = new Jianpu();
  display_jianpu.render(music_xml);
}

async function load_music_xml_file() {
  const [picker] = await window.showOpenFilePicker({
    types: [{
      accept: { "text/xml": ['.xml', '.musicxml'] },
      description: "乐谱"
    }]
  });
  const file = await picker?.getFile();
  if(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      load_music_xml(e.target.result);
    };
    reader.readAsText(file);
  } else {
    console.info("用户没有选择文件");
  }
};

async function show_score() {
  staff = !staff;
  if(staff) {
    document.querySelector("#staff_container").style  = "display:block;";
    document.querySelector("#jianpu_container").style = "display:none;";
    load_music_xml_staff(music_xml_cache);
  } else {
    document.querySelector("#staff_container").style  = "display:none;";
    document.querySelector("#jianpu_container").style = "display:block;";
    load_music_xml_jianpu(music_xml_cache);
  }
}

async function save_pdf_staff() {
  const backends = display_staff.drawer.Backends;
  let svgElement = backends[0].getSvgElement();
  let pageWidth  = a4_width;
  let pageHeight = a4_height;
  if (!display_staff.rules.PageFormat?.IsUndefined) {
    pageWidth  = display_staff.rules.PageFormat.width;
    pageHeight = display_staff.rules.PageFormat.height;
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
  pdf.save((display_staff.sheet.FullNameString || "lifuren") + ".pdf");
};

async function save_pdf_jianpu() {
};

async function download_img(index, svgElement) {
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
    dlLink.download = (display_staff.sheet.FullNameString || "lifuren") + "-" + index + ".png";
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

async function save_img_staff() {
  const backends = display_staff.drawer.Backends;
  for(let i = 0; i < backends.length; ++i) {
    await download_img(i, backends[i].getSvgElement());
  }
};

async function save_img_jianpu() {
};

async function zoom_in_staff() {
  display_staff.Zoom = zoom += 0.1;
  display_staff.render();
};

async function zoom_in_jianput() {
};

async function zoom_out_staff() {
  display_staff.Zoom = zoom -= 0.1;
  display_staff.render();
};

async function zoom_out_jianput() {
};

async function zoom_reset_staff() {
  display_staff.Zoom = zoom = 1.0;
  display_staff.render();
};

async function zoom_reset_jianput() {
};

function init_lifuren(music_xml) {
  document.querySelector("#open_score").onclick = async () => {
    await load_music_xml_file();
  };
  document.querySelector("#play_score").onclick = async () => {
  };
  document.querySelector("#rule_score").onclick = async () => {
  };
  document.querySelector("#show_score").onclick = async () => {
    await show_score();
  };
  document.querySelector("#tone_score").onclick = async () => {
  };
  document.querySelector("#save_pdf").onclick = async () => {
    if(staff) {
      await save_pdf_staff();
    } else {
      await save_pdf_jianpu();
    }
  };
  document.querySelector("#save_img").onclick = async () => {
    if(staff) {
      await save_img_staff();
    } else {
      await save_img_jianpu();
    }
  };
  document.querySelector("#zoom_in").onclick = async () => {
    if(staff) {
      await zoom_in_staff();
    } else {
      await zoom_in_jianput();
    }
  };
  document.querySelector("#zoom_out").onclick = async () => {
    if(staff) {
      await zoom_out_staff();
    } else {
      await zoom_out_jianput();
    }
  };
  document.querySelector("#zoom_reset").onclick = async () => {
    if(staff) {
      await zoom_reset_staff();
    } else {
      await zoom_reset_jianput();
    }
  };
  if(music_xml) {
    load_music_xml(music_xml);
  }
  if(player) {
  } else {
    player = new Player();
  }
  player.listen(".key");
}
