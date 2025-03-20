/**
 * 乐谱
 * 
 * https://github.com/nkufree/xml2jianpu
 * https://github.com/opensheetmusicdisplay/opensheetmusicdisplay/blob/develop/demo/index.js
 */

let zoom  = 1.0;  // 缩放
let staff = true; // 五线谱

let display;
let music_xml_cache;

async function load_music_xml(music_xml) {
  if(staff) {
    await load_music_xml_staff(music_xml);
  } else {
    await load_music_xml_jianpu(music_xml);
  }
}

async function load_music_xml_staff(music_xml) {
  music_xml_cache = music_xml;
  display = new opensheetmusicdisplay.OpenSheetMusicDisplay("staff_container");
  display.setOptions({
    backend   : "svg", // "canvas"
    drawTitle : true,
    autoResize: false,
    pageFormat: "A4_P",
    cursorsOptions: [{ type: 3, color: "#CCCC00", alpha: 0.6, follow: true }],
    pageBackgroundColor: "#FFFFFF",
  });
  display.load(music_xml).then(() => {
    display.render();
    // display.cursor.show();
    // display.cursor.hide();
    // display.cursor.reset();
    // display.cursor.previous();
    // display.cursor.next();
  });
}

async function load_music_xml_jianpu(music_xml) {
  music_xml_cache = music_xml;
  display = new Jianpu();
  display.render(music_xml);
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

async function staff_jianpu() {
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
  const backends = display.drawer.Backends;
  let svgElement = backends[0].getSvgElement();
  let pageWidth  = 210;
  let pageHeight = 297;
  const displayPageFormat = display.rules.PageFormat;
  if (displayPageFormat && !displayPageFormat.IsUndefined) {
    pageWidth  = displayPageFormat.width;
    pageHeight = displayPageFormat.height;
  } else {
    pageHeight = pageWidth * svgElement.clientHeight / svgElement.clientWidth;
  }
  const orientation = pageHeight > pageWidth ? "p" : "l";
  const pdf = new jspdf.jsPDF({
    unit  : "mm",
    format: [pageWidth, pageHeight],
    orientation: orientation
  });
  for (let index = 0; index < backends.length; ++index) {
    if (index > 0) {
      pdf.addPage();
    }
    svgElement = backends[index].getSvgElement();
    await pdf.svg(svgElement, {
      x: 0,
      y: 0,
      width: pageWidth,
      height: pageHeight,
    })
  }
  pdf.save((display.sheet.FullNameString || "lifuren") + ".pdf");
};

async function save_pdf_jianpu() {
};

async function save_img_staff() {
  display.setOptions({
    backend   : "canvas",
    pageFormat: ""
  });
  display.render();
  const canvas = document.querySelector("#staff_container canvas");
  const imgURL = canvas.toDataURL({ format: "image/png" });
  const dlLink = document.createElement('a');
  dlLink.href     = imgURL;
  dlLink.download = (display.sheet.FullNameString || "lifuren") + ".png";
  dlLink.dataset.downloadurl = ['image/png', dlLink.download, dlLink.href].join(':');
  document.body.appendChild(dlLink);
  dlLink.click();
  document.body.removeChild(dlLink);
  display.setOptions({
    backend   : "svg",
    pageFormat: "A4_P"
  });
  display.render();
};

async function save_img_jianpu() {
// var svgElement = document.getElementsByTagName('svg')[0];
// var svgContent = new XMLSerializer().serializeToString(svgElement);
// var canvas = document.createElement('canvas');
// canvas.width = svgElement.width.baseVal.value;
// canvas.height = svgElement.height.baseVal.value;
// var ctx = canvas.getContext('2d');
// var img = new Image();
// img.onload = function() {
//   ctx.drawImage(img, 0, 0);
// };
// img.src = 'data:image/svg+xml;base64,' + btoa(svgContent);
};

async function zoom_in_staff() {
  display.Zoom = zoom += 0.1;
  display.render();
};

async function zoom_in_jianput() {
};

async function zoom_out_staff() {
  display.Zoom = zoom -= 0.1;
  display.render();
};

async function zoom_out_jianput() {
};

async function zoom_reset_staff() {
  display.Zoom = zoom = 1.0;
  display.render();
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
  document.querySelector("#staff_jianpu").onclick = async () => {
    await staff_jianpu();
  };
  document.querySelector("#plus_score").onclick = async () => {
  };
  document.querySelector("#fall_score").onclick = async () => {
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
}
