/**
 * 简谱
 */
class Jianpu {
  
  zoom  = 1.0;  // 缩放
  music = null; // 乐谱

  a4_width  = 210; // A4宽度
  a4_height = 297; // A4高度

  score_selector = ""; // 谱面选择器

  page_svg  = new Map(); // 页码映射
  page_size = 0;         // 总的页数

  margin_top  = 20; // 上边距
  margin_left = 20; // 左边距
  x = 0; // 当前X
  y = 0; // 当前Y
  width  = 0; // 宽度
  height = 0; // 高度
  note_row_spacing = 20; // 行间距
  note_col_spacing = 20; // 列间距

  // 0=C/a
  // 负数降调：1=F/d 2=bB/g 3=bE/c 4=bA/f 5=bD/bb 6=bG/be 7=bC/ba
  // 正数升调：1=G/e 2=D/b  3=A/#f 4=E/#c 5=B/#g  6=#F/#d 7=#C/#a
  fifths_mapping = {
    '-1': 'F',
    '-2': 'bB',
    '-3': 'bE',
    '-4': 'bA',
    '-5': 'bD',
    '-6': 'bG',
    '-7': 'bC',
    '1' : 'G',
    '2' : 'D',
    '3' : 'A',
    '4' : 'E',
    '5' : 'B',
    '6' : '#F',
    '7' : '#C',
  };

  constructor(score_selector = "#score_container") {
    this.score_selector = score_selector;
  }

  load(music_xml) {
    const parser = new fxp.XMLParser();
    try {
      this.music = parser.parse(music_xml);
      console.debug("乐谱内容", this.music);
      this.page_size = 1;
      this.page_svg.clear();
      d3.select(this.score_selector).html("");
    } catch (error) {
      console.error("解析musicXML异常", music_xml, error);
      return new Promise((resolve, reject) => {
        resolve();
      });
    }
    return new Promise((resolve, reject) => {
      resolve();
    });
  }

  tone() {
    // TODO
    // 五声调式：CDEGA（宫、商、角、徵、羽）
    // 自然调式：CDEFGAB
  }

  render_page(g, svg) {
      // TODO 每个开始位置:per-minute
    // tone.append("tspan")
    //   .attr("dx", this.zoom * 20)
    //   .text("♩ = 187")
  }

  render_credit(g, svg, credits, measures) {
    if(!credits) {
      return;
    }
    // 标题
    for(const credit of credits) {
      if(credit['credit-type'] === "title") {
        const node = g.append("text")
          .attr("transform",   `translate(${this.width / 2}, ${this.y + this.zoom * 30})`)
          .attr("font-size",   this.zoom * 30)
          .attr("font-weight", "bold")
          .attr("text-anchor", "middle")
          .text(credit['credit-words']);
          this.y += node.node().scrollHeight;
        }
    }
    // 副标题
    for(const credit of credits) {
      if(credit['credit-type'] === "subtitle") {
        const node = g.append("text")
          .attr("transform",   `translate(${this.width / 2}, ${this.y + this.zoom * 20})`)
          .attr("font-size",   this.zoom * 20)
          .attr("text-anchor", "middle")
          .text(credit['credit-words']);
        this.y += node.node().scrollHeight;
      }
    }
    let left_height  = 0;
    let right_height = 0;
    // 调式 节奏
    const measure    = measures[0];
    const attributes = measure?.attributes;
    if(attributes) {
      // 调式
      const fifths = attributes.key?.fifths;
      const node = g.append("text")
        .attr("transform", `translate(${this.margin_left}, ${this.y + this.zoom * 16})`)
        .attr("font-size", this.zoom * 16);
      if(fifths && fifths !== 0) {
        const value = this.fifths_mapping[fifths + ""];
        if(value) {
          node.append("tspan")
            .text("1=")
          if(value.length == 2) {
            node.append("tspan")
              .attr("font-size", this.zoom * 14)
              .attr("baseline-shift", "super")
              .text(value.substring(0, 1))
              node.append("tspan")
              .text(value.substring(1))
          } else {
            node.append("tspan")
              .text(value)
          }
        }
      } else {
        node.append("tspan").text("1=C");
      }
      // 节奏
      const time = attributes.time;
      if(time) {
        const beats    = time["beats"];
        const beatType = time["beat-type"];
        if(beats && beatType) {
          node.append("tspan")
            .attr("dx", this.zoom * 10)
            .text(`${beats}/${beatType}`)
        }
      }
      left_height += node.node().scrollHeight;
    }
    for(const credit of credits) {
      if(credit['credit-type'] === "composer") {
        const words = credit['credit-words'];
        if(Array.isArray(words)) {
          for(const word of words) {
            console.info(word)
            const node = g.append("text")
              .attr("transform", `translate(${this.width / 4 * 3}, ${this.y + right_height + this.zoom * 16})`)
              .attr("font-size", this.zoom * 16)
              .text(word);
            right_height += node.node().scrollHeight;
          }
        } else {
          const node = g.append("text")
            .attr("transform", `translate(${this.width / 4 * 3}, ${this.y + right_height + this.zoom * 16})`)
            .attr("font-size", this.zoom * 16)
            .text(credit['credit-words']);
          right_height += node.node().scrollHeight;
        }
      }
    }
    this.y += Math.max(left_height, right_height);
  }

  render() {
    const container = document.querySelector(this.score_selector);
    this.width  = container.offsetWidth;
    this.height = (1.0 * this.width / this.a4_width * this.a4_height).toFixed(0);
    const root = this.music["score-partwise"];
    if(!root || !root.part) {
      console.warn("无效乐谱");
      return;
    }
    
    // this.render_old(root.part?.measure);
    // return;

    const measures = [];
    if(Array.isArray(root.part)) {
      for(const part of root.part) {
        measures.push(part.measure);
      }
    } else {
      measures.push(root.part.measure);
    }
    this.y = this.margin_top;
    this.x = this.margin_left;
    const svg = d3.select(this.score_selector)
      .append("svg");
    this.page_svg.set(this.page_size++, svg);
    const g = svg
      .attr("width",  this.width)
      .attr("height", this.height)
      .append("g");
    this.render_credit(g, svg, root.credit, measures);
    this.render_page(g, svg);

  }

  render_old(measures) {
    const container = document.querySelector(this.score_selector);
    const width  = container.offsetWidth;
    const height = (1.0 * width / this.a4_width * this.a4_height).toFixed(0);
    const svg = d3.select(this.score_selector)
      .append("svg");
    this.page_svg.set(this.page_size++, svg);
    const g = svg
      .attr("width",  width)
      .attr("height", height)
      .append("g");
      
    var partAttr = measures[0].attributes;
    //绘制小节音符
    var start = 0;//该小节前的小节的位置
    var length = 0;//该小节的长度
    var lineIndex = 0;//该小节所在行
    var eachHeight = 70;//每个小节的高度
    var marginLeft = 100;//左边距
    var marginTop = 100;//上边距
    var tiePath = [-1, -1, -1, -1];//连音始末位置
    var eighthPath = 0; //八分音符下划线的始末位置
    var sixteenthPath = 0;//16分音符下划线的始末位置
    var titleTop = 30;
    var initSpacing = 18;
    var noteSpacing = 20;

    var noteCount = [];
    var eachNoteCount = 0;
    for (let j = 0; j < measures.length; j++) {
      if (measures[j].print) {
        noteCount.push(eachNoteCount);
        eachNoteCount = 0;
      }

      if (measures[j].note.length == undefined) {
        measures[j].note = [measures[j].note];
      }
      //选择各小节音符
      g.selectAll(".note")
        .data(measures[j].note)
        .enter()
        .each(function (d, i, n) {
          eachNoteCount++;
          var divisions = partAttr.divisions;
          if (d.duration > divisions) {
            let addNote = Math.floor(d.duration / divisions);
            eachNoteCount += addNote - 1;
          }
        })
      eachNoteCount++;

    }
    console.info(noteCount)

    var maxLength = d3.max(noteCount);
    var totalWidth = maxLength * initSpacing;
    noteCount.push(eachNoteCount);
    marginLeft = (width - totalWidth) / 2;
    svg.attr("height", marginTop + noteCount.length * eachHeight);
    g.append("text")
      .attr("transform", `translate(${marginLeft + totalWidth / 2 - 20},${titleTop + 30})`)
      .attr("font-weight", "bold")
      .attr("text-anchor", "middle")
      .attr("font-size", 30)
      .text("三色绘恋");
    var textD = g.append("text")
      .attr("transform", `translate(${marginLeft},${titleTop + 60})`)
      .attr("font-size", 18);
    textD.append("tspan")
      .text("1=")
    textD.append("tspan")
      .attr("baseline-shift", "super")
      .attr("font-size", 15)
      .text("b")
    textD.append("tspan")
      .text("D")
    textD.append("tspan")
      .attr("dx", 20)
      .text("4/4")
    g.append("text")
      .attr("transform", `translate(${marginLeft},${titleTop + 90})`)
      .attr("font-size", 18)
      .text("BPM = 187");
    g.append("text")
      .attr("transform", `translate(${marginLeft + maxLength * initSpacing - 150},${titleTop + 60})`)
      .attr("font-size", 15)
      .text("作词：吴柳");
    g.append("text")
      .attr("transform", `translate(${marginLeft + maxLength * initSpacing - 150},${titleTop + 80})`)
      .attr("font-size", 15)
      .text("作编曲：丸山公詳");
    g.append("text")
      .attr("transform", `translate(${marginLeft + maxLength * initSpacing - 150},${titleTop + 100})`)
      .attr("font-size", 15)
      .text("歌：雲翼星辰");


    for (var j = 0; j < measures.length; j++) {
      var dx = 0;//有全音符时位置偏移
      var reset = 0;//有全音符时位置修正
      if (measures[j].print) {
        start = 0;
        lineIndex++;
        noteSpacing = initSpacing * maxLength / noteCount[lineIndex];
        //console.log(noteSpacing);
      }
      else
        start = start + length * noteSpacing + noteSpacing;
      //console.log(j);
      //console.log(measures[j].note.length);
      length = measures[j].note.length;

      //处理只有一个音符的小节
      if (length == undefined) {
        measures[j].note = [measures[j].note];
        length = 1;
      }
      //选择各小节音符
      g.selectAll(".note")
        .data(measures[j].note)
        .enter()
        .each(function (d, i, n) {
          //console.log(n[0].__data__);
          var number = note2number(d);
          var divisions = partAttr.divisions;
          var dy = 0;//绘制下划线和点时的位置偏移
          var ddy = 0//绘制上方点和线时位置偏移
          //console.log(number);
          //绘制音符
          var noteNumberIs = d3.select(this)
            .append("text")
            .attr("text-anchor", "middle")
            .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing},${marginTop + lineIndex * eachHeight})`);
          if (number.text.length == 1)
            noteNumberIs.text(number.text);
          else {
            noteNumberIs.append("tspan")
              .attr("baseline-shift", "super")
              .attr("dy", () => {
                if (number.text[0] == "#")
                  return 8;
                else
                  return 4;
              })
              .attr("font-size", 12)
              .attr("dx", -5)
              .text(number.text[0]);
            noteNumberIs.append("tspan")
              .attr("dy", () => {
                if (number.text[0] == "#")
                  return -8;
                else
                  return -4;
              })
              .text(number.text[1]);
          }
          //绘制附点
          if (d.dot != undefined && number.duration < 2 * divisions) {
            d3.select(this)
              .append("text")
              .attr("text-anchor", "left")
              .attr("font-weight", "bold")
              .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing + 5},${marginTop + lineIndex * eachHeight})`)
              .text("·");
          }
          //绘制歌词
          if (d.lyric != undefined) {
            let text = "";
            if (d.lyric.length > 1) {
              for (let i = 0; i < d.lyric.length; i++) {
                text += d.lyric[i].text;
              }
            }
            d3.select(this)
              .append("text")
              .attr("text-anchor", "middle")
              .attr("font-family", "SimSun")
              .attr("font-weight", "600")
              .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing},${marginTop + lineIndex * eachHeight + 35})`)
              .text(d.lyric.text || text);
          }
          //绘制下划线
          let durList = d3.map(n, d => note2number(d.__data__).dur);
          let octList = d3.map(n, d => note2number(d.__data__).octave);
          if (number.dur == divisions / 2) {

            d3.select(this)
              .append("line")
              .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing},${marginTop + lineIndex * eachHeight})`)
              .attr("x1", -5)
              .attr("y1", 5)
              .attr("x2", 5)
              .attr("y2", 5)
              .attr("stroke", "black")
              .attr("stroke-width", "1px");

            if (d.beam != undefined) {
              if (d.beam == "begin")
                eighthPath = -noteSpacing;
              else if (d.beam == "continue")
                eighthPath -= noteSpacing;
              else if (d.beam == "end") {
                d3.select(this)
                  .append("line")
                  .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing},${marginTop + lineIndex * eachHeight})`)
                  .attr("x1", 0)
                  .attr("y1", 5)
                  .attr("x2", eighthPath)
                  .attr("y2", 5)
                  .attr("stroke", "black")
                  .attr("stroke-width", "1px");
                eighthPath = 0;
              }
            }
            else {
              durList.push(0);
              //console.log(durList);
              if ((i == 0 && durList[i] == durList[i + 1]) || (i != 0 && durList[i] != durList[i - 1] && durList[i + 1] == durList[i])) {
                d3.select(this)
                  .append("line")
                  .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing},${marginTop + lineIndex * eachHeight})`)
                  .attr("x1", 0)
                  .attr("y1", 5)
                  .attr("x2", noteSpacing)
                  .attr("y2", 5)
                  .attr("stroke", "black")
                  .attr("stroke-width", "1px");
              }
            }

            dy = 5;
          }
          else if (number.dur == divisions / 4) {
            d3.select(this)
              .append("line")
              .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing},${marginTop + lineIndex * eachHeight})`)
              .attr("x1", -5)
              .attr("y1", 5)
              .attr("x2", 5)
              .attr("y2", 5)
              .attr("stroke", "black")
              .attr("stroke-width", "1px")
            d3.select(this)
              .append("line")
              .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing},${marginTop + lineIndex * eachHeight})`)
              .attr("x1", -5)
              .attr("y1", 8)
              .attr("x2", 5)
              .attr("y2", 8)
              .attr("stroke", "black")
              .attr("stroke-width", "1px");
            if (d.beam != undefined) {
              if (d.beam[0] == "begin")
                eighthPath = -noteSpacing * number.text.length;
              else if (d.beam[0] == "continue")
                eighthPath -= noteSpacing * number.text.length;
              else if (d.beam[0] == "end") {
                d3.select(this)
                  .append("line")
                  .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing},${marginTop + lineIndex * eachHeight})`)
                  .attr("x1", 0)
                  .attr("y1", 5)
                  .attr("x2", eighthPath)
                  .attr("y2", 5)
                  .attr("stroke", "black")
                  .attr("stroke-width", "1px");
                eighthPath = 0;
              }
              if (d.beam[1] == "begin")
                sixteenthPath = -noteSpacing;
              else if (d.beam[1] == "continue")
                sixteenthPath -= noteSpacing;
              else if (d.beam[1] == "end") {
                d3.select(this)
                  .append("line")
                  .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing},${marginTop + lineIndex * eachHeight})`)
                  .attr("x1", 0)
                  .attr("y1", 8)
                  .attr("x2", eighthPath)
                  .attr("y2", 8)
                  .attr("stroke", "black")
                  .attr("stroke-width", "1px");
                sixteenthPath = 0;
              }
            }
            else {
              if ((i == 0 && durList[i] == durList[i + 1]) || (i != 0 && durList[i] != durList[i - 1] && durList[i + 1] == durList[i])) {
                d3.select(this)
                  .append("line")
                  .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing},${marginTop + lineIndex * eachHeight})`)
                  .attr("x1", 0)
                  .attr("y1", 5)
                  .attr("x2", noteSpacing)
                  .attr("y2", 5)
                  .attr("stroke", "black")
                  .attr("stroke-width", "1px");
                d3.select(this)
                  .append("line")
                  .attr("transform", `translate(${marginLeft + start + (i + dx) * noteSpacing},${marginTop + lineIndex * eachHeight})`)
                  .attr("x1", 0)
                  .attr("y1", 8)
                  .attr("x2", noteSpacing)
                  .attr("y2", 8)
                  .attr("stroke", "black")
                  .attr("stroke-width", "1px");
              }
            }
            dy = 8;
          }
          else if (number.dur > divisions) {
            let addNote = Math.floor(number.dur / divisions);
            for (let k = 1; k < addNote; k++) {
              d3.select(this)
                .append("text")
                .attr("transform", `translate(${marginLeft + start + (i + dx + k) * noteSpacing},${marginTop + lineIndex * eachHeight})`)
                .attr("font-weight", () => {
                  if (number.text == "0")
                    return "normal";
                  else
                    return "bold";
                })
                .attr("text-anchor", "middle")
                .text(() => {
                  if (number.text == "0")
                    return "0";
                  else
                    return "-";
                });
              length++;
            }
            dx += addNote - 1;
          }

          if (number.dur > divisions) reset = i;
          else reset = i + dx;
          //绘制点
          if (number.octave == 3) {
            d3.select(this)
              .append("circle")
              .attr("transform", `translate(${marginLeft + start + reset * noteSpacing},${marginTop + lineIndex * eachHeight})`)
              .attr("cx", 0)
              .attr("cy", 5 + dy)
              .attr("r", 1.5)
              .attr("fill", "black");
            dy += 5;
          }
          else if (number.octave == 5) {
            d3.select(this)
              .append("circle")
              .attr("transform", `translate(${marginLeft + start + reset * noteSpacing},${marginTop + lineIndex * eachHeight})`)
              .attr("cx", 0)
              .attr("cy", -18)
              .attr("r", 1.5)
              .attr("fill", "black");
            ddy += 5;
          }
          //绘制连音的曲线
          if (number.tied) {

            if (tiePath[0] == -1) {
              tiePath[0] = marginLeft + start + reset * noteSpacing;
              tiePath[1] = marginTop + lineIndex * eachHeight - (ddy + 17);
            }
            else if (tiePath[2] == -1) {
              tiePath[2] = marginLeft + start + reset * noteSpacing;
              tiePath[3] = marginTop + lineIndex * eachHeight - (ddy + 17);
              //如果两个音符在一行，绘制一条曲线；如果在两行，绘制两条曲线。
              if (Math.abs(tiePath[3] - tiePath[1]) < 20) {
                d3.select(this)
                  .append("path")
                  .attr("fill", "none")
                  .attr("stroke", "black")
                  .attr("stroke-width", 1)
                  .attr("d", pathTied(tiePath));
                tiePath[0] = -1;
                tiePath[2] = -1;
              }
              else if (Math.abs(tiePath[3] - tiePath[1]) > 20) {
                let path1 = [tiePath[0], tiePath[1], tiePath[0] + noteSpacing, tiePath[1]];
                let path2 = [tiePath[2] - 10, tiePath[3], tiePath[2], tiePath[3]];
                d3.select(this)
                  .append("path")
                  .attr("fill", "none")
                  .attr("stroke", "black")
                  .attr("stroke-width", 1)
                  .attr("d", pathTied(path1));
                d3.select(this)
                  .append("path")
                  .attr("fill", "none")
                  .attr("stroke", "black")
                  .attr("stroke-width", 1)
                  .attr("d", pathTied(path2));
                tiePath[0] = -1;
                tiePath[2] = -1;
              }
            }
          }
          //绘制三连音的标志
          if (number.dur == divisions / 3 || number.dur == divisions * 2 / 3) {
            if (i != 0 && i + 1 < length && durList[i - 1] == number.dur && number.dur == durList[i + 1]) {
              if (octList[i - 1] == 5 || octList[i] == 5 || octList[i + 1] == 5)
                ddy = 5;
              d3.select(this)
                .append("path")
                .attr("fill", "none")
                .attr("stroke", "black")
                .attr("stroke-width", "1px")
                .attr("transform", `translate(${marginLeft + start + reset * noteSpacing},${marginTop + lineIndex * eachHeight})`)
                .attr("d", `M ${-noteSpacing} ${-(16 + ddy)} L ${-noteSpacing} ${-(19 + ddy)} L ${noteSpacing} ${-(19 + ddy)} L ${noteSpacing} ${-(16 + ddy)}`);
              d3.select(this)
                .append("text")
                .attr("font-size", 10)
                .attr("text-anchor", "middle")
                .attr("x", 0)
                .attr("y", -(16 + ddy))
                .attr("transform", `translate(${marginLeft + start + reset * noteSpacing},${marginTop + lineIndex * eachHeight})`)
                .text("3");
            }
          }
        })

      //绘制小节线
      g.append("text")
        .attr("transform", (d, i) => `translate(${marginLeft + start + length * noteSpacing},${marginTop + lineIndex * eachHeight})`)
        .attr("text-anchor", "middle")
        .text("|")
    }
    function pathTied(p) {
      if (p[1] > p[3] && p[1] - p[3] < 20)
        p[3] = p[1];
      else if (p[3] > p[1] && p[3] - p[1] < 20)
        p[1] = p[3];
      var dx = p[2] - p[0];
      return `M ${p[0]} ${p[1]} C ${p[0] + dx / 4} ${p[1] - 4} ${p[2] - dx / 4} ${p[1] - 4} ${p[2]} ${p[1]}`;
    }
    function note2number(note) {
      var keyAlter = [{ fifth: 7, key: "#C", alter: -1 },
      { fifth: 0, key: "C", alter: 0 },
      { fifth: -7, key: "bC", alter: 1 },
      { fifth: -5, key: "bD", alter: -1 },
      { fifth: -4, key: "D", alter: -2 },
      { fifth: -3, key: "bE", alter: -3 },
      { fifth: 4, key: "E", alter: -4 },
      { fifth: 5, key: "B", alter: 1 },
      { fifth: -1, key: "F", alter: -5 },
      { fifth: 1, key: "G", alter: 5 }];
      var stepList = ["1", "#1", "2", "#2", "3", "4", "#4", "5", "#5", "6", "#6", "7"];
      var step2num = [{ step: "C", num: 0 }, { step: "D", num: 2 }, { step: "E", num: 4 },
      { step: "F", num: 5 }, { step: "G", num: 7 }, { step: "A", num: 9 }, { step: "B", num: 11 }];
      var number = { text: "0", tied: 0, octave: 4, dur: 0 };
      var tempNum;
      if (note.notations != undefined && note.notations.tied != undefined)
        number.tied = 1;
      else
        number.tied = 0;
      if (note.rest != undefined) {
        number.text = "0";
        number.dur = note.duration;
        number.octave = note.octave;
        return number;
      }
      for (let i = 0; i < step2num.length; i++) {
        if (step2num[i].step == note.pitch.step) {
          tempNum = step2num[i].num;
          break;
        }
      }
      for (let i = 0; i < keyAlter.length; i++) {
        if (keyAlter[i].fifth == partAttr.key.fifths) {
          tempNum += keyAlter[i].alter;
          break;
        }
      }
      if (note.pitch.alter != undefined) tempNum += note.pitch.alter;
      number.octave = note.pitch.octave;
      if (tempNum < 0) {
        tempNum += 12;
        number.octave--;
      }
      else if (tempNum > 11) {
        tempNum -= 12;
        number.octave++;
      }
      number.dur = note.duration;
      number.text = stepList[tempNum];
      return number;
    }
  }

};
