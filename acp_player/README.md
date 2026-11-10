# ACP 参考动作播放器

单端口（默认 8188）静态播放器：左侧审阅 + 右侧 three.js 视口 + 底部接触时间线。
后端 `server.py` 负责把片段烘焙成一个二进制 payload，前端在本地按 50 Hz 播放。

## 手机 / 窄屏

宽度 ≤ 820px（或主指针为触屏）时，控制栏收进底部抽屉，视口占满整屏：

- 右上角「☰ 控制」打开抽屉，「✕ 收起」或点遮罩关闭；选完片段会自动收起；
- 左上角「‹ 片段名 ›」直接切上一段 / 下一段，不用打开抽屉；
- 画布上单指旋转，双指捏合缩放、同时拖动平移；
- 时间线和底部拉手都能用手指拖；时间轴高度按屏幕高度自动选一个较矮的默认值。

## 前端：TypeScript 源码

源码在 `src/`，`tsc` 直接编译到 `static/`，浏览器用原生 ESM 加载，没有打包器。

```bash
cd acp_player
npm install          # 第一次；只装 typescript 和 @types/three
npm run build        # src/*.ts -> static/*.js
npm run watch        # 改源码时自动编译
npm run typecheck    # 只做类型检查
```

`static/app.js`、`static/ui.js` 是生成文件，不要直接改；要改就改 `src/app.ts` / `src/ui.ts`。
`static/vendor/` 里是 vendored three.js，类型来自 `src/vendor/three.module.d.ts`（`@types/three`）。

页面是静态文件，改完重新编译、刷新浏览器即可；只有改 `server.py` 才需要
`systemctl --user restart acp-webviewer`。

## 时间线

底部时间线横向就是整段接触，像剪辑软件一样平移和缩放：

- 滚轮左右平移；`Ctrl` / `⌘` + 滚轮以光标位置为中心缩放；
- `-` / `=` 缩放，`0` 恢复全览；工具条上的 `−` / `＋` / `全览` 等价；
- `Alt` + 左键或中键拖动平移；播放时视口自动跟随播放头；
- 顶部那条刻度条就是播放头（小三角在它上面）：按住左右拖移动播放头，编辑模式下也一样，
  只移动不改动接触；底部的时间坐标轴区域同样可以拖；
- 拖动视口与时间线之间的拉手可以调高时间线（行高跟着变大），双击拉手恢复默认高度，
  高度记在浏览器的 `localStorage` 里；
- 3D 视口里每个生效的参考接触会在地面投影一个同色圆环，接触点离地时还有一条细线连到地面
  （跟随「显示 → 参考接触」开关）。

## 键盘

播放头按剪辑软件的习惯走键盘。只有真正在打字的地方（备注框、搜索框、下拉里的搜索）才把方向键
留给输入框；`Esc` 会退出搜索框、把键盘还给时间线：

| 按键 | 作用 |
| --- | --- |
| `←` / `→` | 上一帧 / 下一帧（按住连发） |
| `Shift` + `←` / `→`，或 `,` / `.` | ±10 帧 |
| `Home` / `End` | 片段首 / 尾 |
| `↑` / `↓` | 上一个 / 下一个接触点（接触标记翻转的帧）；选中的部位没有接触时跟最忙的那条轨道 |
| `空格` | 播放 / 暂停 |
| `[` / `]` | 上一段 / 下一段 |
| `-` / `=` / `0` | 缩小 / 放大 / 全览 |

点过「播放」「循环」「部位」这些控件之后方向键仍然控制播放头（以前焦点留在复选框上，
方向键就失灵了）；点过速度滑块会自动把焦点还给时间线。

## 手修接触（时间线）

底部时间线就是接触数据，可以手动改：

- `✎ 编辑接触`（快捷键 `E`）进入编辑；进入时会自动暂停播放，每条接触轨道上的色块会加描边。
- 直接拖色块，像剪辑软件里拖片段：
  - 拖块的左 / 右边缘（光标变成 `↔`）改这一段的起点 / 终点；
  - 拖块的中间（光标变成手形）整段平移；
  - 拖动时实时预览，松开即生效，`Esc` 取消整次拖动；一次拖动只记一步撤销。
  - 与相邻段之间至少留 1 帧空隙，拖不过去；拖出的新帧沿用该段最近的接触点，
    所以 3D 标记仍然落在脚 / 手原来的落点上，不会跟着摆动中的 FK 飘走。
- 在空白格子上拖动涂抹；单击切换单帧；按住 `Shift` 拖动是擦除（`Shift` 时不会拖块）。
- 工具条里选部位（足 / 膝 / 手 / 肘 / 骨盆），`＋ 接触` 在播放头处开始区间，
  把播放头拖到终点（或 `Enter`、再点一次按钮）结束；`－ 擦除` 同理；`Esc` 取消区间。
- `Ctrl+Z` / `Ctrl+Shift+Z` 撤销、重做；`Ctrl+S` 保存。
- 编辑模式下拖时间线顶部的刻度条（或底部的坐标轴）仍然可以拖播放头，`←` / `→` 逐帧挪。
- 空心格 = 相对上次保存被擦掉的接触；工具栏显示未保存 / 已保存的改动帧数。

保存不写数据集本身，而是写到独立的修改层：

```
/home/duanxin/Datasets/acp/contact_edits/<dataset>/<clip>.npz
```

可用 `ACP_CONTACT_EDIT_ROOT` 环境变量改位置。播放器烘焙片段时会自动套用修改层，
所以 `v3_final` 保持 `tools/build_final.sh` 生成的原样。`清除已保存` 会删掉对应修改层文件。

新增接触的位置优先用 sidecar 里该帧的 FK 位置；sidecar 没给位置时回退到机器人 FK。
前端保存时会把整段 `contact_pos_w`（float32，`T × N × 3`）一起上传，服务端只覆盖被编辑的
帧，零值 / 非有限值视为缺失并回退 FK，所以烘焙出来的 3D 标记正好落在对应身体部位上。

### 把修改写进训练集副本

`v3_train3` 这类训练副本要真正吃到手修结果，用 `apply` 把修改层写进它的 sidecar：

```bash
# 先看会改哪些
uv run --no-sync python acp_player/server.py apply \
  --dataset /home/duanxin/Datasets/acp/v3_train3 --dry-run

# 真正写回（会重写 <clip>.contacts.npz，并在 params.motionedit 里留一条记录）
uv run --no-sync python acp_player/server.py apply \
  --dataset /home/duanxin/Datasets/acp/v3_train3

# 只处理某一段
uv run --no-sync python acp_player/server.py apply \
  --dataset /home/duanxin/Datasets/acp/v3_train3 \
  --clip beam/beam__orig_stones_full_beam_zc2_v10.npz
```

修改层按 **数据集目录名** 对应（`v3_final`、`v3_train3` …）。重新跑 `build_final.sh`
不会带上手修；需要时再对新的训练副本执行一次 `apply`。
