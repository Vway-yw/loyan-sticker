import asyncio
import io
import json
import logging
import os
import re

import aiohttp

from loyan.core.decorators import on_command, on_regex, plugin_handler, PluginContext
from loyan.core.decorators.registration import on_fallback as _on_fallback
from graci import LoyanImage

_logger = logging.getLogger("Loyan.表情包")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
CONFIG_FILE = os.path.join(DATA_DIR, "keywords.json")
os.makedirs(TEMPLATES_DIR, exist_ok=True)


def _load_config():
    if not os.path.exists(CONFIG_FILE):
        return {"keywords": {}}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


async def _send_preview(ctx, word, cfg=None, frame_index=None):
    if cfg is None:
        cfg = _load_config()
    kw = cfg.get("keywords", {})
    info = kw.get(word)
    if not info:
        await ctx.reply(f"关键词不存在: {word}")
        return
    tmpl_path = os.path.join(TEMPLATES_DIR, info["template"])
    if not os.path.exists(tmpl_path):
        await ctx.reply("模板文件不存在")
        return
    from PIL import Image, ImageDraw

    tmpl = Image.open(tmpl_path)
    is_gif = getattr(tmpl, "format", None) == "GIF" and getattr(tmpl, "n_frames", 1) > 1

    scale = info.get("scale", 1.0)

    if is_gif and frame_index is not None:
        n_frames = getattr(tmpl, "n_frames", 1)
        if frame_index < 0 or frame_index >= n_frames:
            await ctx.reply(f"帧号超出范围: 0~{n_frames-1}")
            return
        tmpl.seek(frame_index)
        img = tmpl.convert("RGBA")
        fp = info.get("frame_positions")
        if fp and isinstance(fp, dict) and "x" in fp:
            x = fp["x"][frame_index] if frame_index < len(fp["x"]) else info.get("position", {}).get("x", 0)
            y = fp["y"][frame_index] if frame_index < len(fp["y"]) else info.get("position", {}).get("y", 0)
        else:
            pos = info.get("position", {"x": 0, "y": 0, "w": 80, "h": 80})
            x, y = pos["x"], pos["y"]
        w = int(info.get("position", {}).get("w", 80) * scale)
        h = int(info.get("position", {}).get("h", 80) * scale)
        label = f"帧{frame_index}/{n_frames-1} ({x},{y}) {w}x{h}"
    else:
        img = tmpl.convert("RGBA")
        pos = info.get("position", {"x": 0, "y": 0, "w": 80, "h": 80})
        x, y = pos["x"], pos["y"]
        w = int(pos["w"] * scale)
        h = int(pos["h"] * scale)
        if is_gif:
            n_frames = getattr(tmpl, "n_frames", 1)
            label = f"默认位置 ({x},{y}) {w}x{h} | 共{n_frames}帧,用framepreview看单帧"
        else:
            label = f"({x},{y}) {w}x{h}"

    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=3)
    cx, cy = x + w//2, y + h//2
    draw.line([cx-10, cy, cx+10, cy], fill=(255, 0, 0), width=3)
    draw.line([cx, cy-10, cx, cy+10], fill=(255, 0, 0), width=3)
    draw.text((6, 6), label, fill=(255, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    temp_path = os.path.join(DATA_DIR, f"preview_{word}.png")
    with open(temp_path, "wb") as f:
        f.write(buf.getvalue())
    await ctx.send(LoyanImage(file_path=temp_path))


def _get_avatar_url(ctx):
    author = ctx.raw_data.get("author", {}) if ctx.raw_data else {}
    return author.get("avatar", "")


def _get_all_triggers(cfg):
    triggers = {}
    for word, info in cfg.get("keywords", {}).items():
        for t in info.get("triggers", [word]):
            triggers[t.lower()] = word
    return triggers


async def _fetch_image(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.read()
    except Exception as e:
        _logger.warning(f"下载图片失败: {e}")
    return None


def _make_circular(avatar):
    from PIL import Image, ImageDraw
    mask = Image.new("L", avatar.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, avatar.width, avatar.height), fill=255)
    avatar.putalpha(mask)
    return avatar


def _find_avatar_region(img_rgb, aw, ah):
    import numpy as np
    img = np.array(img_rgb, dtype=np.float32)
    h, w = img.shape[:2]
    if aw > w or ah > h:
        return 0, 0, min(aw, w), min(ah, h)

    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    gy, gx = np.gradient(gray)
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)

    best = (-1, 0, 0, 40)
    min_r = max(15, min(w, h) // 10)
    max_r = min(w, h) // 2
    if max_r <= min_r:
        max_r = min_r + 20
    r_step = max(2, (max_r - min_r) // 6)

    for r in range(min_r, max_r + 1, r_step):
        y_step = max(5, h // 12)
        x_step = max(5, w // 12)
        n_theta = max(12, r // 2)
        for cy in range(r, h - r, y_step):
            for cx in range(r, w - r, x_step):
                circ_val = 0.0
                for i in range(n_theta):
                    t = 2 * np.pi * i / n_theta
                    px = int(cx + r * np.cos(t))
                    py = int(cy + r * np.sin(t))
                    circ_val += edge_mag[py, px]
                circ_val /= n_theta

                in_r = max(int(r * 0.6), r - 10)
                inside = img[cy - in_r:cy + in_r, cx - in_r:cx + in_r]
                inside_var = inside.std() if inside.size > 0 else 0

                total = circ_val + r * 0.3 + inside_var * 0.2
                if total > best[0]:
                    best = (total, cx, cy, r)

    cx, cy, r = best[1], best[2], best[3]
    return max(0, cx - r), max(0, cy - r), min(r * 2, h - max(0, cy - r)), min(r * 2, w - max(0, cx - r))


def _auto_detect_frame_positions(template_path, ref_pos, ref_size):
    from PIL import Image
    import numpy as np

    tmpl = Image.open(template_path)
    is_gif = getattr(tmpl, "format", None) == "GIF" and getattr(tmpl, "n_frames", 1) > 1
    if not is_gif:
        return None

    rw, rh = ref_size

    n_frames = getattr(tmpl, "n_frames", 1)
    results = [None] * n_frames

    frames_rgb = []
    for i in range(n_frames):
        tmpl.seek(i)
        frames_rgb.append(np.array(tmpl.convert("RGB"), dtype=np.float32))

    rx, ry, rw, rh = _find_avatar_region(frames_rgb[0], rw, rh)
    results[0] = (rx, ry)

    ref = frames_rgb[0][ry:ry+rh, rx:rx+rw]
    ref_mean = ref.mean()
    ref_norm = ref - ref_mean
    ref_std = max(np.sqrt((ref_norm ** 2).sum()), 1e-6)

    for i in range(1, n_frames):
        curr = frames_rgb[i]
        best_score = -1.0
        best_pos = (rx, ry)
        margin = 60
        sy_start = max(0, min(ry - margin, curr.shape[0] - rh))
        sy_end = min(curr.shape[0] - rh, max(ry + margin, 0)) + 1
        sx_start = max(0, min(rx - margin, curr.shape[1] - rw))
        sx_end = min(curr.shape[1] - rw, max(rx + margin, 0)) + 1
        for sy in range(sy_start, sy_end):
            for sx in range(sx_start, sx_end):
                crop = curr[sy:sy+rh, sx:sx+rw]
                crop_mean = crop.mean()
                crop_norm = crop - crop_mean
                prod = (ref_norm * crop_norm).sum()
                denom = ref_std * max(np.sqrt((crop_norm ** 2).sum()), 1e-6)
                score = prod / denom
                if score > best_score:
                    best_score = score
                    best_pos = (sx, sy)
        results[i] = best_pos

    return results


def _compose_sticker(template_path, avatar_data, avatar_size=(80, 80),
                     position=None, frame_positions=None, scale=1.0):
    from PIL import Image, ImageSequence

    if scale != 1.0:
        avatar_size = (max(1, int(avatar_size[0] * scale)), max(1, int(avatar_size[1] * scale)))

    tmpl = Image.open(template_path)
    avatar = Image.open(io.BytesIO(avatar_data)).convert("RGBA")
    avatar = avatar.resize(avatar_size, Image.LANCZOS)
    avatar = _make_circular(avatar)

    is_gif = getattr(tmpl, "format", None) == "GIF" and getattr(tmpl, "n_frames", 1) > 1

    if is_gif:
        durations = []
        frames = []
        for i, frame in enumerate(ImageSequence.Iterator(tmpl)):
            if frame_positions and i < len(frame_positions) and frame_positions[i]:
                px, py = frame_positions[i]
            elif position:
                px, py = position
            else:
                px = (tmpl.width - avatar.width) // 2
                py = (tmpl.height - avatar.height) // 2

            px = max(0, min(px, tmpl.width - 1))
            py = max(0, min(py, tmpl.height - 1))
            aw = min(avatar.width, tmpl.width - px)
            ah = min(avatar.height, tmpl.height - py)
            avatar_frame = avatar if (aw == avatar.width and ah == avatar.height) else avatar.crop((0, 0, aw, ah))

            frame_rgba = frame.convert("RGBA")
            frame_rgba.paste(avatar_frame, (px, py), avatar_frame)
            frame_p = frame_rgba.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
            frames.append(frame_p)
            durations.append(frame.info.get("duration", 60))

        buf = io.BytesIO()
        frames[0].save(
            buf, format="GIF", save_all=True, append_images=frames[1:],
            duration=durations, loop=0, disposal=2,
        )
        buf.seek(0)
        return buf, "gif"

    if position:
        x, y = position
    else:
        x = (tmpl.width - avatar.width) // 2
        y = (tmpl.height - avatar.height) // 2

    x = max(0, min(x, tmpl.width - 1))
    y = max(0, min(y, tmpl.height - 1))
    aw = min(avatar.width, tmpl.width - x)
    ah = min(avatar.height, tmpl.height - y)
    if aw < avatar.width or ah < avatar.height:
        avatar = avatar.crop((0, 0, aw, ah))

    tmpl_rgba = tmpl.convert("RGBA")
    tmpl_rgba.paste(avatar, (x, y), avatar)
    buf = io.BytesIO()
    tmpl_rgba.save(buf, format="PNG")
    buf.seek(0)
    return buf, "png"


async def _do_generate(ctx, word, qq=None):
    if not qq:
        avatar_url = _get_avatar_url(ctx)
        if not avatar_url:
            await ctx.reply("无法获取你的头像")
            return
    else:
        if not re.match(r"^\d{5,11}$", qq):
            await ctx.reply("QQ号格式不正确")
            return
        avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={qq}&s=100"

    cfg = _load_config()
    kw = cfg.get("keywords", {})
    info = kw.get(word)
    if not info:
        return

    tmpl_path = os.path.join(TEMPLATES_DIR, info["template"])
    if not os.path.exists(tmpl_path):
        await ctx.reply(f"模板文件不存在: {info['template']}")
        return

    pos = info.get("position", {"x": 0, "y": 0, "w": 80, "h": 80})
    aw, ah = pos.get("w", 80), pos.get("h", 80)
    out_scale = info.get("scale", 1.0)

    fp = info.get("frame_positions")
    if fp is None:
        from PIL import Image
        tmp_img = Image.open(tmpl_path)
        tmin = min(tmp_img.size)
        max_av = max(40, min(tmin, int(tmin * 0.8)))

        is_gif = getattr(tmp_img, "format", None) == "GIF" and getattr(tmp_img, "n_frames", 1) > 1
        if is_gif:
            detected = _auto_detect_frame_positions(tmpl_path, (0, 0), (min(aw, max_av), min(ah, max_av)))
            if detected:
                fp = [(p[0], p[1]) for p in detected]
                kw[word]["frame_positions"] = {"x": [p[0] for p in detected], "y": [p[1] for p in detected]}
                _save_config(cfg)
            else:
                fp = None
        else:
            import numpy as np
            arr = np.array(tmp_img.convert("RGB"))
            dx, dy, dw, dh = _find_avatar_region(arr, min(aw, max_av), min(ah, max_av))
            kw[word]["position"]["x"] = dx
            kw[word]["position"]["y"] = dy
            kw[word]["position"]["w"] = min(dw, max_av)
            kw[word]["position"]["h"] = min(dh, max_av)
            _save_config(cfg)
            fp = None

    if fp:
        if isinstance(fp, dict) and "x" in fp:
            frame_positions = list(zip(fp["x"], fp["y"]))
        else:
            frame_positions = fp
    else:
        frame_positions = None

    avatar_data = await _fetch_image(avatar_url)
    if not avatar_data:
        await ctx.reply("下载头像失败")
        return

    try:
        buf, ext = _compose_sticker(tmpl_path, avatar_data,
                                    avatar_size=(aw, ah),
                                    position=(pos.get("x", 0), pos.get("y", 0)),
                                    frame_positions=frame_positions,
                                    scale=out_scale)
        temp_path = os.path.join(DATA_DIR, f"temp_sticker.{ext}")
        with open(temp_path, "wb") as f:
            f.write(buf.getvalue())
        await ctx.send(LoyanImage(file_path=temp_path))
    except Exception as e:
        _logger.error(f"生成表情失败: {e}")
        await ctx.reply(f"生成失败: {e}")


@on_command("/生成表情", "/表情关键词", "/表情帮助", "/表情")
@plugin_handler
async def handle_dispatch(ctx: PluginContext):
    cmd = ctx.command

    if cmd == "/表情帮助":
        await ctx.reply(
            "互动表情包生成 v2.1.0\n"
            "功能: 将QQ头像合成到模板图片生成表情包\n"
            "\n"
            "命令:\n"
            "  <关键词> [QQ号] — 直接输入关键词生成（可指定QQ号）\n"
            "    例: 摸     摸 123456\n"
            "  /表情 list — 查看所有关键词\n"
            "  /表情 setpos <关键词> <方向+数字> — 微调位置\n"
            "    例: /表情 setpos 摸 右移10   /表情 setpos 摸 缩小5\n"
            "    支持: 左移/右移/上移/下移/放大/缩小\n"
            "  /表情 preview <关键词> — 预览头像框位置\n"
            "  /表情 scale <关键词> <比例> — 缩放头像大小(0.1~2.0)\n"
            "\n"
            "调试命令:\n"
            "  /表情 debug <关键词> [QQ号] — 直接生成成品表情包用于确认效果\n"
            "  /表情 setframe <关键词> <帧号> <x,y> — 设置动图指定帧的位置\n"
            "    例: /表情 setframe 摸 2 50,60\n"
            "  /表情 framepreview <关键词> [帧号] — 预览动图某帧/查看全部帧位置\n"
            "    例: /表情 framepreview 摸 3\n"
            "  /表情 clearframe <关键词> — 清除动图逐帧位置,回退默认\n"
            "\n"
            "  /表情 — 显示本帮助\n"
            "\n"
            "示例:\n"
            "  /生成表情 摸 3441456163\n"
            "  /生成表情 摸 (用自己的头像)\n"
            "  /表情 scale 摸 0.7"
        )
        return

    if cmd == "/表情关键词" or cmd == "/表情":
        parts = ctx.raw_text.split(None, 2)
        if len(parts) < 2:
            if cmd == "/表情":
                await ctx.reply(
                    "互动表情包生成 v2.1.0\n"
                    "命令:\n"
                    "  /表情 list — 查看所有关键词\n"
                    "  /表情 setpos <关键词> <方向+数字> — 微调位置\n"
                    "  /表情 preview <关键词> — 预览头像框位置\n"
                    "  /表情 scale <关键词> <比例> — 缩放头像大小\n"
                    "  /表情 debug <关键词> [QQ号] — 调试-生成成品确认效果\n"
                    "  /表情 setframe <关键词> <帧号> <x,y> — 调试-设置动图帧位置\n"
                    "  /表情 framepreview <关键词> [帧号] — 调试-预览动图帧位置\n"
                    "  /表情 clearframe <关键词> — 调试-清除动图逐帧位置\n"
                    "  /表情帮助 — 查看完整帮助"
                )
            else:
                await ctx.reply(
                    "用法:\n"
                    "  /表情关键词 list — 查看关键词列表\n"
                    "  /表情关键词 add <关键词> <模板文件名> [x,y,w,h] — 添加关键词\n"
                    "  /表情关键词 del <关键词> — 删除关键词\n"
                    "  /表情关键词 setpos <关键词> <x,y,w,h> — 设置头像位置和大小\n"
                    "\n"
                    "坐标说明: x,y 为头像左上角位置, w,h 为头像宽高\n"
                    "示例: /表情关键词 add 舔狗日记 template.png 50,100,120,120"
                )
            return

        sub = parts[1]
        cfg = _load_config()

        if sub == "list":
            kw = cfg.get("keywords", {})
            if not kw:
                await ctx.reply("暂无关键词")
                return
            lines = ["已配置关键词:"]
            for word, info in kw.items():
                tmpl = info.get("template", "")
                triggers = ", ".join(info.get("triggers", [word])[:3])
                pos = info.get("position", {})
                pos_str = f"{pos.get('x',0)},{pos.get('y',0)},{pos.get('w',80)},{pos.get('h',80)}"
                lines.append(f"  {word} [{pos_str}] ({triggers}...)")
            await ctx.reply("\n".join(lines))
            return

        if sub == "add" and len(parts) >= 3:
            sub_parts = parts[2].split(None, 2)
            if len(sub_parts) < 2:
                await ctx.reply("用法: /表情关键词 add <关键词> <模板文件名> [x,y,w,h]")
                return
            word = sub_parts[0]
            tmpl_file = sub_parts[1]
            tmpl_path = os.path.join(TEMPLATES_DIR, tmpl_file)
            if not os.path.exists(tmpl_path):
                await ctx.reply(f"模板文件不存在: {tmpl_file}")
                return
            pos = {"x": 0, "y": 0, "w": 80, "h": 80}
            if len(sub_parts) > 2:
                try:
                    vals = [int(v.strip()) for v in sub_parts[2].split(",")]
                    if len(vals) == 4:
                        pos = {"x": vals[0], "y": vals[1], "w": vals[2], "h": vals[3]}
                except:
                    pass
            cfg.setdefault("keywords", {})[word] = {"template": tmpl_file, "position": pos, "triggers": [word]}
            _save_config(cfg)
            await ctx.reply(f"已添加关键词: {word} → {tmpl_file}")
            return

        if sub == "del" and len(parts) >= 3:
            word = parts[2]
            cfg.setdefault("keywords", {}).pop(word, None)
            _save_config(cfg)
            await ctx.reply(f"已删除关键词: {word}")
            return

        if sub == "setpos" and len(parts) >= 3:
            sub_parts = parts[2].split(None, 1)
            if len(sub_parts) < 2:
                await ctx.reply("用法: /表情 setpos <关键词> <x,y,w,h>")
                return
            word = sub_parts[0]
            arg = sub_parts[1].strip()
            kw = cfg.setdefault("keywords", {})
            if word not in kw:
                await ctx.reply(f"关键词不存在: {word}")
                return
            pos = kw[word].get("position", {"x": 0, "y": 0, "w": 80, "h": 80})

            import re as _re
            m = _re.match(r"^([左右上下][移]?|[放大缩][小]?)\s*(\d+)$", arg)
            if m:
                action, n = m.group(1), int(m.group(2))
                if action in ("左移", "左"):
                    pos["x"] = max(0, pos["x"] - n)
                    action_label = f"左移{n}"
                elif action in ("右移", "右"):
                    pos["x"] = pos["x"] + n
                    action_label = f"右移{n}"
                elif action in ("上移", "上"):
                    pos["y"] = max(0, pos["y"] - n)
                    action_label = f"上移{n}"
                elif action in ("下移", "下"):
                    pos["y"] = pos["y"] + n
                    action_label = f"下移{n}"
                elif action in ("放大", "大"):
                    pos["w"] = pos["w"] + n
                    pos["h"] = pos["h"] + n
                    action_label = f"放大{n}"
                elif action in ("缩小", "小"):
                    pos["w"] = max(10, pos["w"] - n)
                    pos["h"] = max(10, pos["h"] - n)
                    action_label = f"缩小{n}"
                else:
                    await ctx.reply("支持: 左移/右移/上移/下移/放大/缩小 + 数字")
                    return
            else:
                try:
                    vals = [int(v.strip()) for v in arg.split(",")]
                    if len(vals) != 4:
                        raise ValueError
                except:
                    await ctx.reply("支持: x,y,w,h 或 左移/右移/上移/下移/放大/缩小 + 数字\n如: /表情 setpos 摸 右移10")
                    return
                pos = {"x": vals[0], "y": vals[1], "w": vals[2], "h": vals[3]}
                kw[word]["position"] = pos
                _save_config(cfg)
                await ctx.reply(f"已更新 {word} 位置: {vals[0]},{vals[1]} 大小: {vals[2]}x{vals[3]}")
                await _send_preview(ctx, word)
                return

            kw[word]["position"] = pos
            kw[word].pop("frame_positions", None)
            _save_config(cfg)
            await _send_preview(ctx, word)
            return

        if sub == "autoframe" and len(parts) >= 3:
            word = parts[2]
            kw = cfg.setdefault("keywords", {})
            if word not in kw:
                await ctx.reply(f"关键词不存在: {word}")
                return
            tmpl_path = os.path.join(TEMPLATES_DIR, kw[word]["template"])
            if not os.path.exists(tmpl_path):
                await ctx.reply(f"模板文件不存在")
                return
            from PIL import Image
            tmpl = Image.open(tmpl_path)
            nf = getattr(tmpl, "n_frames", 1)
            if nf <= 1:
                await ctx.reply(f"{word} 不是动图")
                return
            pos = kw[word].get("position", {"x": 0, "y": 0, "w": 80, "h": 80})
            detected = _auto_detect_frame_positions(
                tmpl_path, (pos["x"], pos["y"]), (pos["w"], pos["h"]))
            if detected:
                kw[word]["frame_positions"] = {
                    "x": [p[0] for p in detected],
                    "y": [p[1] for p in detected]
                }
                _save_config(cfg)
                await ctx.reply(f"已自动检测 {word} 的 {nf} 帧位置")
            else:
                await ctx.reply("自动检测失败")
            return

        if sub == "preview" and len(parts) >= 3:
            preview_parts = parts[2].split(None, 1)
            word = preview_parts[0]
            preview_scale = None
            if len(preview_parts) > 1:
                try:
                    s = float(preview_parts[1].strip())
                    if 0.1 <= s <= 2.0:
                        preview_scale = s
                except ValueError:
                    pass
            if preview_scale is not None:
                kw = cfg.setdefault("keywords", {})
                if word in kw:
                    kw[word]["scale"] = preview_scale
                    _save_config(cfg)
            await _send_preview(ctx, word, cfg)
            return

        if sub == "scale" and len(parts) >= 3:
            sub_parts = parts[2].split(None, 1)
            if len(sub_parts) < 2:
                await ctx.reply("用法: /表情关键词 scale <关键词> <缩放比例>\n例如: /表情关键词 scale 摸 0.7")
                return
            word = sub_parts[0]
            try:
                s = float(sub_parts[1])
                if s <= 0 or s > 2:
                    raise ValueError
            except:
                await ctx.reply("缩放比例需为 0.1~2.0 的数字")
                return
            kw = cfg.setdefault("keywords", {})
            if word not in kw:
                await ctx.reply(f"关键词不存在: {word}")
                return
            kw[word]["scale"] = s
            _save_config(cfg)
            await ctx.reply(f"已设置 {word} 缩放比例: {s}")
            return

        if sub == "debug" and len(parts) >= 3:
            sub_parts = parts[2].split(None, 1)
            word = sub_parts[0]
            qq = sub_parts[1] if len(sub_parts) > 1 else None
            kw = cfg.setdefault("keywords", {})
            if word not in kw:
                await ctx.reply(f"关键词不存在: {word}")
                return
            await ctx.reply(f"正在生成 {word} 的调试预览...")
            await _do_generate(ctx, word, qq)
            return

        if sub == "setframe" and len(parts) >= 3:
            sub_parts = parts[2].split(None, 2)
            if len(sub_parts) < 3:
                await ctx.reply("用法: /表情 setframe <关键词> <帧号> <x,y>\n例: /表情 setframe 摸 2 50,60")
                return
            word = sub_parts[0]
            kw = cfg.setdefault("keywords", {})
            if word not in kw:
                await ctx.reply(f"关键词不存在: {word}")
                return
            tmpl_path = os.path.join(TEMPLATES_DIR, kw[word]["template"])
            if not os.path.exists(tmpl_path):
                await ctx.reply("模板文件不存在")
                return
            from PIL import Image
            tmpl = Image.open(tmpl_path)
            nf = getattr(tmpl, "n_frames", 1)
            if nf <= 1:
                await ctx.reply(f"{word} 不是动图,请用 setpos 调整静态位置")
                return
            try:
                frame_idx = int(sub_parts[1])
            except ValueError:
                await ctx.reply("帧号必须是数字")
                return
            if frame_idx < 0 or frame_idx >= nf:
                await ctx.reply(f"帧号超出范围: 0~{nf-1}")
                return
            try:
                vals = [int(v.strip()) for v in sub_parts[2].split(",")]
                if len(vals) != 2:
                    raise ValueError
            except:
                await ctx.reply("格式: x,y\n例: /表情 setframe 摸 2 50,60")
                return
            fp = kw[word].get("frame_positions")
            if fp is None or not isinstance(fp, dict) or "x" not in fp:
                fp = {"x": [0] * nf, "y": [0] * nf}
                for i in range(nf):
                    fp["x"][i] = kw[word].get("position", {}).get("x", 0)
                    fp["y"][i] = kw[word].get("position", {}).get("y", 0)
            while len(fp["x"]) < nf:
                fp["x"].append(fp["x"][-1] if fp["x"] else 0)
                fp["y"].append(fp["y"][-1] if fp["y"] else 0)
            fp["x"][frame_idx] = vals[0]
            fp["y"][frame_idx] = vals[1]
            kw[word]["frame_positions"] = fp
            _save_config(cfg)
            await ctx.reply(f"已设置 {word} 帧{frame_idx} 位置: ({vals[0]},{vals[1]})")
            await _send_preview(ctx, word, cfg, frame_index=frame_idx)
            return

        if sub == "framepreview" and len(parts) >= 3:
            sub_parts = parts[2].split(None, 1)
            word = sub_parts[0]
            kw = cfg.setdefault("keywords", {})
            if word not in kw:
                await ctx.reply(f"关键词不存在: {word}")
                return
            tmpl_path = os.path.join(TEMPLATES_DIR, kw[word]["template"])
            if not os.path.exists(tmpl_path):
                await ctx.reply("模板文件不存在")
                return
            from PIL import Image
            tmpl = Image.open(tmpl_path)
            nf = getattr(tmpl, "n_frames", 1)
            if nf <= 1:
                await ctx.reply(f"{word} 不是动图,用 preview 即可")
                return
            frame_idx = None
            if len(sub_parts) > 1:
                try:
                    frame_idx = int(sub_parts[1])
                except ValueError:
                    pass
            if frame_idx is None:
                info = kw[word]
                fp = info.get("frame_positions")
                if fp and isinstance(fp, dict) and "x" in fp:
                    lines = [f"{word} 各帧位置:"]
                    for i in range(min(nf, len(fp["x"]))):
                        lines.append(f"  帧{i}: ({fp['x'][i]},{fp['y'][i]})")
                    lines.append(f"共{nf}帧, 用 /表情 framepreview {word} <帧号> 查看单帧")
                    await ctx.reply("\n".join(lines))
                else:
                    pos = info.get("position", {})
                    await ctx.reply(f"{word} 未设置逐帧位置, 默认: ({pos.get('x',0)},{pos.get('y',0)})\n共{nf}帧")
                return
            if frame_idx < 0 or frame_idx >= nf:
                await ctx.reply(f"帧号超出范围: 0~{nf-1}")
                return
            await _send_preview(ctx, word, cfg, frame_index=frame_idx)
            return

        if sub == "clearframe" and len(parts) >= 3:
            word = parts[2]
            kw = cfg.setdefault("keywords", {})
            if word not in kw:
                await ctx.reply(f"关键词不存在: {word}")
                return
            kw[word].pop("frame_positions", None)
            _save_config(cfg)
            await ctx.reply(f"已清除 {word} 的逐帧位置, 回退到默认位置")
            return

        await ctx.reply("未知子命令，使用 /表情关键词 查看帮助")
        return

    if cmd == "/生成表情":
        parts = ctx.raw_text.split(None, 2)
        if len(parts) < 2:
            await ctx.reply("用法: /生成表情 <关键词> [QQ号]\n不填QQ号则用你自己的头像")
            return

        word = parts[1]
        qq = parts[2] if len(parts) > 2 else None

        cfg = _load_config()
        kw = cfg.get("keywords", {})
        if word not in kw:
            triggers = _get_all_triggers(cfg)
            if word.lower() in triggers:
                word = triggers[word.lower()]
            else:
                await ctx.reply(f"未找到关键词: {word}\n用 /表情关键词 list 查看可用关键词")
                return

        await ctx.reply("正在生成...")
        await _do_generate(ctx, word, qq)
        return


@on_regex(r"^(list|帮助|help)$")
@plugin_handler
async def handle_list(ctx: PluginContext):
    cfg = _load_config()
    kw = cfg.get("keywords", {})
    if not kw:
        await ctx.reply("暂无关键词")
        return
    lines = ["已配置关键词:"]
    for word, info in kw.items():
        tmpl = info.get("template", "")
        triggers = ", ".join(info.get("triggers", [word])[:3])
        pos = info.get("position", {})
        pos_str = f"{pos.get('x',0)},{pos.get('y',0)},{pos.get('w',80)},{pos.get('h',80)}"
        lines.append(f"  {word} [{pos_str}] ({triggers}...)")
    await ctx.reply("\n".join(lines))


@_on_fallback()
async def handle_fallback(self_bot, bot, message, user_id, chat_type, permission, log_func):
    try:
        import json, os, re
        cfg_path = os.path.join(os.path.dirname(__file__), "keywords.json")
        cfg = json.load(open(cfg_path))
        triggers = {}
        for word, info in cfg.get("keywords", {}).items():
            for t in info.get("triggers", [word]):
                triggers[t.lower()] = word

        raw = message.get("text", "").strip()
        parts = raw.split(None, 1)
        key = parts[0].lower()
        qq = parts[1] if len(parts) > 1 and re.match(r"^\d{5,11}$", parts[1]) else None

        word = triggers.get(key)
        if not word:
            return False

        target = str(message.get("raw_data", {}).get("group_id") if chat_type == "group" else user_id)

        from loyan.core.decorators.context import PluginContext
        from graci import loyan_send_msg
        ctx = PluginContext(
            sender_id=user_id,
            target_id=target,
            chat_type=chat_type,
            raw_text=raw,
            is_at_bot=message.get("is_at_bot", False),
            raw_data=message.get("raw_data", {}),
        )

        async def ctx_send(*segs, ct=None):
            return await loyan_send_msg(
                target, *segs, chat_type=ct or chat_type,
            )
        ctx.send = ctx_send

        async def _reply(text):
            from graci import LoyanText
            await loyan_send_msg(target, LoyanText(text=text), chat_type=chat_type)
        ctx.reply = _reply

        log_func.info(f"[表情包] 关键词触发: {word} QQ={qq or '自己'}")
        await _do_generate(ctx, word, qq)
        return True
    except Exception:
        return False
