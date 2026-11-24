# pairwise-icc-color-transform

## 领域

游戏、图形与多媒体

## 主题

ICC 色彩转换

## 说明

配置文件回退、渲染意图与精度。

实现了一个基于 Little CMS 2（lcms2）的完整 ICC 色彩转换引擎，修复了初始
版本中的六类缺陷。所有平台通过同一条 lcms2 代码路径（ctypes 直连
`liblcms2`）执行转换，保证跨平台结果一致。

运行测试：

```bash
python -m pytest -q
```

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `imaging/_lcms.py` | lcms2 的 ctypes 绑定、TYPE/意图/标志常量（取自 `lcms2.h` 并经实测核对） |
| `imaging/model.py` | 数据模型：`ColorImage`、`ICCProfile`、`Intent`、`ColorSpace`、审计事件 |
| `imaging/engine.py` | 核心引擎：指纹隔离缓存、高位深流水线、回退、校验、Lab/delta-E、批处理 |
| `imaging/icc_color_transform.py` | 15 个公共入口函数 |
| `tests/test_icc_color_transform.py` | 原始占位契约测试 |
| `tests/test_icc_fixes.py` | 覆盖六项修复的行为测试 |

## 六项修复与对应实现

1. **缺失配置按声明回退** —— `resolve_profile()` / `profile_fallback()`。
   无嵌入配置时使用图像声明的回退配置并写入审计；严格模式下既无嵌入也无
   声明则抛错，宽松模式才做显式、可审计的 sRGB 假设，绝不静默当 sRGB。
2. **渲染意图跟随文件** —— `resolve_intent()` / `intent_match()`。优先级为
   显式覆盖 > 文件声明 > 配置文件头 > 默认相对色度；绝对色度会被标记为
   压暗部（`crushes_shadows`）。默认相对色度以保留暗部细节。
3. **全程保持高位深** —— 8 位走 8 位流水线、16 位走 16 位流水线，输出位深
   不低于输入；32 位/浮点提升到 16 位而非降到 8 位。见 `bit_depth()` 与
   `_promote_to_supported()`。
4. **嵌入与像素空间校验** —— `validate()` / `embed_mismatch()`。嵌入配置的
   色彩空间与实际像素空间不一致时给出告警，严格模式抛
   `TransformValidationError`。
5. **缓存按配置指纹隔离** —— `_TransformCache` 的键包含源/目标配置 SHA-256
   指纹、色彩空间、位深、意图与标志，不同批次、不同配置永不复用同一转换，
   杜绝跨批次颜色漂移；相同配置才命中缓存。
6. **跨平台同一转换库** —— 统一使用 lcms2，并以 `NOOPTIMIZE` 关闭平台相关
   LUT 优化，`determinism()` 验证多次运行字节级一致。

## 关键不变量（测试固化）

- 中性灰阶在任意亮度下 Lab 的 a*=b*=0，白点映射到 L*=100（色度适应正确）。
- 16 位暗部的相邻层级转换后仍可区分，不产生断层。
- 相同输入多次转换字节级一致；同色 delta-E 为 0，白/黑 delta-E 约 100。
