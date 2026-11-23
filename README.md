# pairwise-icc-color-transform

## 领域

游戏、图形与多媒体

## 主题

ICC 色彩转换

## 说明

配置文件回退、渲染意图与精度。

初始状态下 `imaging/icc_color_transform.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
