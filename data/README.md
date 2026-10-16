# FER2013 数据准备

实时摄像头演示使用已训练的模型，不需要下载训练数据。只有重新训练或运行独立测试评估时才需要本数据集。

请自行准备 FER2013 的 CSV 版本，并放到：

```text
data/fer2013.csv
```

CSV 必须包含以下三列：

| 列名 | 内容 |
| --- | --- |
| `emotion` | 类别编号 `0` 到 `6` |
| `pixels` | 2304 个以空格分隔的像素值，对应 48×48 灰度图 |
| `Usage` | `Training`、`PublicTest` 或 `PrivateTest` |

类别顺序：`0=Angry`、`1=Disgust`、`2=Fear`、`3=Happy`、`4=Sad`、`5=Surprise`、`6=Neutral`。

当前本地 CSV 约为 301 MB，默认由 `.gitignore` 排除，不上传到普通 Git 仓库。克隆仓库后，需要自行补充此文件才能训练或评估。
