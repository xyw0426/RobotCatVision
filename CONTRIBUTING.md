# 协作说明

## 开始前

1. 按 README 安装依赖并运行当前版本。
2. 新功能或较大改动先创建 Issue，写明目标、负责人和验收方式。
3. 每个分支集中处理一个任务，便于队友检查和合并。

## 日常开发

以下示例适用于已获得仓库写入权限的协作者，初次克隆后在项目根目录执行：

```powershell
git switch main
git pull --ff-only
git switch -c feature/your-task
```

完成修改后，先检查本次变更，再提交指定文件：

```powershell
git status
git diff
git add path/to/changed-file.py
git commit -m "Describe the change"
git push -u origin feature/your-task
```

将示例中的分支名、文件路径和提交说明替换为本次任务的实际内容，然后在 GitHub 创建 Pull Request，目标分支选择 `main`。请队友检查通过后再合并。

未获得写入权限的贡献者可以先 Fork 仓库，从自己的仓库提交 Pull Request。

## Pull Request 中写什么

- 修改解决了什么问题，行为有什么变化。
- 如何验证，使用了什么操作系统、Python 和依赖版本。
- 涉及视觉或控制时，附上必要的运行截图或结果。
- 涉及模型训练时，记录参数、数据划分及验证/测试结果，并说明是否替换基线模型。

## 文件约定

- 数据集保留在本地 `data/`，按 `data/README.md` 准备，不提交 CSV。
- 不提交虚拟环境、缓存、临时文件或包含密钥的本地配置。
- 仓库保留小型基线模型及对应训练记录。实验结果先放独立目录，替换基线时在 Pull Request 中说明依据。
- 修改代码时同步更新有关运行步骤和参数说明。

## 验证

目前项目没有自动化测试流程。请根据改动范围验证实际功能，并在 Pull Request 中说明未能验证的部分；硬件控制的改动需要记录实际使用的设备与结果。
