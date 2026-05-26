# Gibbs采样算法可视化平台

基于Flask的Gibbs采样算法可视化分析工具，支持参数自定义、实时采样、收敛诊断和结果导出。

## 功能特性

- **参数自定义**: 支持调整条件分布参数、初始值、迭代次数等
- **多种分布支持**: 正态分布和均匀分布混合使用
- **实时可视化**: 轨迹图、直方图、自相关图、联合分布散点图
- **收敛诊断**: 有效样本量(ESS)、自相关系数计算
- **结果导出**: 支持JSON和CSV格式导出
- **历史记录**: 自动保存带时间戳的快照

## 快速开始

### 本地开发

```bash
# 克隆仓库
git clone <your-repo-url>
cd main

# 安装依赖
pip install -r requirements.txt

# 运行开发服务器
python app.py
```

打开浏览器访问 `http://localhost:5000`

### GitHub Pages 部署

本项目可以使用 GitHub Actions 部署到 GitHub Pages 或使用 Vercel/Render 等平台。

## 项目结构

```
main/
├── app.py              # Flask应用入口
├── requirements.txt     # 依赖清单
├── templates/
│   └── index.html      # 可视化页面
└── json/               # JSON结果存储目录
```

## API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 返回可视化页面 |
| `/api/generate` | POST | 生成Gibbs采样数据 |
| `/api/load_json` | GET | 加载最新数据 |
| `/api/list_files` | GET | 列出历史文件 |
| `/api/load_file/<filename>` | GET | 加载指定文件 |
| `/api/export` | POST | 导出CSV |
| `/api/presets` | GET | 获取预设配置 |

## 使用示例

```javascript
// 生成采样数据
fetch('/api/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        param_x: [0.5, 0.3, 0.8],
        param_y: [0.5, 0.4, 0.7],
        param_z: [0.3, 0.4, 0.9],
        init_values: [1.0, 2.0, 2.0],
        iterations: 10000,
        burn_in: 1000
    })
})
```

## 技术栈

- **后端**: Flask 2.3.3
- **前端**: HTML5 + Chart.js 4.4.0
- **数据处理**: NumPy

## 许可证

MIT License
