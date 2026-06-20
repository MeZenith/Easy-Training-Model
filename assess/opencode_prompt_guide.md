## OpenCode 使用指南 - 去AI味 Prompt 模板

### 初始化项目时（第一次对话）

将以下内容作为第一条消息发给 OpenCode：

```
我要用 Python + PySide6 开发一个AI大模型训练工具的桌面应用。

请先阅读项目根目录的 AGENTS.md 文件，这是设计规范，所有后续生成的代码必须严格遵守。

关键要求：
1. 使用 professional_theme.qss 作为全局样式表
2. 所有自定义组件继承 QFrame 并通过 setObjectName + QSS 控制样式
3. 数据可视化用 QPainter 自绘，不要引入 matplotlib
4. 布局用 QHBoxLayout/QVBoxLayout/QGridLayout，间距统一
5. 所有数值展示用等宽字体
6. 不要生成"看起来像demo"的代码，要生成生产级代码
7. 每个组件类都要有完整的类型注解和docstring
8. 错误处理要完善，不能裸 except

现在请先帮我创建项目结构。
```

### 开发具体页面时

```
基于 AGENTS.md 的设计规范，帮我实现训练监控页面。

具体需求：
- 左侧已有侧边导航（复用现有组件）
- 顶部：实验名称 + 暂停/停止按钮
- 指标卡片行：4个卡片显示 Train Loss / Learning Rate / Step / GPU Memory
- 中间区域：左侧 Loss 曲线（自绘），右侧实时日志
- 底部：进度条 + 检查点信息

注意：
- 指标卡片用我已定义的 MetricCard 组件
- Loss曲线用我已定义的 LossChart 组件  
- 日志用 QPlainTextEdit，等宽字体
- 所有样式走 QSS，不要在代码里写 setStyleSheet（除非是组件内部必须的）
- 数据更新通过 QTimer 模拟，后续替换为真实回调
```

### 修改现有页面时（最重要）

```
帮我修改 [页面名称] 页面，当前存在以下问题：

1. [具体问题描述，如：按钮间距不统一]
2. [具体问题描述，如：表格表头没有大写]

修改要求：
- 保持现有业务逻辑不变
- 只调整UI布局和样式
- 确保符合 AGENTS.md 中的设计规范
- 修改后告诉我改了哪些文件和具体改动点
```

### 审查AI生成代码时

当 OpenCode 生成代码后，检查以下清单：
□ 是否引用了 professional_theme.qss
□ 是否有硬编码颜色值（应该走QSS）
□ 是否有 Emoji 图标
□ 控件间距是否统一(12/16/20/24)
□ 是否有类型注解
□ 是否有裸 except
□ 文案是否使用了真实内容而非占位符
