# Gemini Telegram AI Bot

这是一个功能强大的Telegram机器人，集成了Google的Gemini和Imagen模型，提供多模态对话、高级图片生成、文本转语音等多种AI功能。项目支持用户授权和Docker化部署，确保了安全性和易用性。

## ✨ 功能特性

- **多模态对话**: 支持与Gemini模型进行流畅的文本对话，并能在对话中理解图片内容。
- **高级图片生成**:
    - **Imagen**: 通过 `/image` 或 `/generate` 命令，使用Imagen模型生成高质量、特定风格的图片。
    - **Gemini**: 支持在对话中通过关键词（如“画图”）或回复图片进行创作和编辑。
- **文本转语音 (TTS)**: 使用 `/speak` 命令将文本转换为自然的语音。
- **文件问答**: 上传图片或文档，即可就其内容进行提问。
- **用户授权**: 通过 `auth.json` 文件管理用户白名单，只有授权用户才能使用机器人，防止API滥用。
- **Docker化**: 提供 `Dockerfile`，支持一键构建和部署，方便移植和管理。
- **可配置性**: 通过 `/settings` 命令，用户可以方便地切换聊天模式、AI模型、语音声线等。

## 🚀 安装与配置

### 1. 前提条件

- Python 3.9+
- Docker (推荐)
- 一个Telegram Bot Token
- 一个Google AI (Gemini) API Key

### 2. 克隆项目

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

在项目根目录创建一个名为 `.env` 的文件，并填入您的API密钥：

```env
TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# GEMINI_BASE_URL="YOUR_CUSTOM_GEMINI_API_ENDPOINT" # (可选) 如果您使用代理，请取消注释并配置
```

### 5. 配置授权用户

在您的 `.env` 文件中，添加一个新的环境变量 `AUTHORIZED_USER_IDS`。这是一个用**逗号**分隔的Telegram用户ID列表。

```env
TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
AUTHORIZED_USER_IDS="123456789,987654321" # 将这里的ID替换为您自己的授权用户ID
```
只有在这个列表中的用户才能使用机器人。

## 🏃‍♂️ 运行机器人

### 本地运行

直接通过Python运行：

```bash
python bot.py
```

### 使用 Docker 运行 (推荐)

1.  **构建Docker镜像**:
    ```bash
    docker build -t telegram-gemini-bot .
    ```

2.  **运行Docker容器**:
    ```bash
    docker run --env-file .env -d --name my-gemini-bot telegram-gemini-bot
    ```
    - `--env-file .env`: 安全地将您的API密钥从 `.env` 文件注入到容器中。
    - `-d`: 在后台运行容器。
    - `--name my-gemini-bot`: 为您的容器指定一个名称。

## 🤖 可用指令

- `/start` - 显示欢迎信息。
- `/help` - 显示帮助信息和指令列表。
- `/settings` - 打开设置菜单，配置聊天模式、AI模型等。
- `/new_chat` - 在多轮模式下开启一个全新的对话。
- `/image <提示词>` - 使用Imagen模型快速生成一张图片。
- `/generate` - 打开高级图片生成面板，可配置风格、尺寸等。
- `/speak <文本>` - 将文本转换为语音。
- `/cancel` - 取消当前正在进行的操作（如设置系统提示）。
