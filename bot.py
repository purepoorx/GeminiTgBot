import logging
import os
import asyncio
import re
import json
import signal
import wave
import io
import sys

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, JobQueue, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode
from telegram.error import BadRequest

import google.genai as genai
from google.genai import types

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- 状态与缓存 ---
user_data = {}
message_text_cache = {}
client: genai.Client = None
USER_SETTINGS_FILE = "data/user_settings.json"
authorized_user_ids = set()

# --- 授权 ---
def load_authorized_users_from_env():
    """从环境变量加载授权用户ID"""
    global authorized_user_ids
    ids_str = os.getenv("AUTHORIZED_USER_IDS")
    if ids_str:
        try:
            # 解析逗号分隔的ID字符串，并转换成整数集合
            authorized_user_ids = {int(user_id.strip()) for user_id in ids_str.split(',')}
            logger.info(f"已从环境变量加载 {len(authorized_user_ids)} 个授权用户。")
        except ValueError:
            logger.error("环境变量 AUTHORIZED_USER_IDS 格式错误，应为逗号分隔的数字ID。")
            authorized_user_ids = set()
    else:
        logger.warning("环境变量 AUTHORIZED_USER_IDS 未设置，将不允许任何用户。")
        authorized_user_ids = set()

def authorized(func):
    """一个装饰器，用于限制只有授权用户才能访问处理函数"""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = None
        if update.effective_user:
            user_id = update.effective_user.id
        
        if user_id not in authorized_user_ids:
            logger.warning(f"未授权的访问尝试，用户ID: {user_id}")
            # 对于回调查询，我们需要回答它，而不是发送新消息
            if update.callback_query:
                await update.callback_query.answer("抱歉，您没有权限执行此操作。", show_alert=True)
            elif update.message:
                await update.message.reply_text("抱歉，您没有权限使用此机器人。")
            return
        return await func(update, context, *args, **kwargs)
    return wrapper

# --- 数据持久化 ---
def save_user_data():
    """将可序列化的用户数据保存到文件"""
    persistable_data = {}
    for user_id, data in user_data.items():
        persistable_data[user_id] = {
            key: value for key, value in data.items()
            if isinstance(value, (str, int, float, bool, list, dict)) and key not in ['chat', 'file_context']
        }
        if 'chat_history' in data and isinstance(data['chat_history'], list):
             pass

    # 确保目录存在
    os.makedirs(os.path.dirname(USER_SETTINGS_FILE), exist_ok=True)
    with open(USER_SETTINGS_FILE, 'w') as f:
        json.dump(persistable_data, f, indent=4)

def load_user_data():
    """从文件加载用户数据"""
    global user_data
    if os.path.exists(USER_SETTINGS_FILE):
        with open(USER_SETTINGS_FILE, 'r') as f:
            try:
                user_data = {int(k): v for k, v in json.load(f).items()}
            except (json.JSONDecodeError, ValueError):
                logger.error("无法解析用户设置文件，将使用空设置。")
                user_data = {}

# --- 辅助函数 ---
def escape_markdown_v2(text: str) -> str:
    """转义 Telegram MarkdownV2 的特殊字符"""
    if not isinstance(text, str):
        text = str(text)
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

def pcm_to_wav_in_memory(pcm_data: bytes, channels=1, sample_width=2, rate=24000) -> io.BytesIO:
    """将原始 PCM 数据编码为内存中的 WAV 文件"""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    wav_buffer.seek(0)
    return wav_buffer

# --- 模型与默认值 ---
DEFAULT_CHAT_MODEL = 'models/gemini-1.5-flash'
DEFAULT_IMAGE_MODEL = 'models/imagen-3.0-generate-002'
DEFAULT_MULTIMODAL_MODEL = 'models/gemini-2.0-flash-preview-image-generation'
DEFAULT_TTS_MODEL = 'models/tts-001'
DEFAULT_TTS_VOICE = 'Zephyr'

PRESET_IMAGE_MODELS = [
    'models/imagen-4.0-generate-preview-06-06',
    'models/imagen-4.0-ultra-generate-preview-06-06',
    'models/imagen-3.0-generate-002',
]

PRESET_MULTIMODAL_MODELS = [
    'models/gemini-2.0-flash-preview-image-generation',
]

PRESET_VOICES = {
    "Zephyr": "Zephyr", "Puck": "Puck", "Charon": "Charon", "Kore": "Kore", "Fenrir": "Fenrir",
    "Leda": "Leda", "Orus": "Orus", "Aoede": "Aoede", "Callirrhoe": "Callirrhoe", "Autonoe": "Autonoe",
    "Enceladus": "Enceladus", "Iapetus": "Iapetus", "Umbriel": "Umbriel", "Algieba": "Algieba",
    "Despina": "Despina", "Erinome": "Erinome", "Algenib": "Algenib", "Rasalgethi": "Rasalgethi",
    "Laomedeia": "Laomedeia", "Achernar": "Achernar", "Alnilam": "Alnilam", "Schedar": "Schedar",
    "Gacrux": "Gacrux", "Pulcherrima": "Pulcherrima", "Achird": "Achird", "Zubenelgenubi": "Zubenelgenubi",
    "Vindemiatrix": "Vindemiatrix", "Sadachbia": "Sadachbia", "Sadaltager": "Sadaltager", "Sulafat": "Sulafat",
}

ASPECT_RATIOS = {
    "1:1": "1:1 (方形)", "4:3": "4:3 (横向)", "3:4": "3:4 (纵向)",
    "16:9": "16:9 (宽屏)", "9:16": "9:16 (竖屏)"
}

IMAGE_GENERATION_KEYWORDS = ["画", "生成图片", "创建一张图片", "draw", "generate image", "create an image"]

# --- 高级图片生成 ---
GENERATE_SETTINGS_KEY = 'generation_settings'
STYLE_SELECT = {
    "none": "无风格", "anime": "动漫", "photorealistic": "写实摄影",
    "cyberpunk": "赛博朋克", "watercolor": "水彩画", "pixel-art": "像素艺术"
}
STYLE_PROMPTS = {
    'none': '', 'anime': ', anime style, vibrant colors, detailed line art',
    'photorealistic': ', photorealistic, 8k, sharp focus, detailed, professional photography',
    'cyberpunk': ', cyberpunk, neon lighting, futuristic city, dystopian',
    'watercolor': ', watercolor painting, soft wash, blended colors',
    'pixel-art': ', pixel art, 16-bit, retro gaming style'
}

async def optimize_prompt_async(prompt: str, negative_prompt: str) -> str:
    """使用 Gemini 优化提示词"""
    if not prompt: return ""
    optimization_text = prompt
    if negative_prompt: optimization_text += f". (排除以下内容: {negative_prompt})"
    system_instruction = "你是一个顶级的AI绘画提示词工程师。你的任务是分析用户提供的包含正面和负面描述的中文草稿，然后创作出一个全新的、艺术性的、详细生动的英文提示词。请将负面描述（例如'排除太阳'）自然地融入到正面描述中，通过使用替代性或描述性词语来实现排除效果（例如，使用'overcast sky'或'cloudy day'来代替'no sun'）。最终的输出必须是一段流畅、连贯、可以直接用于AI绘画的英文描述，**绝对不能包含** 'exclude', 'without', 'no' 等直接的否定词或括号。你的目标是创造一个画面，而不是列出指令。"
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=optimization_text,
            config=genai.types.GenerateContentConfig(system_instruction=system_instruction)
        )
        return response.text
    except Exception as e:
        logger.error(f"提示词优化失败: {e}")
        return prompt

@authorized
async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /generate 命令，显示高级图片生成面板"""
    context.user_data[GENERATE_SETTINGS_KEY] = {
        "prompt": "", "negative_prompt": "", "style": "none",
        "aspect_ratio": "1:1", "optimize": True, "is_waiting_for": None,
    }
    await send_generate_panel(update, context)

async def send_generate_panel(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """发送或编辑高级图片生成面板"""
    settings = context.user_data.get(GENERATE_SETTINGS_KEY, {})
    style_text = STYLE_SELECT.get(settings.get('style', 'none'), "无风格")
    optimize_text = "✅ 开启" if settings.get('optimize') else "❌ 关闭"
    prompt_text = f"`{settings.get('prompt')}`" if settings.get('prompt') else "_尚未设置_"
    negative_prompt_text = f"`{settings.get('negative_prompt')}`" if settings.get('negative_prompt') else "_尚未设置_"
    text = (
        f"🎨 *高级图片生成面板*\n\n"
        f"1️⃣ *主提示词*: {prompt_text}\n"
        f"2️⃣ *负面提示词*: {negative_prompt_text}\n"
        f"🎨 *艺术风格*: {style_text}\n"
        f"🖼️ *宽高比*: {settings.get('aspect_ratio')}\n"
        f"✨ *提示词优化*: {optimize_text}\n\n"
        f"请通过下方的按钮进行设置，完成后点击“生成图片”。"
    )
    keyboard = [
        [InlineKeyboardButton("1️⃣ 设置主提示词", callback_data="generate_action_set_main_prompt"), InlineKeyboardButton("2️⃣ 设置负面提示词", callback_data="generate_action_set_negative_prompt")],
        [InlineKeyboardButton(f"🎨 风格: {style_text}", callback_data="generate_menu_style"), InlineKeyboardButton(f"🖼️ 宽高比: {settings.get('aspect_ratio')}", callback_data="generate_menu_aspect_ratio")],
        [InlineKeyboardButton(f"✨ 优化: {optimize_text}", callback_data="generate_toggle_optimize")],
        [InlineKeyboardButton("✅ 生成图片!", callback_data="generate_do_generate")],
        [InlineKeyboardButton("❌ 关闭", callback_data="generate_close")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    if query:
        try:
            await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
        except BadRequest as e:
            if "Message is not modified" in str(e):
                pass 
            else:
                raise 
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)

@authorized
async def generate_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理高级图片生成面板的所有回调"""
    query = update.callback_query
    await query.answer()
    action = query.data
    settings = context.user_data.get(GENERATE_SETTINGS_KEY)
    if not settings:
        await query.edit_message_text("抱歉，生成会话已过期，请重新发起 /generate。")
        return

    if action == "generate_close":
        del context.user_data[GENERATE_SETTINGS_KEY]
        await query.edit_message_text("✅ 高级图片生成面板已关闭。")
    elif action == "generate_action_set_main_prompt":
        settings['is_waiting_for'] = 'main_prompt'
        await query.edit_message_text("✍️ 请直接发送您的 **主提示词**。", parse_mode=ParseMode.MARKDOWN_V2)
    elif action == "generate_action_set_negative_prompt":
        settings['is_waiting_for'] = 'negative_prompt'
        await query.edit_message_text("✍️ 请直接发送您的 **负面提示词** \\(不希望出现的内容\\)\\.", parse_mode=ParseMode.MARKDOWN_V2)
    elif action == "generate_toggle_optimize":
        settings['optimize'] = not settings.get('optimize', False)
        await send_generate_panel(update, context, query)
    elif action == "generate_menu_style":
        keyboard = [[InlineKeyboardButton(text, callback_data=f"generate_set:style:{key}")] for key, text in STYLE_SELECT.items()]
        keyboard.append([InlineKeyboardButton("⬅️ 返回", callback_data="generate_back_to_main")])
        await query.edit_message_text("🎨 请选择一个艺术风格:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif action == "generate_menu_aspect_ratio":
        keyboard = [[InlineKeyboardButton(text, callback_data=f"generate_set:aspect_ratio:{key}")] for key, text in ASPECT_RATIOS.items()]
        keyboard.append([InlineKeyboardButton("⬅️ 返回", callback_data="generate_back_to_main")])
        await query.edit_message_text("🖼️ 请选择宽高比:", reply_markup=InlineKeyboardMarkup(keyboard))
    elif action == "generate_back_to_main":
        await send_generate_panel(update, context, query)
    elif action == "generate_do_generate":
        await query.edit_message_text("✅ 已收到请求，正在后台处理…")
        asyncio.create_task(generate_image_task(update, context, query.from_user.id, query.message.message_id))

async def generate_image_task(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int, message_id: int):
    """后台执行图片生成任务"""
    settings = context.user_data.get(GENERATE_SETTINGS_KEY)
    if not settings or not settings.get("prompt"):
        await context.bot.edit_message_text("错误：没有找到提示词，请重新开始。", chat_id=update.effective_chat.id, message_id=message_id)
        return

    final_prompt = settings["prompt"]
    try:
        if settings.get("optimize"):
            try:
                await context.bot.edit_message_text("✨ 正在优化提示词…", chat_id=update.effective_chat.id, message_id=message_id, parse_mode=ParseMode.MARKDOWN_V2)
                final_prompt = await optimize_prompt_async(settings["prompt"], settings.get("negative_prompt", ""))
            except Exception as e:
                logger.error(f"提示词优化失败，将使用原始提示词: {e}")
        
        style_prompt = STYLE_PROMPTS.get(settings.get("style", "none"), "")
        if final_prompt is None:
            final_prompt = ""
        if style_prompt: final_prompt += style_prompt

        await context.bot.edit_message_text(f"🎨 正在生成图片…\n\n*最终提示词*:\n`{escape_markdown_v2(final_prompt)}`", chat_id=update.effective_chat.id, message_id=message_id, parse_mode=ParseMode.MARKDOWN_V2)
        
        image_model = user_data.get(user_id, {}).get('image_model', DEFAULT_IMAGE_MODEL)
        response = await client.aio.models.generate_images(
            model=image_model, prompt=final_prompt,
            config=genai.types.GenerateImagesConfig(
                aspect_ratio=settings.get("aspect_ratio", "1:1"),
            )
        )

        if response.generated_images:
            img_bytes = response.generated_images[0].image.image_bytes
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=img_bytes, caption=f"🖼️ “{settings['prompt']}”")
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=message_id)
        else:
            await context.bot.edit_message_text("抱歉，无法生成图片。", chat_id=update.effective_chat.id, message_id=message_id)
    except Exception as e:
        logger.error(f"高级图片生成失败: {e}", exc_info=True)
        error_message = f"❌ 生成失败: {escape_markdown_v2(str(e))}"
        await context.bot.edit_message_text(error_message, chat_id=update.effective_chat.id, message_id=message_id, parse_mode=ParseMode.MARKDOWN_V2)
    finally:
        if GENERATE_SETTINGS_KEY in context.user_data:
            del context.user_data[GENERATE_SETTINGS_KEY]

@authorized
async def generate_selection_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理高级图片生成中的选项选择"""
    query = update.callback_query
    await query.answer()
    
    parts = query.data.split(':', 2)
    action_type = parts[1]
    value = parts[2]

    settings = context.user_data.get(GENERATE_SETTINGS_KEY)
    if not settings:
        await query.edit_message_text("抱歉，生成会话已过期，请重新发起 /generate。")
        return

    if action_type == "style":
        settings['style'] = value
    elif action_type == "aspect_ratio":
        settings['aspect_ratio'] = value

    await send_generate_panel(update, context, query)

@authorized
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """发送 /start 指令时的欢迎消息"""
    welcome_message = (
        "欢迎使用 Gemini AI 机器人！\n\n"
        "我可以为您提供多种 AI 服务。使用 /settings 来配置您的机器人，"
        "使用 /help 查看所有可用指令。"
    )
    await update.message.reply_text(welcome_message)

@authorized
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """发送 /help 指令时的帮助消息"""
    help_text = (
        "📖 *可用指令列表* 📖\n\n"
        "*核心功能*\n"
        "`/start` \\- 显示欢迎信息\n"
        "`/help` \\- 显示此帮助信息\n"
        "`/settings` \\- 打开设置菜单\n"
        "`/new_chat` \\- 在多轮模式下开启新对话\n\n"
        "*图片生成: 两种模式*\n"
        "1️⃣ *专业模式: Imagen*: 生成高质量、特定风格的图片。\n"
        "   `/image <英文提示词>` \\- 快速生成\n"
        "   `/generate` \\- 打开高级面板，可配置风格、尺寸等\n"
        "2️⃣ *对话模式: Gemini*: 在聊天中快速创作和编辑图片。\n"
        "   在对话中包含“画图”、“生成图片”等关键词。\n"
        "   回复一张图片并提出修改要求，如“把这个改成红色”。\n\n"
        "*其他交互*\n"
        "`/speak <要转换的文本>` \\- 快速生成语音 \\(限250字符\\)\n"
        "发送文件或图片，然后向我提问即可进行文件问答。"
    )
    await update.message.reply_markdown_v2(help_text)

@authorized
async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """开启一个新对话，清空历史记录"""
    user_id = update.effective_user.id
    chat_mode = user_data.get(user_id, {}).get('chat_mode', 'multi_turn')

    if chat_mode == 'single_turn':
        await update.message.reply_text("当前为单轮模式，无需开启新对话。每次对话都是全新的。")
        return

    if user_id in user_data:
        cleared = False
        if 'chat' in user_data[user_id]:
            del user_data[user_id]['chat']
            cleared = True
        if 'file_context' in user_data[user_id]:
            del user_data[user_id]['file_context']
            cleared = True
        
        if cleared:
            save_user_data()
            await update.message.reply_text("✨ 新的对话已经开始，之前的上下文（包括历史记录和文件）已被清除。")
        else:
            await update.message.reply_text("您还没有开始任何多轮对话。")
    else:
        await update.message.reply_text("您还没有开始任何多轮对话。")

@authorized
async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """取消当前操作"""
    user_id = update.effective_user.id
    if user_data.get(user_id, {}).get('waiting_for'):
        del user_data[user_id]['waiting_for']
        save_user_data()
        await update.message.reply_text("操作已取消。")
        await settings_command(update, context)
    else:
        await update.message.reply_text("当前没有可以取消的操作。")

@authorized
async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/image 指令的快捷方式"""
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("请提供图片描述。\n用法: `/image 一只正在太空漫步的猫`")
        return
    prompt = ' '.join(context.args)
    
    image_model = user_data.get(user_id, {}).get('image_model', DEFAULT_IMAGE_MODEL)
    placeholder_message = await update.message.reply_text(f"🎨 正在使用 `{image_model}` 生成图片…")

    try:
        response = await client.aio.models.generate_images(
            model=image_model,
            prompt=prompt,
            config=genai.types.GenerateImagesConfig()
        )
        if response.generated_images:
            img_bytes = response.generated_images[0].image.image_bytes
            await update.message.reply_photo(photo=img_bytes, caption=f"🖼️ \"{prompt}\"")
            await context.bot.delete_message(chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)
        else:
            await context.bot.edit_message_text("抱歉，无法生成图片。", chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)
    except Exception as e:
        logger.error(f"生成图片时出错: {e}")
        await context.bot.edit_message_text(f"抱歉，生成图片时出错: {e}", chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)

@authorized
async def speak_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/speak 指令的快捷方式"""
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("请提供要转换为语音的文本。\n用法: `/speak 你好，世界！`")
        return
    text_to_speak = ' '.join(context.args)

    if len(text_to_speak) > 250:
        await update.message.reply_text("抱歉，语音转换的文本长度不能超过 250 个字符。")
        return

    tts_model = user_data.get(user_id, {}).get('tts_model', DEFAULT_TTS_MODEL)
    user_voice = user_data.get(user_id, {}).get('voice', DEFAULT_TTS_VOICE)
    placeholder_message = await update.message.reply_text(f"🎤 正在使用 `{tts_model}` 和 `{user_voice}` 声线为您生成语音…")

    try:
        config = genai.types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai.types.SpeechConfig(
                voice_config=genai.types.VoiceConfig(
                    prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(voice_name=user_voice)
                )
            )
        )
        response = await client.aio.models.generate_content(
            model=tts_model, contents=text_to_speak, config=config
        )
        if response.candidates and response.candidates[0].content.parts[0].inline_data.data:
            pcm_data = response.candidates[0].content.parts[0].inline_data.data
            wav_data = pcm_to_wav_in_memory(pcm_data)
            await update.message.reply_voice(voice=wav_data)
            await context.bot.delete_message(chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)
        else:
            await context.bot.edit_message_text("抱歉，无法生成语音。", chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)
    except Exception as e:
        logger.error(f"生成语音时出错: {e}")
        await context.bot.edit_message_text(f"抱歉，生成语音时出错: {e}", chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)

# --- 设置菜单 ---

@authorized
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """显示主设置菜单"""
    user_id = update.effective_user.id
    if user_id not in user_data: user_data[user_id] = {}

    chat_mode = user_data[user_id].get('chat_mode', 'multi_turn')
    thinking_mode = user_data[user_id].get('thinking_mode', False)
    chat_model = user_data.get(user_id, {}).get('chat_model', DEFAULT_CHAT_MODEL)
    image_model = user_data.get(user_id, {}).get('image_model', DEFAULT_IMAGE_MODEL)
    multimodal_model = user_data.get(user_id, {}).get('multimodal_model', DEFAULT_MULTIMODAL_MODEL)
    tts_model = user_data.get(user_id, {}).get('tts_model', DEFAULT_TTS_MODEL)
    current_voice = user_data.get(user_id, {}).get('voice', DEFAULT_TTS_VOICE)
    
    status_text = (
        f"当前聊天模式: *{'多轮' if chat_mode == 'multi_turn' else '单轮'}*\n"
        f"当前思考过程: *{'开启' if thinking_mode else '关闭'}*\n"
        f"当前聊天模型: `{escape_markdown_v2(chat_model)}`\n"
        f"当前专业图片模型: Imagen: `{escape_markdown_v2(image_model)}`\n"
        f"当前多模态模型: Gemini: `{escape_markdown_v2(multimodal_model)}`\n"
        f"当前语音模型: `{escape_markdown_v2(tts_model)}`\n"
        f"当前语音声线: `{escape_markdown_v2(current_voice)}`"
    )
    text = f"⚙️ *机器人当前设置*\n\n{status_text}\n\n请选择要配置的项目："
    
    keyboard = [
        [InlineKeyboardButton("🤖 模型设置", callback_data="settings_menu_models")],
        [InlineKeyboardButton(f"切换到 {'单轮' if chat_mode == 'multi_turn' else '多轮'} 模式", callback_data="settings_action_toggle_chat_mode")],
        [InlineKeyboardButton(f"思考过程: {'点击关闭' if thinking_mode else '点击开启'}", callback_data="settings_action_toggle_thinking_mode")],
        [InlineKeyboardButton("🗣️ 语音声线设置", callback_data="settings_menu_voice")],
        [InlineKeyboardButton("📝 设置系统提示", callback_data="settings_action_set_prompt")],
        [InlineKeyboardButton("❌ 关闭菜单", callback_data="settings_action_close")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query = update.callback_query
    if query:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN_V2)

@authorized
async def settings_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理所有来自设置菜单的回调"""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    action = query.data
    if user_id not in user_data: user_data[user_id] = {}

    if action == "settings_action_toggle_chat_mode":
        current_mode = user_data[user_id].get('chat_mode', 'multi_turn')
        new_mode = 'single_turn' if current_mode == 'multi_turn' else 'multi_turn'
        user_data[user_id]['chat_mode'] = new_mode
        if 'chat' in user_data[user_id]:
            del user_data[user_id]['chat']
        if 'file_context' in user_data[user_id]:
            del user_data[user_id]['file_context']
        save_user_data()
        await settings_command(update, context)
    elif action == "settings_action_toggle_thinking_mode":
        current_mode = user_data[user_id].get('thinking_mode', False)
        new_mode = not current_mode
        user_data[user_id]['thinking_mode'] = new_mode
        save_user_data()
        await settings_command(update, context)
    elif action == "settings_action_close":
        await query.edit_message_text("✅ 设置菜单已关闭。")
        return
    elif action == "settings_menu_models":
        await models_menu(update, context)
    elif action == "settings_menu_voice":
        await voice_menu(update, context)
    elif action == "settings_action_set_prompt":
        user_data[user_id]['waiting_for'] = 'system_prompt'
        current_prompt = user_data.get(user_id, {}).get('system_prompt')
        prompt_text = f"当前的系统提示为:\n`{escape_markdown_v2(current_prompt)}`\n\n请直接回复新的提示词以进行修改，或发送 /cancel 取消。" if current_prompt else "您还没有设置系统提示。请直接回复您想设置的提示词。"
        await query.edit_message_text(prompt_text, parse_mode=ParseMode.MARKDOWN_V2)

@authorized
async def models_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """显示模型设置子菜单"""
    query = update.callback_query
    user_id = query.from_user.id
    chat_model = user_data.get(user_id, {}).get('chat_model', DEFAULT_CHAT_MODEL)
    image_model = user_data.get(user_id, {}).get('image_model', DEFAULT_IMAGE_MODEL)
    multimodal_model = user_data.get(user_id, {}).get('multimodal_model', DEFAULT_MULTIMODAL_MODEL)
    tts_model = user_data.get(user_id, {}).get('tts_model', DEFAULT_TTS_MODEL)
    text = f"🤖 *模型设置*\n\n当前聊天模型: `{escape_markdown_v2(chat_model)}`\n当前专业图片模型: Imagen: `{escape_markdown_v2(image_model)}`\n当前多模态模型: Gemini: `{escape_markdown_v2(multimodal_model)}`\n当前语音模型: `{escape_markdown_v2(tts_model)}`\n\n请选择要更改的模型类型："
    keyboard = [
        [InlineKeyboardButton("💬 聊天模型", callback_data="model_list_chat")],
        [InlineKeyboardButton("🖼️ 专业图片模型: Imagen", callback_data="model_list_image")],
        [InlineKeyboardButton("🎨 多模态模型: Gemini", callback_data="model_list_multimodal")],
        [InlineKeyboardButton("🔊 语音模型", callback_data="model_list_tts")],
        [InlineKeyboardButton("⬅️ 返回主菜单", callback_data="settings_main_menu")],
    ]
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)

@authorized
async def voice_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """显示语音声线选择菜单"""
    query = update.callback_query
    user_id = query.from_user.id
    current_voice = user_data.get(user_id, {}).get('voice', DEFAULT_TTS_VOICE)
    text = f"🗣️ *语音声线设置*\n\n当前声线: `{escape_markdown_v2(current_voice)}`\n\n请选择您喜欢的语音声线："
    keyboard = [[InlineKeyboardButton(f"✅ {v}" if v == current_voice else v, callback_data=f"voice_set_{v}")] for v in PRESET_VOICES.values()]
    keyboard.append([InlineKeyboardButton("⬅️ 返回主菜单", callback_data="settings_main_menu")])
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)

@authorized
async def model_list_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """显示特定类型的模型列表"""
    query = update.callback_query
    model_type = query.data.split('_')[-1]
    await query.edit_message_text(f"正在查询可用的 *{model_type}* 模型…", parse_mode=ParseMode.MARKDOWN_V2)
    try:
        all_models = await client.aio.models.list()
        if model_type == 'chat':
            filtered_models = [m for m in all_models if 'generateContent' in m.supported_actions and 'tts' not in m.name and 'imagen' not in m.name]
        elif model_type == 'image':
            from types import SimpleNamespace
            filtered_models = [SimpleNamespace(name=name) for name in PRESET_IMAGE_MODELS]
        elif model_type == 'multimodal':
            # 由于当前只有一个可用的多模态模型，直接为用户设置并告知
            user_id = query.from_user.id
            if user_id not in user_data: user_data[user_id] = {}
            user_data[user_id]['multimodal_model'] = DEFAULT_MULTIMODAL_MODEL
            save_user_data()
            await query.edit_message_text(
                f"✅ 已为您设置唯一可用的多模态模型:\n`{escape_markdown_v2(DEFAULT_MULTIMODAL_MODEL)}`",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ 返回模型设置", callback_data="settings_menu_models")]]),
                parse_mode=ParseMode.MARKDOWN_V2
            )
            return
        elif model_type == 'tts':
            filtered_models = [m for m in all_models if 'tts' in m.name]
        else:
            filtered_models = []

        if not filtered_models:
            text = f"未找到可用的 *{model_type}* 模型。"
            keyboard = [[InlineKeyboardButton("⬅️ 返回模型设置", callback_data="settings_menu_models")]]
        else:
            text = f"请为 *{model_type}* 任务选择一个模型:"
            keyboard = [[InlineKeyboardButton(m.name.replace('models/', ''), callback_data=f"model_set_{model_type}_{m.name}")] for m in filtered_models]
            keyboard.append([InlineKeyboardButton("⬅️ 返回模型设置", callback_data="settings_menu_models")])
        
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logger.error(f"查询模型列表时出错: {e}")
        await query.edit_message_text("查询模型列表失败，请稍后再试。")

@authorized
async def set_model_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理模型选择的回调"""
    query = update.callback_query
    await query.answer()
    parts = query.data.split('_')
    model_type, model_name = parts[2], '_'.join(parts[3:])
    user_id = query.from_user.id
    if user_id not in user_data: user_data[user_id] = {}
    user_data[user_id][f'{model_type}_model'] = model_name
    if model_type == 'chat' and 'chat' in user_data[user_id]: del user_data[user_id]['chat']
    save_user_data()
    await models_menu(update, context)

@authorized
async def set_voice_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理语音声线选择的回调"""
    query = update.callback_query
    await query.answer()
    voice_id = query.data.split('_')[-1]
    user_id = query.from_user.id
    if user_id not in user_data: user_data[user_id] = {}
    user_data[user_id]['voice'] = voice_id
    save_user_data()
    await voice_menu(update, context)

@authorized
async def tts_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理“转换语音”按钮的回调"""
    query = update.callback_query
    await query.answer("抱歉，此按钮已禁用。请使用 /speak 命令。", show_alert=True)

# --- 消息与文件处理器 ---


async def handle_gemini_image_generation(update: Update, context: ContextTypes.DEFAULT_TYPE, image_bytes: bytearray = None, text: str = None) -> None:
    """使用 Gemini 的多模态能力进行对话式图片生成或编辑"""
    user_id = update.effective_user.id
    placeholder_message = await update.message.reply_text("🎨 正在进行多模态创作…")

    try:
        # 1. 构建请求内容
        contents = []
        if image_bytes:
            image_part = types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes))
            contents.append(image_part)
        
        if text:
            contents.append(text)

        # 2. API 调用
        # 使用用户配置的多模态模型
        multimodal_model = user_data.get(user_id, {}).get('multimodal_model', DEFAULT_MULTIMODAL_MODEL)

        response = await client.aio.models.generate_content(
            model=multimodal_model,
            contents=contents,
            config=types.GenerateContentConfig(
              response_modalities=['TEXT', 'IMAGE']
            )
        )

        # 3. 处理响应
        text_response = ""
        image_parts = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_response += part.text
                elif part.inline_data:
                    image_parts.append(part.inline_data.data)

        await placeholder_message.delete()
        
        if text_response:
            await update.message.reply_text(text_response)
        
        for img_bytes in image_parts:
            await update.message.reply_photo(photo=img_bytes)

    except Exception as e:
        logger.error(f"Gemini 图片生成失败: {e}", exc_info=True)
        await placeholder_message.edit_text(f"抱歉，图片创作失败: {escape_markdown_v2(str(e))}", parse_mode=ParseMode.MARKDOWN_V2)


@authorized
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理用户发送的文件（图片、文档等）"""
    user_id = update.effective_user.id
    message = update.message
    file_to_process = message.document or (message.photo[-1] if message.photo else None)

    if not file_to_process:
        await message.reply_text("无法识别的文件类型。")
        return

    placeholder_message = await message.reply_text("正在处理文件…")

    try:
        file_obj = await context.bot.get_file(file_to_process.file_id)
        file_bytes = await file_obj.download_as_bytearray()

        # 准备上传到 Gemini
        # 注意：这里我们假设所有文件都可以作为 'image/jpeg' 或 'text/plain' 处理
        # 在实际应用中，你可能需要更复杂的 MIME 类型检测
        if message.photo:
            mime_type = "image/jpeg" # 照片直接指定MIME类型
        else:
            mime_type = file_to_process.mime_type or "application/octet-stream"
        
        if "image" in mime_type:
            file_for_gemini = types.Part(inline_data=types.Blob(mime_type=mime_type, data=file_bytes))
            prompt = "请描述这张图片的内容。"
        elif "text" in mime_type:
            # 假设是 UTF-8 编码
            text_content = file_bytes.decode('utf-8', errors='ignore')
            file_for_gemini = types.Part(text=text_content)
            prompt = "请总结这个文件的内容。"
        else:
            # 对于其他文件类型，我们可以尝试将其作为纯文本处理
            try:
                text_content = file_bytes.decode('utf-8', errors='ignore')
                file_for_gemini = types.Part(text=text_content)
                prompt = "请分析这个文件的内容。"
            except Exception:
                 await placeholder_message.edit_text("不支持的文件类型，无法作为文本解析。")
                 return

        if user_id not in user_data: user_data[user_id] = {}
        user_data[user_id]['file_context'] = file_for_gemini
        save_user_data()

        await placeholder_message.edit_text("✅ 文件处理完成！现在您可以就这个文件向我提问了。")

    except Exception as e:
        logger.error(f"处理文件时出错: {e}", exc_info=True)
        await placeholder_message.edit_text(f"处理文件时出错: {e}")


@authorized
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理用户的文本消息"""
    user_id = update.effective_user.id
    message = update.message
    user_message = message.text

    if user_id not in user_data: user_data[user_id] = {}

    # --- 检查是否在等待特定输入（如设置提示词） ---
    if user_data.get(user_id, {}).get('waiting_for') == 'system_prompt':
        user_data[user_id]['system_prompt'] = user_message
        user_data[user_id]['waiting_for'] = None
        save_user_data()
        if 'chat' in user_data[user_id]: del user_data[user_id]['chat']
        await message.reply_text(f"✅ 系统提示词已更新为: \"{user_message}\"")
        return

    gen_settings = context.user_data.get(GENERATE_SETTINGS_KEY)
    if gen_settings and gen_settings.get('is_waiting_for'):
        wait_type = gen_settings['is_waiting_for']
        if wait_type == 'main_prompt':
            gen_settings['prompt'] = user_message
            await message.reply_text("✅ 主提示词已更新。")
        elif wait_type == 'negative_prompt':
            gen_settings['negative_prompt'] = user_message
            await message.reply_text("✅ 负面提示词已更新。")
        
        gen_settings['is_waiting_for'] = None
        await send_generate_panel(update, context)
        return

    # --- 多模态意图检测 ---
    # 场景1: 回复一张图片
    if message.reply_to_message and message.reply_to_message.photo:
        file_id = message.reply_to_message.photo[-1].file_id
        file_obj = await context.bot.get_file(file_id)
        image_bytes = await file_obj.download_as_bytearray()
        await handle_gemini_image_generation(update, context, image_bytes=image_bytes, text=user_message)
        return
        
    # 场景2: 仅通过文字关键词触发
    if user_message and any(keyword in user_message.lower() for keyword in IMAGE_GENERATION_KEYWORDS):
        await handle_gemini_image_generation(update, context, text=user_message)
        return
        
    # --- 纯文本聊天逻辑 ---
    chat_mode = user_data[user_id].get('chat_mode', 'multi_turn')
    placeholder_message = await message.reply_text("🤔 正在思考中…")

    try:
        full_response = ""
        thought_summary = ""
        last_edit_time = 0
        edit_interval = 1.0  # Slower update interval to reduce API calls

        chat_model = user_data.get(user_id, {}).get('chat_model')
        if not chat_model:
            await placeholder_message.edit_text("您还没有配置聊天模型。请使用 /settings 命令选择一个聊天模型后才能开始对话。")
            return

        system_instruction_text = user_data[user_id].get('system_prompt')
        thinking_mode = user_data.get(user_id, {}).get('thinking_mode', False)

        # --- Build Content ---
        user_content_parts = []
        if user_message:
            user_content_parts.append(user_message)

        file_context_part = user_data.get(user_id, {}).get('file_context')
        if file_context_part:
            user_content_parts.insert(0, file_context_part)
            # 在单轮模式下，使用后立即清除文件上下文
            if chat_mode == 'single_turn':
                del user_data[user_id]['file_context']
                save_user_data()

        # --- API Call Config ---
        thinking_config = types.ThinkingConfig(include_thoughts=True) if thinking_mode else None
        request_config = genai.types.GenerateContentConfig(thinking_config=thinking_config)
        session_creation_config = genai.types.GenerateContentConfig(
             system_instruction=genai.types.Content(parts=[genai.types.Part(text=system_instruction_text)]) if system_instruction_text else None
        )

        # --- API Call ---
        if chat_mode == 'multi_turn':
            if 'chat' not in user_data[user_id] or user_data[user_id].get('chat_model_name') != chat_model:
                user_data[user_id]['chat'] = client.aio.chats.create(model=chat_model, config=session_creation_config, history=[])
                user_data[user_id]['chat_model_name'] = chat_model
            chat = user_data[user_id]['chat']
            response_stream = await chat.send_message_stream(user_content_parts, config=request_config)
        else: # single_turn
            request_config.system_instruction = session_creation_config.system_instruction
            response_stream = await client.aio.models.generate_content_stream(model=chat_model, contents=user_content_parts, config=request_config)

        # --- Stream Processing ---
        async for chunk in response_stream:
            if chunk.candidates:
                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if getattr(part, 'thought', False):
                                thought_summary += part.text
                            elif part.text:
                                full_response += part.text
            
            current_time = asyncio.get_event_loop().time()
            if current_time - last_edit_time > edit_interval:
                raw_display_text = ""
                if thought_summary:
                    raw_display_text += f"🤔 思考摘要:\n{thought_summary}\n\n---\n"
                raw_display_text += f"{full_response} ▌"
                
                display_text = escape_markdown_v2(raw_display_text)
                if len(display_text) > 4000:
                    display_text = "[...]\n" + display_text[-3995:]

                try:
                    await context.bot.edit_message_text(
                        display_text,
                        chat_id=placeholder_message.chat_id,
                        message_id=placeholder_message.message_id,
                        parse_mode=ParseMode.MARKDOWN_V2
                    )
                    last_edit_time = current_time
                except BadRequest as e:
                    if "Message is not modified" not in str(e):
                        logger.warning(f"Stream edit failed (but not a 'not modified' error): {e}")


        # --- Final Update ---
        message_id = placeholder_message.message_id
        final_response_text = full_response if full_response.strip() else "模型没有返回任何内容。"

        # If thinking is enabled and we have a summary, prepare for interactive display
        if thought_summary and thinking_mode:
            if 'thought_caches' not in context.bot_data:
                context.bot_data['thought_caches'] = {}
            context.bot_data['thought_caches'][message_id] = {'thought': thought_summary, 'response': final_response_text}
            context.job_queue.run_once(cleanup_thought_cache, 300, data={'message_id': message_id}, name=f"cleanup_{message_id}")
            
            keyboard = [[InlineKeyboardButton("🤔 显示思考摘要", callback_data=f"toggle_thought:{message_id}:show")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await context.bot.edit_message_text(
                text=escape_markdown_v2(final_response_text),
                chat_id=placeholder_message.chat_id,
                message_id=message_id,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=reply_markup
            )
        else:
            # For non-thinking mode or if no thoughts were generated
            escaped_final_text = escape_markdown_v2(final_response_text)
            if len(escaped_final_text) > 4096:
                await context.bot.delete_message(chat_id=placeholder_message.chat_id, message_id=message_id)
                parts = [escaped_final_text[i:i + 4096] for i in range(0, len(escaped_final_text), 4096)]
                for part in parts:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=part,
                        parse_mode=ParseMode.MARKDOWN_V2
                    )
            else:
                 await context.bot.edit_message_text(
                    text=escaped_final_text,
                    chat_id=placeholder_message.chat_id,
                    message_id=message_id,
                    parse_mode=ParseMode.MARKDOWN_V2
                )

    except Exception as e:
        logger.error(f"处理消息时出错: {e}", exc_info=True)
        error_message = f"抱歉，处理您的请求时遇到了一个错误: {type(e).__name__}"
        try:
            await context.bot.edit_message_text(error_message, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)
        except BadRequest:
             await update.message.reply_text(error_message) # Fallback if editing fails
    finally:
        # 无论是单轮还是多轮，文件上下文在使用后都应被清除，
        # 因为AI模型会在其对话历史中记住图片内容。
        if file_context_part and user_id in user_data and 'file_context' in user_data[user_id]:
            del user_data[user_id]['file_context']
            save_user_data()

async def cleanup_thought_cache(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Job to remove an expired thought cache."""
    job = context.job
    message_id = job.data['message_id']
    if 'thought_caches' in context.bot_data and message_id in context.bot_data['thought_caches']:
        del context.bot_data['thought_caches'][message_id]
        logger.info(f"Cleaned up expired thought cache for message_id: {message_id}")

async def set_bot_commands(application: Application):
    """设置机器人的命令列表，以便在Telegram中显示提示"""
    commands = [
        BotCommand("start", "显示欢迎信息"),
        BotCommand("help", "显示帮助和指令列表"),
        BotCommand("settings", "打开设置菜单"),
        BotCommand("new_chat", "在多轮模式下开启新对话"),
        BotCommand("image", "快速生成一张图片"),
        BotCommand("generate", "打开高级图片生成面板"),
        BotCommand("speak", "将文本转换为语音"),
        BotCommand("cancel", "取消当前操作"),
    ]
    await application.bot.set_my_commands(commands)

@authorized
async def toggle_thought_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理显示/隐藏思考摘要的按钮回调"""
    query = update.callback_query
    await query.answer()

    try:
        _, message_id_str, action = query.data.split(':')
        message_id = int(message_id_str)
    except (ValueError, IndexError):
        await query.edit_message_text("无效的回调数据。")
        return

    cached_data = context.bot_data.get('thought_caches', {}).get(message_id)
    if not cached_data:
        await query.edit_message_text("抱歉，这条消息的思考摘要已过期。")
        return

    thought = cached_data['thought']
    response = cached_data['response']
    
    new_text = ""
    keyboard = None

    if action == 'show':
        new_text = f"🤔 思考摘要:\n{thought}\n\n---\n{response}"
        keyboard = [[InlineKeyboardButton("🫣 隐藏思考摘要", callback_data=f"toggle_thought:{message_id}:hide")]]
    elif action == 'hide':
        new_text = response
        keyboard = [[InlineKeyboardButton("🤔 显示思考摘要", callback_data=f"toggle_thought:{message_id}:show")]]

    try:
        # Check if the combined text will be too long
        if action == 'show' and len(escape_markdown_v2(new_text)) > 4096:
            # If too long, send the thought process as new messages
            await query.edit_message_text(
                text=escape_markdown_v2(response),
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("✅ 思考摘要已在下方发送", callback_data="noop")]]),
                parse_mode=ParseMode.MARKDOWN_V2
            )
            
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="🤔 *思考摘要*",
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_to_message_id=query.message.message_id
            )

            thought_parts = [thought[i:i + 4096] for i in range(0, len(thought), 4096)]
            for part in thought_parts:
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=escape_markdown_v2(part),
                    parse_mode=ParseMode.MARKDOWN_V2
                )
        else:
            # If not too long, edit the message as usual
            await query.edit_message_text(
                text=escape_markdown_v2(new_text),
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN_V2
            )
    except BadRequest as e:
        if "Message is not modified" not in str(e):
            logger.error(f"编辑消息以切换思考摘要时出错: {e}")
            await query.message.reply_text(f"抱歉，切换时发生错误: {e}")

# --- 主函数 ---
def main() -> None:
    """启动机器人"""
    load_dotenv()
    load_user_data()
    load_authorized_users_from_env()
    global client
    
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_base_url = os.getenv("GEMINI_BASE_URL")

    if not telegram_token or not gemini_api_key:
        logger.error("错误: TELEGRAM_BOT_TOKEN 和 GEMINI_API_KEY 必须在 .env 文件中设置。")
        return

    http_options = genai.types.HttpOptions(base_url=gemini_base_url) if gemini_base_url else None
    if http_options: logger.info(f"使用自定义 Gemini Base URL: {gemini_base_url}")

    client = genai.Client(api_key=gemini_api_key, http_options=http_options)
    
    job_queue = JobQueue()
    application = Application.builder().token(telegram_token).job_queue(job_queue).post_init(set_bot_commands).build()

    # 注册指令处理器
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("new_chat", new_chat_command))
    application.add_handler(CommandHandler("image", image_command))
    application.add_handler(CallbackQueryHandler(lambda u, c: u.callback_query.answer(), pattern="^noop$"))
    application.add_handler(CommandHandler("generate", generate_command))
    application.add_handler(CommandHandler("speak", speak_command))
    application.add_handler(CommandHandler("cancel", cancel_command))

    # 注册消息和文件处理器
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler((filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND, handle_file))
    
    # 注册回调处理器
    application.add_handler(CallbackQueryHandler(settings_callback_handler, pattern="^settings_action_"))
    application.add_handler(CallbackQueryHandler(settings_command, pattern="^settings_main_menu$"))
    application.add_handler(CallbackQueryHandler(models_menu, pattern="^settings_menu_models$"))
    application.add_handler(CallbackQueryHandler(voice_menu, pattern="^settings_menu_voice$"))
    application.add_handler(CallbackQueryHandler(model_list_menu, pattern="^model_list_"))
    application.add_handler(CallbackQueryHandler(set_model_callback_handler, pattern="^model_set_"))
    application.add_handler(CallbackQueryHandler(set_voice_callback_handler, pattern="^voice_set_"))
    application.add_handler(CallbackQueryHandler(tts_callback_handler, pattern="^tts_"))
    application.add_handler(CallbackQueryHandler(generate_callback_handler, pattern="^generate_(?!set)"))
    application.add_handler(CallbackQueryHandler(generate_selection_callback_handler, pattern="^generate_set:"))
    application.add_handler(CallbackQueryHandler(toggle_thought_handler, pattern="^toggle_thought:"))

    logger.info("机器人正在启动...")
    application.run_polling()
    logger.info("机器人已停止。")

if __name__ == '__main__':
    main()