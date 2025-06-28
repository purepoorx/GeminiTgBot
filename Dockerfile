# 使用官方 Python 运行时作为父镜像
FROM python:3.11-slim

# 将工作目录设置为 /app
WORKDIR /app

# 复制依赖文件到工作目录
COPY requirements.txt .

# 安装所需的包
# --no-cache-dir 选项可以减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 将当前目录内容复制到容器的 /app 目录
COPY . .

# 将容器的 80 端口暴露给外部
# (虽然这个特定的bot不使用端口，但这是一个好习惯)
EXPOSE 80

# 定义容器启动时运行的命令
CMD ["python", "bot.py"]