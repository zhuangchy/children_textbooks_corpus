@echo off
chcp 65001 >nul
echo [1/4] 安装 Git LFS...
git lfs install

echo [2/4] 跟踪数据库文件...
git lfs track "*.db"

echo [3/4] 添加 .gitattributes...
git add .gitattributes

echo [4/4] 提交 LFS 配置...
git commit -m "配置 Git LFS 支持"

echo.
echo 完成。你现在可以在 GitHub Desktop 中推送到远程仓库。
pause
