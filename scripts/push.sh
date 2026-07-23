#!/usr/bin/env bash
# 思悟 Agent —— Git Push 脚本
# 从项目根目录的 .github-token 文件读取 token 并推送。
# 用法：bash scripts/push.sh [branch]
#
# 前置条件：在项目根目录创建 .github-token 文件，内容为 GitHub Personal Access Token
# （已加入 .gitignore，不会被提交到仓库）

set -euo pipefail

BRANCH="${1:-main}"
shift 2>/dev/null || true
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOKEN_FILE="$PROJECT_ROOT/.github-token"
REMOTE_NAME="origin"

if [ ! -f "$TOKEN_FILE" ]; then
    echo "错误：未找到 .github-token 文件"
    echo ""
    echo "请先在项目根目录创建该文件："
    echo "  echo 'ghp_xxxxxxxxxxxx' > $TOKEN_FILE"
    echo ""
    echo "如何生成 token："
    echo "  GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)"
    echo "  → Generate new token (classic) → 勾选 'repo' → 复制 token"
    exit 1
fi

TOKEN="$(cat "$TOKEN_FILE" | tr -d '\n\r ')"
if [ -z "$TOKEN" ]; then
    echo "错误：.github-token 文件为空"
    exit 1
fi

# 获取远程仓库 URL 并注入 token
REMOTE_URL="$(cd "$PROJECT_ROOT" && git remote get-url "$REMOTE_NAME")"
if echo "$REMOTE_URL" | grep -q "^https://"; then
    # https://github.com/USER/REPO.git → https://TOKEN@github.com/USER/REPO.git
    AUTH_URL="$(echo "$REMOTE_URL" | sed "s|https://|https://${TOKEN}@|")"
else
    echo "错误：远程仓库 URL 不是 HTTPS 格式: $REMOTE_URL"
    echo "GitHub 已不再支持密码认证，请使用 token"
    exit 1
fi

cd "$PROJECT_ROOT"
echo "[push] branch=$BRANCH remote=$REMOTE_NAME url=${REMOTE_URL%%@*}@***"
git push "$AUTH_URL" "$BRANCH" "$@"

echo "[push] 完成"
