.PHONY: help init dev test commit cz cm cm-check clean

help:
	@echo "开发环境:"
	@echo "  make init          初始化开发环境"
	@echo "  make dev           启动 CLI 交互"
	@echo "  make test          运行测试"
	@echo ""
	@echo "提交规范:"
	@echo "  make commit        交互式生成 Conventional Commits 提交"
	@echo "  make cm            同 make commit"
	@echo "  make cz            同 make commit"
	@echo "  make cm-check      检查最近一次提交是否符合规范"
	@echo ""
	@echo "其他:"
	@echo "  make clean         清理临时文件"

# ---------- 本地开发 ----------
init:
	uv sync
	cp -n .env.example .env 2>/dev/null || true
	echo 'source .venv/bin/activate' > .envrc
	direnv allow 2>/dev/null || true
	pre-commit install --hook-type commit-msg 2>/dev/null || true
	@echo "开发环境初始化完成（需编辑 .env 填入 DEEPSEEK_API_KEY）"

dev:
	PYTHONIOENCODING=utf-8 uv run python -m src.aigc.main

test:
	uv run pytest -v

# ---------- Conventional Commits ----------
commit:
	uv run cz commit

cz: commit

cm: commit

cm-check:
	git log -1 --format=%B | uv run cz check

# ---------- 清理 ----------
clean:
	rm -rf __pycache__ src/aigc/**/__pycache__
	@echo "临时文件已清理"
