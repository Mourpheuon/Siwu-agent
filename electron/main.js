/**
 * 思悟 Agent —— Electron 主进程
 *
 * 职责：
 * 1. 启动 Python uvicorn 后端（子进程）
 * 2. 等待端口就绪
 * 3. 创建 BrowserWindow 加载前端
 * 4. 窗口关闭时清理 Python 进程
 */

const { app, BrowserWindow, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');
const net = require('net');

// ── 常量 ──────────────────────────────────────────────────────────
const HOST = '127.0.0.1';
const PYTHON_COMMAND = process.platform === 'win32' ? 'python' : 'python3';

// 项目根目录：electron/ 的父目录
const PROJECT_ROOT = path.resolve(__dirname, '..');

// ── 端口查找（避免与开发版冲突）────────────────────────────────────
function findFreePort(startPort) {
    return new Promise((resolve, reject) => {
        const server = net.createServer();
        server.unref();
        server.on('error', () => resolve(findFreePort(startPort + 1)));
        server.listen(startPort, HOST, () => {
            const port = server.address().port;
            server.close(() => resolve(port));
        });
    });
}

// ── 全局状态 ──────────────────────────────────────────────────────
let mainWindow = null;
let pythonProcess = null;
let actualPort = null;

// ── Python 后端管理 ───────────────────────────────────────────────

function startPythonBackend(port) {
    return new Promise((resolve, reject) => {
        const env = { ...process.env };
        // Ensure .env file env vars are read by Python
        // (python-dotenv loads them in siwu/config.py)

        pythonProcess = spawn(PYTHON_COMMAND, [
            '-m', 'uvicorn',
            'siwu.api.server:app',
            '--host', HOST,
            '--port', String(port),
        ], {
            cwd: PROJECT_ROOT,
            env,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        pythonProcess.on('error', (err) => {
            console.error('[siwu] Python 启动失败:', err.message);
            reject(new Error(`无法启动 Python: ${err.message}\n请确认已安装 Python 3.11+ 及依赖 (pip install -e ".[web]")`));
        });

        pythonProcess.on('exit', (code, signal) => {
            console.log(`[siwu] Python 后端已退出 (code=${code}, signal=${signal})`);
            pythonProcess = null;
        });

        // 收集 stderr 用于调试
        let stderrLog = '';
        pythonProcess.stderr.on('data', (data) => {
            const msg = data.toString();
            stderrLog += msg;
            // uvicorn 的启动信息走 stderr，所以不隐藏
            process.stderr.write(`[py] ${msg}`);
        });

        // Poll until backend is ready
        const startTime = Date.now();
        const MAX_WAIT = 30000;
        const RETRY_INTERVAL = 300;

        const checkReady = () => {
            const req = http.get(`http://${HOST}:${port}/api/v1/setup/status`, (res) => {
                // 任何响应（包括 200/404/500）都说明端口在监听
                console.log(`[siwu] Python 后端就绪 (${Date.now() - startTime}ms)`);
                resolve();
            });

            req.on('error', () => {
                if (Date.now() - startTime > MAX_WAIT) {
                    reject(new Error(
                        `Python 后端超时未就绪 (${MAX_WAIT}ms)\n\n` +
                        `stderr 输出:\n${stderrLog.slice(-2000)}`
                    ));
                } else {
                    setTimeout(checkReady, RETRY_INTERVAL);
                }
            });

            req.setTimeout(2000, () => {
                req.destroy();
                if (Date.now() - startTime > MAX_WAIT) {
                    reject(new Error(`Python 后端超时未就绪 (${MAX_WAIT}ms)`));
                } else {
                    setTimeout(checkReady, RETRY_INTERVAL);
                }
            });
        };

        // 给 Python 进程一点启动时间，然后开始轮询
        setTimeout(checkReady, 1000);
    });
}

function stopPythonBackend() {
    if (pythonProcess) {
        console.log('[siwu] 正在关闭 Python 后端...');
        if (process.platform === 'win32') {
            // Windows: 使用 taskkill 确保子进程树全部终止
            spawn('taskkill', ['/pid', String(pythonProcess.pid), '/f', '/t']);
        } else {
            pythonProcess.kill('SIGTERM');
            // 如果 3 秒后还没退出，强制 kill
            setTimeout(() => {
                if (pythonProcess) {
                    pythonProcess.kill('SIGKILL');
                }
            }, 3000);
        }
        pythonProcess = null;
    }
}

// ── 窗口管理 ──────────────────────────────────────────────────────

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        minWidth: 800,
        minHeight: 600,
        title: '思悟 Agent',
        icon: path.join(__dirname, '..', 'assets', 'icon.png'),
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });

    // Load the page from our Python backend on its actually-allocated port
    mainWindow.loadURL(`http://${HOST}:${actualPort}`);

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Open external links in default browser
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        if (!url.startsWith(`http://${HOST}`)) {
            require('electron').shell.openExternal(url);
            return { action: 'deny' };
        }
        return { action: 'allow' };
    });
}

// ── 应用生命周期 ──────────────────────────────────────────────────

app.whenReady().then(async () => {
    try {
        actualPort = await findFreePort(8000);
        console.log(`[siwu] Using port ${actualPort}`);
        await startPythonBackend(actualPort);
        createWindow();
    } catch (err) {
        console.error('[siwu] 启动失败:', err.message);
        dialog.showErrorBox('思悟 Agent 启动失败', err.message);
        app.quit();
    }
});

app.on('window-all-closed', () => {
    stopPythonBackend();
    // macOS: 保持应用在 Dock 中
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    // macOS: 点击 Dock 图标时重新创建窗口
    if (mainWindow === null && pythonProcess) {
        createWindow();
    }
});

app.on('before-quit', () => {
    stopPythonBackend();
});

// macOS: 通过 Cmd+Q 退出时确保清理
app.on('will-quit', () => {
    stopPythonBackend();
});
