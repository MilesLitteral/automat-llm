const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const readline = require("readline");
const dotenv = require("dotenv");

let mainWindow;
let pythonProcess;
let rl = null;
let backendReady = false;

// --------------------
// Create the main window
// --------------------
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1080,
        height: 780,
        webPreferences: {
            preload: path.join(__dirname, "preload.js"),
            contextIsolation: true
        }
    });

    mainWindow.loadFile("Cybel Dashboard.html");
}

// --------------------
// Determine Python path
// --------------------
function getPythonPath() {
    const pythonExe = process.env.PYTHON_PATH;
    if (!pythonExe) {
        throw new Error("PYTHON_PATH not set in .env");
    }
    return pythonExe;
}

function getScriptPath() {
    let obj = path.join(__dirname, "backend/main.py");
    print(obj);
    return obj;
}

// --------------------
// Spawn Python backend
// --------------------
function startPython() {
    const pythonExe = getPythonPath();
    const scriptPath = getScriptPath();

    pythonProcess = spawn(pythonExe, [scriptPath], {
        cwd: __dirname,
        stdio: ["pipe", "pipe", "pipe"]
    });

    // Read stdout
    rl = readline.createInterface({ input: pythonProcess.stdout });

    rl.on("line", (line) => {
        console.log("PYTHON:", line);
        if (line.trim() === "READY") {
            backendReady = true;
            console.log("Backend is READY");
        }
    });

    pythonProcess.stderr.on("data", (data) => {
        console.error("PYTHON ERROR:", data.toString());
    });

    pythonProcess.on("exit", (code) => {
        console.log(`Python process exited with code ${code}`);
    });
}

// --------------------
// IPC handler
// --------------------
ipcMain.handle("chat-message", async (event, message) => {
    if (!backendReady) {
        return "Backend not ready";
    }

    return new Promise((resolve) => {
        rl.once("line", (line) => {
            resolve(line);
        });

        pythonProcess.stdin.write(message + "\n");
    });
});

// --------------------
// App ready
// --------------------
app.whenReady().then(() => {
    createWindow();
    startPython();
});
