import dotenv from "dotenv";
import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const API_KEY = process.env.GROQ_API_KEY;
const MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

async function initializeSDK() {
  try {
    if (!API_KEY) {
      console.warn("⚠️ GROQ_API_KEY not set; running in fallback mode.");
      return { ready: false, fallback: true };
    }

    return { ready: true, fallback: false };
  } catch (err) {
    console.error("❌ Failed to initialize Groq client:", err.message);
    return { ready: false, fallback: true };
  }
}

function getApiStatusMessage() {
  return API_KEY ? "API key configured" : "No API key configured; set GROQ_API_KEY in the hosting environment";
}

let aiClient = null;

const ROOT = __dirname;
const INDEX = path.join(ROOT, "index.html");

function json(res, status, data) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
  });

  res.end(JSON.stringify(data));
}

function send404(res) {
  res.writeHead(404);
  res.end("Not Found");
}

function serveFile(res, file) {
  fs.readFile(file, (err, data) => {
    if (err) {
      send404(res);
      return;
    }

    const ext = path.extname(file);

    const types = {
      ".html": "text/html",
      ".css": "text/css",
      ".js": "application/javascript",
      ".json": "application/json",
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".svg": "image/svg+xml",
      ".ico": "image/x-icon"
    };

    res.writeHead(200, {
      "Content-Type": types[ext] || "application/octet-stream",
      "Access-Control-Allow-Origin": "*"
    });

    res.end(data);
  });
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", chunk => body += chunk);
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

// ---- Groq API (OpenAI-compatible chat completions) ----

function buildStructuredPrompt(userPrompt) {
  return [
    "You are Fahxd_Core, a helpful AI assistant.",
    "Answer the user's question in a clear, easy-to-scan format.",
    "Prefer a short intro followed by numbered steps (1., 2., 3...) or short bullet points.",
    "Keep each point concise and practical.",
    "If the user asks for a simple answer, still break it into 2-4 points.",
    "",
    `User question: ${userPrompt}`
  ].join("\n");
}

async function callGroqAPI(prompt) {
  const endpoint = "https://api.groq.com/openai/v1/chat/completions";

  const payload = {
    model: MODEL,
    messages: [
      {
        role: "system",
        content: "You are a helpful assistant. Respond in a structured, point-by-point way."
      },
      { role: "user", content: buildStructuredPrompt(prompt) }
    ]
  };

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${API_KEY}`
    },
    body: JSON.stringify(payload)
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.error?.message || `Groq API error ${response.status}`);
  }

  return data.choices?.[0]?.message?.content || "";
}

async function callGroqAPIStream(prompt, onChunk) {
  const endpoint = "https://api.groq.com/openai/v1/chat/completions";

  const payload = {
    model: MODEL,
    messages: [
      {
        role: "system",
        content: "You are a helpful assistant. Respond in a structured, point-by-point way."
      },
      { role: "user", content: buildStructuredPrompt(prompt) }
    ],
    stream: true
  };

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${API_KEY}`
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error?.message || `Groq API error ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;

      const dataStr = trimmed.slice(5).trim();
      if (dataStr === "[DONE]") continue;

      try {
        const parsed = JSON.parse(dataStr);
        const delta = parsed.choices?.[0]?.delta?.content;
        if (delta) onChunk(delta);
      } catch {
        // ignore malformed chunk
      }
    }
  }
}

async function handleGroq(req, res) {
  if (!API_KEY) {
    return json(res, 500, {
      error: "Missing GROQ_API_KEY"
    });
  }

  if (!aiClient) {
    return json(res, 500, {
      error: "AI client not initialized"
    });
  }

  let body;

  try {
    body = JSON.parse(await readBody(req));
  } catch {
    return json(res, 400, {
      error: "Invalid JSON"
    });
  }

  const prompt = body.prompt;

  if (!prompt) {
    return json(res, 400, {
      error: "Prompt required"
    });
  }

  try {
    console.log("🚀 Sending prompt to Groq:", prompt.substring(0, 50) + "...");

    const reply = await callGroqAPI(prompt);

    if (!reply) {
      console.warn("⚠️  Empty reply from Groq");
      return json(res, 500, {
        error: "Groq returned empty response"
      });
    }

    console.log("✅ Got reply from Groq");
    json(res, 200, { reply });

  } catch (error) {
    console.error("❌ Groq API Error:", error.message);
    console.error("Full Error:", error);

    json(res, 500, {
      error: error.message || "Unknown API error",
      details: error.code || "No error code"
    });
  }
}

async function handleStream(req, res) {
  if (!API_KEY) {
    res.writeHead(500);
    return res.end();
  }

  if (!aiClient) {
    res.writeHead(500);
    return res.end();
  }

  let body;

  try {
    body = JSON.parse(await readBody(req));
  } catch {
    res.writeHead(400);
    return res.end();
  }

  try {
    console.log("🚀 Starting stream for:", body.prompt.substring(0, 50) + "...");

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "Access-Control-Allow-Origin": "*"
    });

    await callGroqAPIStream(body.prompt, (chunk) => {
      res.write(`data: ${JSON.stringify({ text: chunk })}\n\n`);
    });

    res.write("data: [DONE]\n\n");
    res.end();

  } catch (err) {
    console.error("❌ Stream Error:", err.message);
    res.writeHead(500);
    res.end(err.message);
  }
}

const server = http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
    });
    return res.end();
  }

  const url = new URL(req.url, `http://${req.headers.host}`);

  if (url.pathname === "/api/health") {
    return json(res, 200, {
      status: aiClient ? "ok" : "initializing",
      model: MODEL,
      hasApiKey: !!API_KEY,
      sdkLoaded: !!aiClient
    });
  }

  if (url.pathname === "/api/groq") {
    return handleGroq(req, res);
  }

  if (url.pathname === "/api/groq-stream") {
    return handleStream(req, res);
  }

  let file = url.pathname === "/"
    ? INDEX
    : path.join(ROOT, decodeURIComponent(url.pathname));

  if (!file.startsWith(ROOT)) {
    return send404(res);
  }

  if (!fs.existsSync(file)) {
    file = INDEX;
  }

  serveFile(res, file);
});

// Initialize and start server
async function start() {
  console.log("\n===================================");
  console.log("Initializing Groq client...");

  aiClient = await initializeSDK();

  if (!aiClient?.ready) {
    console.warn("AI client not ready; the chat will use fallback responses until a valid API key is configured.");
  }

  server.listen(PORT, () => {
    console.log("");
    console.log("===================================");
    console.log(`Server : http://localhost:${PORT}`);
    console.log(`Model  : ${MODEL}`);
    console.log(`API Key: ${API_KEY ? "Loaded ✅" : "Missing ❌"}`);
    console.log(`SDK    : ${aiClient?.ready ? "Loaded ✅" : "Fallback mode ⚠️"}`);
    console.log("===================================");
    console.log("");
  });
}

start().catch(err => {
  console.error("Failed to start server:", err);
  process.exit(1);
});