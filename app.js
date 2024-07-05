import express from 'express';
import bodyParser from 'body-parser';
import pkg from '@aws-sdk/client-bedrock-runtime';

const { BedrockRuntimeClient, InvokeModelCommand } = pkg;

const app = express();
const port = process.env.PORT || 3000;

// 中间件
app.use(bodyParser.json());

// 辅助函数（保持不变）
function openaiToClaudeParams(messages) {
  messages = messages.filter((message) => message.role !== "system");
  messages.forEach((message) => {
    if (message.content && typeof message.content !== "string") {
      message.content.forEach((item) => {
        if (item.type === "image_url") {
          const imageUrl = item.image_url.url;
          const base64Image = imageUrl.substring(
            imageUrl.indexOf("{") + 1,
            imageUrl.indexOf("}")
          );
          item.type = "image";
          item.source = {
            type: "base64",
            media_type: "image/jpeg",
            data: base64Image,
          };
          delete item.image_url;
        }
      });
    }
  });

  return messages;
}

function claudeToChatgptResponseStream(claudeFormat) {
  const obj2Data = {
    choices: [
      {
        finish_reason: "stop",
        index: 0,
        message: {
          content: claudeFormat.content[0].text,
          role: claudeFormat.role,
        },
        logprobs: null,
      },
    ],
    created: Math.floor(Date.now() / 1000),
    id: claudeFormat.id,
    model: claudeFormat.model,
    object: "chat.completion",
    usage: {
      completion_tokens: claudeFormat.usage.output_tokens,
      prompt_tokens: claudeFormat.usage.input_tokens,
      total_tokens:
        claudeFormat.usage.input_tokens + claudeFormat.usage.output_tokens,
    },
  };
  return obj2Data;
}

// 路由处理
app.post(['/v1/chat/completions', '/v1/messages'], async (req, res) => {
  const path = req.path;
  const isClaude = path === "/v1/messages";

  if (req.body && req.body.model && req.body.messages && req.body.messages.length > 0) {
    let body = req.body;
    let system = body.system;
    if (body.messages[0].role === "system") system = body.messages[0].content;
    let convertedMessages = isClaude ? body.messages : openaiToClaudeParams(body.messages);
    console.log("begin invoke message", convertedMessages);

    if (convertedMessages.length <= 0) {
      return res.status(400).json("Invalid request!");
    }

    let max_tokens = body.max_tokens || 1000;
    let top_p = body.top_p || 1;
    let top_k = body.top_k || 250;
    let modelId = "anthropic.claude-3-sonnet-20240229-v1:0";
    if (body.model.startsWith("anthropic")) modelId = body.model;
    let temperature = body.temperature || 0.5;
    const contentType = "application/json";

    const rockerRuntimeClient = new BedrockRuntimeClient({
      region: process.env.REGION,
    });

    const inputCommand = {
      modelId,
      contentType,
      accept: contentType,
      body: system
        ? JSON.stringify({
            anthropic_version: "bedrock-2023-05-31",
            max_tokens: max_tokens,
            temperature: temperature,
            top_k: top_k,
            top_p: top_p,
            system: system,
            messages: convertedMessages,
          })
        : JSON.stringify({
            anthropic_version: "bedrock-2023-05-31",
            max_tokens: max_tokens,
            temperature: temperature,
            top_k: top_k,
            top_p: top_p,
            messages: convertedMessages,
          }),
    };

    try {
      const command = new InvokeModelCommand(inputCommand);
      const response = await rockerRuntimeClient.send(command);
      const responseBody = JSON.parse(new TextDecoder().decode(response.body));

      const result = isClaude
        ? responseBody
        : claudeToChatgptResponseStream(responseBody);

      console.log("invoke success response", result);
      res.json(result);
    } catch (error) {
      console.error("Error invoking model:", error);
      res.status(500).json({ error: "Internal Server Error" });
    }
  } else {
    res.status(400).json("Invalid request!");
  }
});

// 启动服务器
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
