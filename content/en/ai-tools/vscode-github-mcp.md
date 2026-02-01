---
title: "VS Code + GitHub MCP: AI-Powered Development"
description: "Setup and use Model Context Protocol in VS Code"
date: 2026-01-31
draft: false
tags: ["VS Code", "MCP", "GitHub", "tools"]
categories: ["ai-tools"]
---

## What is MCP?

**Model Context Protocol (MCP)** = Standard protocol for AI tools to interact with external systems

**Benefits:**
- ✅ Connect AI to GitHub, databases, APIs
- ✅ Give AI ability to read/write code
- ✅ Automate development workflows
- ✅ Build custom AI agents

## Setup VS Code MCP

### 1. Install Prerequisites

```bash
# Install GitHub Copilot
# VS Code → Extensions → Search "GitHub Copilot"

# Install Node.js (for MCP servers)
# Download from nodejs.org
```

### 2. Install MCP Server

```bash
# Install GitHub MCP server
npm install -g @modelcontextprotocol/server-github

# Or use npx (no install needed)
npx @modelcontextprotocol/server-github
```

### 3. Configure VS Code

Create `.vscode/mcp.json`:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

### 4. Get GitHub Token

1. Go to github.com → Settings → Developer settings
2. Personal access tokens → Tokens (classic)
3. Generate new token
4. Select scopes: `repo`, `read:org`, `read:user`
5. Copy token to `mcp.json`

## Features

### 1. Repository Search

Ask Copilot to search your repos:

```
"Search my repositories for Python ML projects"
"Find repos with FastAPI in the codebase"
"Show repos I contributed to last month"
```

### 2. Code Navigation

```
"Show me the main.py file in user/repo"
"Find all functions in src/api.py"
"What does the database.py module do?"
```

### 3. Issue Management

```
"List open issues in my project"
"Create an issue: Fix login bug"
"Close issue #42"
"Show issues assigned to me"
```

### 4. Pull Request Automation

```
"List open PRs in my repo"
"Create a PR from feature-branch to main"
"Review PR #15"
"Merge PR #10"
```

### 5. File Operations

```
"Read the README.md file"
"Update package.json to add a new dependency"
"Create a new file: src/utils.py"
```

## Example Workflows

### Workflow 1: Bug Fix

```
You: "Find the login bug in auth.py"
Copilot: [Searches code, finds issue]

You: "Create a branch called fix-login"
Copilot: [Creates branch]

You: "Fix the bug"
Copilot: [Suggests fix, applies it]

You: "Create a PR"
Copilot: [Creates PR with description]
```

### Workflow 2: Feature Development

```
You: "Search for authentication examples in my repos"
Copilot: [Finds examples]

You: "Create new file: src/auth/oauth.py"
Copilot: [Creates file]

You: "Implement OAuth based on the examples"
Copilot: [Writes code]

You: "Run tests"
Copilot: [Executes tests, shows results]
```

### Workflow 3: Code Review

```
You: "Show me open PRs"
Copilot: [Lists PRs]

You: "Review PR #25"
Copilot: [Shows changes, suggests improvements]

You: "Add comment: Please add error handling"
Copilot: [Adds comment]

You: "Approve and merge"
Copilot: [Approves and merges]
```

## Building Custom MCP Server

### Simple MCP Server

```javascript
// server.js
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "my-custom-server",
  version: "1.0.0"
});

// Register tool
server.setRequestHandler("tools/list", async () => ({
  tools: [{
    name: "get_weather",
    description: "Get current weather",
    inputSchema: {
      type: "object",
      properties: {
        city: { type: "string" }
      }
    }
  }]
}));

// Handle tool call
server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "get_weather") {
    const city = request.params.arguments.city;
    // Call weather API here
    return {
      content: [{
        type: "text",
        text: `Weather in ${city}: Sunny, 72°F`
      }]
    };
  }
});

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
```

### Add to VS Code

```json
{
  "mcpServers": {
    "weather": {
      "command": "node",
      "args": ["path/to/server.js"]
    }
  }
}
```

## Best Practices

### 1. Secure Tokens

```bash
# Use environment variables
export GITHUB_TOKEN=ghp_your_token

# Or use VS Code secrets
# Settings → Search "mcp" → Configure securely
```

### 2. Organize MCP Servers

```json
{
  "mcpServers": {
    "github": { /* GitHub integration */ },
    "database": { /* Database queries */ },
    "api": { /* Custom API calls */ }
  }
}
```

### 3. Error Handling

```
If MCP server fails:
1. Check token is valid
2. Verify server is running: `npx @modelcontextprotocol/server-github`
3. Check VS Code Output → MCP Servers
4. Restart VS Code
```

### 4. Performance

```
- Keep MCP servers lightweight
- Cache frequently accessed data
- Use lazy loading for large repos
- Set timeouts for long operations
```

## Productivity Tips

**1. Quick commands:**
```
"@github search X" - Fast GitHub search
"@github pr" - List/manage PRs
"@github issue" - List/manage issues
```

**2. Multi-step workflows:**
```
"Create a feature branch, implement X, create PR"
→ Copilot does all steps automatically
```

**3. Context awareness:**
```
"Fix this bug" (while viewing file)
→ Copilot knows which file you mean
```

**4. Keyboard shortcuts:**
```
Cmd/Ctrl + I - Open Copilot Chat
Cmd/Ctrl + Shift + I - Inline Copilot
```

## Common Use Cases

**Daily tasks:**
- Check PRs and issues
- Search across repositories
- Create branches and commits
- Review code changes

**Development:**
- Find code examples
- Implement features with AI
- Refactor code
- Generate tests

**Collaboration:**
- Create detailed PRs
- Comment on issues/PRs
- Track project progress
- Automate workflows

## Troubleshooting

**MCP server not connecting:**
```bash
# Check if server works standalone
npx @modelcontextprotocol/server-github

# Check token
echo $GITHUB_TOKEN

# Restart VS Code
```

**Copilot not using MCP:**
```
1. Verify mcp.json is in .vscode folder
2. Check JSON syntax
3. Reload VS Code window
4. Check Output → MCP Servers for errors
```

**Rate limits:**
```
GitHub API has rate limits.
Solution: Use GitHub App token (higher limits)
Or wait for rate limit reset
```
