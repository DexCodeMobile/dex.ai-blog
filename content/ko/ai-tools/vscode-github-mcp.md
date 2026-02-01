---
title: "VS Code + GitHub MCP: AI 기반 개발"
description: "VS Code에서 모델 컨텍스트 프로토콜 설정 및 사용"
date: 2026-01-31
draft: false
tags: ["VS Code", "MCP", "GitHub", "도구"]
categories: ["ai-tools"]
---

## MCP란 무엇인가요?

**모델 컨텍스트 프로토콜 (MCP)** = AI 도구가 외부 시스템과 상호작용하기 위한 표준 프로토콜

**이점:**
- ✅ AI를 GitHub, 데이터베이스, API에 연결
- ✅ AI에게 코드 읽기/쓰기 능력 부여
- ✅ 개발 워크플로우 자동화
- ✅ 커스텀 AI 에이전트 구축

## VS Code MCP 설정

### 1. 사전 요구사항 설치

```bash
# GitHub Copilot 설치
# VS Code → Extensions → "GitHub Copilot" 검색

# Node.js 설치 (MCP 서버용)
# nodejs.org에서 다운로드
```

### 2. MCP 서버 설치

```bash
# GitHub MCP 서버 설치
npm install -g @modelcontextprotocol/server-github

# 또는 npx 사용 (설치 불필요)
npx @modelcontextprotocol/server-github
```

### 3. VS Code 구성

`.vscode/mcp.json` 생성:

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

### 4. GitHub 토큰 받기

1. github.com → Settings → Developer settings
2. Personal access tokens → Tokens (classic)
3. Generate new token
4. 범위 선택: `repo`, `read:org`, `read:user`
5. 토큰을 `mcp.json`에 복사

## 기능

### 1. 리포지토리 검색

Copilot에게 리포지토리 검색 요청:

```
"Python ML 프로젝트 리포지토리를 검색해줘"
"코드베이스에 FastAPI가 있는 리포지토리 찾아줘"
"지난달 내가 기여한 리포지토리 보여줘"
```

### 2. 코드 탐색

```
"user/repo의 main.py 파일 보여줘"
"src/api.py의 모든 함수 찾아줘"
"database.py 모듈이 뭐하는 거야?"
```

### 3. 이슈 관리

```
"내 프로젝트의 열린 이슈 목록"
"이슈 생성: 로그인 버그 수정"
"이슈 #42 닫기"
"나에게 할당된 이슈 보여줘"
```

### 4. 풀 리퀘스트 자동화

```
"내 리포의 열린 PR 목록"
"feature-branch에서 main으로 PR 생성"
"PR #15 리뷰해줘"
"PR #10 병합해줘"
```

### 5. 파일 작업

```
"README.md 파일 읽어줘"
"package.json 업데이트해서 새 의존성 추가"
"새 파일 생성: src/utils.py"
```

## 예제 워크플로우

### 워크플로우 1: 버그 수정

```
당신: "auth.py에서 로그인 버그 찾아줘"
Copilot: [코드 검색, 문제 발견]

당신: "fix-login이라는 브랜치 만들어줘"
Copilot: [브랜치 생성]

당신: "버그 고쳐줘"
Copilot: [수정 제안, 적용]

당신: "PR 만들어줘"
Copilot: [설명과 함께 PR 생성]
```

### 워크플로우 2: 기능 개발

```
당신: "내 리포에서 인증 예제 검색해줘"
Copilot: [예제 찾기]

당신: "새 파일 생성: src/auth/oauth.py"
Copilot: [파일 생성]

당신: "예제 기반으로 OAuth 구현해줘"
Copilot: [코드 작성]

당신: "테스트 실행해줘"
Copilot: [테스트 실행, 결과 표시]
```

### 워크플로우 3: 코드 리뷰

```
당신: "열린 PR 보여줘"
Copilot: [PR 목록]

당신: "PR #25 리뷰해줘"
Copilot: [변경사항 표시, 개선 제안]

당신: "코멘트 추가: 에러 처리 추가해주세요"
Copilot: [코멘트 추가]

당신: "승인하고 병합해줘"
Copilot: [승인 및 병합]
```

## 커스텀 MCP 서버 구축

### 간단한 MCP 서버

```javascript
// server.js
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "my-custom-server",
  version: "1.0.0"
});

// 도구 등록
server.setRequestHandler("tools/list", async () => ({
  tools: [{
    name: "get_weather",
    description: "현재 날씨 가져오기",
    inputSchema: {
      type: "object",
      properties: {
        city: { type: "string" }
      }
    }
  }]
}));

// 도구 호출 처리
server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "get_weather") {
    const city = request.params.arguments.city;
    // 여기서 날씨 API 호출
    return {
      content: [{
        type: "text",
        text: `${city}의 날씨: 맑음, 22°C`
      }]
    };
  }
});

// 서버 시작
const transport = new StdioServerTransport();
await server.connect(transport);
```

### VS Code에 추가

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

## 모범 사례

### 1. 토큰 보안

```bash
# 환경 변수 사용
export GITHUB_TOKEN=ghp_your_token

# 또는 VS Code 시크릿 사용
# Settings → "mcp" 검색 → 안전하게 구성
```

### 2. MCP 서버 정리

```json
{
  "mcpServers": {
    "github": { /* GitHub 통합 */ },
    "database": { /* 데이터베이스 쿼리 */ },
    "api": { /* 커스텀 API 호출 */ }
  }
}
```

### 3. 오류 처리

```
MCP 서버 실패 시:
1. 토큰이 유효한지 확인
2. 서버 실행 확인: `npx @modelcontextprotocol/server-github`
3. VS Code Output → MCP Servers 확인
4. VS Code 재시작
```

### 4. 성능

```
- MCP 서버를 가볍게 유지
- 자주 액세스하는 데이터 캐시
- 큰 리포에 대해 지연 로딩 사용
- 긴 작업에 타임아웃 설정
```

## 생산성 팁

**1. 빠른 명령:**
```
"@github search X" - 빠른 GitHub 검색
"@github pr" - PR 목록/관리
"@github issue" - 이슈 목록/관리
```

**2. 다단계 워크플로우:**
```
"기능 브랜치 만들고, X 구현하고, PR 생성해줘"
→ Copilot이 모든 단계를 자동으로 수행
```

**3. 컨텍스트 인식:**
```
"이 버그 고쳐줘" (파일을 보면서)
→ Copilot이 어떤 파일인지 알고 있음
```

**4. 키보드 단축키:**
```
Cmd/Ctrl + I - Copilot Chat 열기
Cmd/Ctrl + Shift + I - 인라인 Copilot
```

## 일반적인 사용 사례

**일상 작업:**
- PR과 이슈 확인
- 리포지토리 전체 검색
- 브랜치와 커밋 생성
- 코드 변경사항 리뷰

**개발:**
- 코드 예제 찾기
- AI로 기능 구현
- 코드 리팩토링
- 테스트 생성

**협업:**
- 상세한 PR 생성
- 이슈/PR에 코멘트
- 프로젝트 진행 상황 추적
- 워크플로우 자동화

## 문제 해결

**MCP 서버 연결 안 됨:**
```bash
# 서버가 독립적으로 작동하는지 확인
npx @modelcontextprotocol/server-github

# 토큰 확인
echo $GITHUB_TOKEN

# VS Code 재시작
```

**Copilot이 MCP를 사용하지 않음:**
```
1. mcp.json이 .vscode 폴더에 있는지 확인
2. JSON 구문 확인
3. VS Code 창 다시 로드
4. Output → MCP Servers에서 오류 확인
```

**요청 제한:**
```
GitHub API에는 요청 제한이 있습니다.
해결책: GitHub App 토큰 사용 (더 높은 제한)
또는 요청 제한 재설정 대기
```
