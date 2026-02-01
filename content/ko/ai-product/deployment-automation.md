---
title: "AI 제품 배포 자동화"
description: "AI 제품 배포 파이프라인 자동화하기"
date: 2026-01-31
draft: true
tags: ["배포", "자동화", "DevOps"]
categories: ["ai-product"]
---

## 배포를 자동화하는 이유

**수동 배포 문제점:**
- ❌ 30-60분 소요
- ❌ 오류 발생 가능 (환경 변수 업데이트 잊음?)
- ❌ 주말/저녁에 배포 불가
- ❌ 배포가 무서움 (문제가 생기면?)

**자동 배포 장점:**
- ✅ 5분 안에 배포
- ✅ 일관된 프로세스
- ✅ 언제든 자신있게 배포
- ✅ 쉬운 롤백

## 빠른 시작: 가장 쉬운 배포

### 1. Vercel (프론트엔드 + API)
**최적:** Next.js, React, Vue

```bash
# Vercel CLI 설치
npm i -g vercel

# 배포
cd your-app
vercel

# 완료! 2분 만에 앱이 실행됩니다
```

**GitHub 통합:**
1. GitHub에 코드 푸시
2. Vercel에 저장소 연결
3. `main`에 푸시할 때마다 = 자동 배포

### 2. Railway (풀스택 + 데이터베이스)
**최적:** 데이터베이스가 있는 Python/Node 앱

```bash
# Railway CLI 설치
npm i -g @railway/cli

# 로그인 및 배포
railway login
railway init
railway up

# 자동: 데이터베이스, 환경, HTTPS
```

### 3. Streamlit Cloud (Python 앱)
**최적:** Streamlit 대시보드

```bash
# CLI 필요 없음!
1. GitHub에 푸시
2. share.streamlit.io 방문
3. 저장소 연결
4. git push 시 자동 배포
```

## AI 제품을 위한 CI/CD

### 완전한 GitHub Actions 워크플로우

**파일:** `.github/workflows/deploy.yml`

```yaml
name: AI 앱 배포

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Python 설정
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: 의존성 설치
        run: |
          pip install -r requirements.txt
          pip install pytest
      
      - name: 테스트 실행
        run: pytest
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      
      - name: AI 엔드포인트 테스트
        run: |
          python test_ai.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Vercel에 배포
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          vercel-args: '--prod'
```

### 이것이 하는 일:

1. **모든 푸시마다:**
   - 테스트 실행
   - AI 엔드포인트 작동 확인
   
2. **main에 푸시 시:**
   - 프로덕션에 배포
   - 테스트 통과 시에만

### 시크릿 설정

```bash
# GitHub에서:
# Settings → Secrets → New repository secret

OPENAI_API_KEY=sk-...
VERCEL_TOKEN=...
DB_CONNECTION_STRING=postgresql://...
```

## 플랫폼 배포 가이드

### 옵션 1: Vercel (MVP에 권장)

**1. 설치 및 구성:**
```json
// vercel.json
{
  "builds": [
    { "src": "api/**/*.py", "use": "@vercel/python" }
  ],
  "routes": [
    { "src": "/api/(.*)", "dest": "api/$1.py" }
  ],
  "env": {
    "OPENAI_API_KEY": "@openai-key"
  }
}
```

**2. 배포:**
```bash
vercel --prod
```

**3. 자동 배포:**
- GitHub에 푸시 → 자동 배포
- PR용 미리보기 URL
- 원클릭 롤백

### 옵션 2: Railway (데이터베이스 + API)

**1. Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**2. 배포:**
```bash
railway up
```

**3. 데이터베이스 추가:**
```bash
railway add postgresql
# 자동: DATABASE_URL 환경 변수 추가됨
```

### 옵션 3: AWS Lambda (서버리스)

**최적:** 간헐적 트래픽, 비용 최적화

**1. 함수 코드:**
```python
# lambda_function.py
import json
import openai
import os

def lambda_handler(event, context):
    openai.api_key = os.environ['OPENAI_API_KEY']
    
    prompt = json.loads(event['body'])['prompt']
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'result': response.choices[0].message.content
        })
    }
```

**2. AWS SAM으로 배포:**
```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  AIFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: .
      Handler: lambda_function.lambda_handler
      Runtime: python3.10
      Timeout: 30
      Environment:
        Variables:
          OPENAI_API_KEY: !Ref OpenAIKey
      Events:
        API:
          Type: Api
          Properties:
            Path: /generate
            Method: post
```

```bash
sam build
sam deploy --guided
```

## Docker 배포

### 최적화된 Dockerfile

```dockerfile
# 더 작은 이미지를 위한 멀티 스테이지 빌드
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .

# 의존성 설치
RUN pip install --user --no-cache-dir -r requirements.txt

# 최종 스테이지
FROM python:3.10-slim

WORKDIR /app

# 빌더에서 의존성 복사
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

# 보안을 위한 비루트 사용자
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose (로컬 개발)

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/aiapp
    depends_on:
      - db
  
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: aiapp
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

```bash
# 로컬 실행
docker-compose up

# 프로덕션 배포
docker-compose -f docker-compose.prod.yml up -d
```

## 롤백 전략

### 1. 빠른 롤백 (Vercel)
```bash
# 배포 목록
vercel ls

# 이전 버전으로 롤백
vercel rollback [deployment-url]
```

### 2. Git 기반 롤백
```bash
# 마지막 커밋 되돌리기
git revert HEAD
git push

# 이전 버전 자동 배포
```

### 3. 블루-그린 배포
```yaml
# GitHub Actions
deploy-blue:
  - 블루 환경에 배포
  - 스모크 테스트 실행
  - 통과 시: 블루로 트래픽 전환
  - 실패 시: 그린에 트래픽 유지
```

## 모니터링 및 알림

### 1. 가동 시간 모니터링

**무료 도구:**
- [UptimeRobot](https://uptimerobot.com): 5분마다 핑
- [Pingdom](https://pingdom.com): 이메일 알림

**설정:**
```bash
# 엔드포인트 모니터링
https://your-app.com/health

# 반환:
{
  "status": "healthy",
  "timestamp": "2026-01-31T10:00:00Z",
  "services": {
    "database": "connected",
    "ai_api": "responding"
  }
}
```

### 2. 오류 추적

**Sentry 통합:**
```python
import sentry_sdk

sentry_sdk.init(
    dsn="https://...",
    traces_sample_rate=0.1,
    profiles_sample_rate=0.1,
)

# 자동 오류 추적
# Slack/이메일로 알림 받기
```

### 3. 성능 모니터링

```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        
        # 모니터링 서비스에 로그
        logger.info(f"{func.__name__}: {duration:.2f}s")
        
        if duration > 5:  # 5초 초과 시 알림
            alert_slack(f"느린 함수: {func.__name__}")
        
        return result
    return wrapper

@monitor_performance
async def generate_content(prompt):
    return await openai_call(prompt)
```

## 모범 사례

### 1. 환경 변수

```bash
# .env (로컬)
OPENAI_API_KEY=sk-test-...
DB_URL=postgresql://localhost/dev
DEBUG=true

# 프로덕션 (플랫폼에서 설정)
OPENAI_API_KEY=sk-prod-...
DB_URL=postgresql://prod-db/app
DEBUG=false
```

**시크릿을 Git에 커밋하지 마세요!**

### 2. 헬스 체크

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    # 데이터베이스 테스트
    try:
        await db.execute("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    # AI API 테스트
    try:
        await openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        ai_status = "healthy"
    except:
        ai_status = "unhealthy"
    
    return {
        "status": "healthy" if all([db_status, ai_status]) else "degraded",
        "database": db_status,
        "ai_api": ai_status
    }
```

### 3. 배포 전 자동 테스트

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

def test_generate_endpoint():
    client = TestClient(app)
    
    response = client.post("/generate", json={
        "prompt": "하이쿠 작성"
    })
    
    assert response.status_code == 200
    assert len(response.json()["content"]) > 10

def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### 4. 점진적 배포

```yaml
# 먼저 10%의 사용자에게 배포
deploy:
  - 새 버전 배포
  - 10% 트래픽을 새 버전으로 라우팅
  - 30분 동안 모니터링
  - 오류 < 1%: 50%로 증가
  - 오류 < 1%: 100%로 증가
  - 그렇지 않으면: 롤백
```

## 배포 체크리스트

```
배포 전:
□ 모든 테스트 통과
□ 환경 변수 설정
□ 데이터베이스 마이그레이션 준비
□ 롤백 계획 문서화

배포:
□ 트래픽이 적은 시간에 배포
□ 15분간 로그 모니터링
□ 헬스 엔드포인트 확인
□ 중요한 사용자 흐름 테스트

배포 후:
□ 지표 확인 (응답 시간, 오류)
□ 사용자 피드백 채널 확인
□ 문제 문서화
□ 변경 로그 업데이트
```

## 첫 배포

**1일차: 설정**
1. 플랫폼 선택 (Vercel/Railway 권장)
2. GitHub 저장소 연결
3. 환경 변수 설정
4. 배포!

**2일차: 자동화**
1. GitHub Actions 워크플로우 추가
2. 모니터링 설정 (UptimeRobot)
3. 오류 추적 구성 (Sentry)

**3일차: 최적화**
1. 헬스 체크 추가
2. 롤백 프로세스 설정
3. 배포 단계 문서화

**결과:** 몇 시간이 아닌 몇 분 만에 새 기능 배포!
