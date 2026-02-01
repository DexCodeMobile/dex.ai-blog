---
title: "Deployment Automation for AI Products"
description: "Automate your AI product deployment pipeline"
date: 2026-01-31
draft: true
tags: ["deployment", "automation", "DevOps"]
categories: ["ai-product"]
---

## Why Automate Deployment?

**Manual deployment problems:**
- ❌ Takes 30-60 minutes
- ❌ Prone to errors (forgot to update env var?)
- ❌ Can't deploy on weekends/evenings
- ❌ Scary to deploy (what if it breaks?)

**Automated deployment benefits:**
- ✅ Deploy in 5 minutes
- ✅ Consistent process
- ✅ Deploy anytime, confidently
- ✅ Easy to rollback

## Quick Start: Easiest Deployments

### 1. Vercel (Frontend + API)
**Best for:** Next.js, React, Vue

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd your-app
vercel

# Done! Your app is live in 2 minutes
```

**GitHub Integration:**
1. Push code to GitHub
2. Connect repo to Vercel
3. Every push to `main` = auto-deploy

### 2. Railway (Full-stack + Database)
**Best for:** Python/Node apps with database

```bash
# Install Railway CLI  
npm i -g @railway/cli

# Login & deploy
railway login
railway init
railway up

# Automatic: Database, environment, HTTPS
```

### 3. Streamlit Cloud (Python apps)
**Best for:** Streamlit dashboards

```bash
# No CLI needed!
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repo
4. Auto-deploys on git push
```

## CI/CD for AI Products

### Complete GitHub Actions Workflow

**File:** `.github/workflows/deploy.yml`

```yaml
name: Deploy AI App

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
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      
      - name: Run tests
        run: pytest
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      
      - name: Test AI endpoint
        run: |
          python test_ai.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          vercel-args: '--prod'
```

### What This Does:

1. **On every push:**
   - Runs tests
   - Checks AI endpoint works
   
2. **On push to main:**
   - Deploys to production
   - Only if tests pass

### Setting Up Secrets

```bash
# On GitHub:
# Settings → Secrets → New repository secret

OPENAI_API_KEY=sk-...
VERCEL_TOKEN=...
DB_CONNECTION_STRING=postgresql://...
```

## Platform Deployment Guides

### Option 1: Vercel (Recommended for MVPs)

**1. Install & Configure:**
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

**2. Deploy:**
```bash
vercel --prod
```

**3. Automatic deploys:**
- Push to GitHub → Auto-deploy
- Preview URLs for PRs
- Rollback in one click

### Option 2: Railway (Database + API)

**1. Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**2. Deploy:**
```bash
railway up
```

**3. Add database:**
```bash
railway add postgresql
# Automatic: DATABASE_URL env var added
```

### Option 3: AWS Lambda (Serverless)

**Best for:** Sporadic traffic, cost optimization

**1. Function code:**
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

**2. Deploy with AWS SAM:**
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

## Docker Deployment

### Optimized Dockerfile

```dockerfile
# Multi-stage build for smaller images
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .

# Install dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose (Local dev)

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
# Run locally
docker-compose up

# Deploy to production
docker-compose -f docker-compose.prod.yml up -d
```

## Rollback Strategies

### 1. Quick Rollback (Vercel)
```bash
# List deployments
vercel ls

# Rollback to previous
vercel rollback [deployment-url]
```

### 2. Git-based Rollback
```bash
# Revert last commit
git revert HEAD
git push

# Auto-deploys previous version
```

### 3. Blue-Green Deployment
```yaml
# GitHub Actions
deploy-blue:
  - Deploy to blue environment
  - Run smoke tests
  - If pass: Switch traffic to blue
  - If fail: Keep traffic on green
```

## Monitoring and Alerts

### 1. Uptime Monitoring

**Free tools:**
- [UptimeRobot](https://uptimerobot.com): Ping every 5 minutes
- [Pingdom](https://pingdom.com): Email alerts

**Setup:**
```bash
# Monitor endpoint
https://your-app.com/health

# Returns:
{
  "status": "healthy",
  "timestamp": "2026-01-31T10:00:00Z",
  "services": {
    "database": "connected",
    "ai_api": "responding"
  }
}
```

### 2. Error Tracking

**Sentry integration:**
```python
import sentry_sdk

sentry_sdk.init(
    dsn="https://...",
    traces_sample_rate=0.1,
    profiles_sample_rate=0.1,
)

# Automatic error tracking
# Get alerts on Slack/Email
```

### 3. Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        
        # Log to monitoring service
        logger.info(f"{func.__name__}: {duration:.2f}s")
        
        if duration > 5:  # Alert if > 5 seconds
            alert_slack(f"Slow function: {func.__name__}")
        
        return result
    return wrapper

@monitor_performance
async def generate_content(prompt):
    return await openai_call(prompt)
```

## Best Practices

### 1. Environment Variables

```bash
# .env (local)
OPENAI_API_KEY=sk-test-...
DB_URL=postgresql://localhost/dev
DEBUG=true

# Production (set in platform)
OPENAI_API_KEY=sk-prod-...
DB_URL=postgresql://prod-db/app
DEBUG=false
```

**Never commit secrets to Git!**

### 2. Health Checks

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    # Test database
    try:
        await db.execute("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    # Test AI API
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

### 3. Automated Testing Before Deploy

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

def test_generate_endpoint():
    client = TestClient(app)
    
    response = client.post("/generate", json={
        "prompt": "Write a haiku"
    })
    
    assert response.status_code == 200
    assert len(response.json()["content"]) > 10

def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### 4. Gradual Rollouts

```yaml
# Deploy to 10% of users first
deploy:
  - Deploy new version
  - Route 10% traffic to new version
  - Monitor for 30 minutes
  - If errors < 1%: increase to 50%
  - If errors < 1%: increase to 100%
  - Else: rollback
```

## Deployment Checklist

```
Pre-deployment:
□ All tests passing
□ Environment variables set
□ Database migrations ready
□ Rollback plan documented

Deployment:
□ Deploy during low-traffic period
□ Monitor logs for 15 minutes
□ Check health endpoint
□ Test critical user flows

Post-deployment:
□ Verify metrics (response time, errors)
□ Check user feedback channels
□ Document any issues
□ Update changelog
```

## Your First Deployment

**Day 1: Setup**
1. Choose platform (Vercel/Railway recommended)
2. Connect GitHub repo
3. Set environment variables
4. Deploy!

**Day 2: Automation**
1. Add GitHub Actions workflow
2. Setup monitoring (UptimeRobot)
3. Configure error tracking (Sentry)

**Day 3: Optimize**
1. Add health checks
2. Setup rollback process
3. Document deployment steps

**Result:** Deploy new features in minutes, not hours!
