# 🚀 Deploy ScrollIntel to scrollintel.com - Step by Step Guide

## 🎯 Your Goal: Get ScrollIntel live at https://scrollintel.com

Your GitHub repository is ready: https://github.com/stanleymay20/scrollintel-ai-system

## 🥇 OPTION 1: Railway (EASIEST - RECOMMENDED)

### Step 1: Deploy to Railway
1. Go to **https://railway.app**
2. Click **"Login"** and sign in with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose: **stanleymay20/scrollintel-ai-system**
6. Railway will auto-detect and deploy!

### Step 2: Add Environment Variables
In Railway dashboard:
1. Go to your project
2. Click **"Variables"** tab
3. Add these variables:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ENVIRONMENT=production
   DEBUG=false
   ```

### Step 3: Add Custom Domain
1. In Railway dashboard, go to **"Settings"**
2. Click **"Domains"**
3. Add custom domain: **scrollintel.com**
4. Railway will give you DNS instructions

### Step 4: Configure DNS
In your domain registrar (where you bought scrollintel.com):
1. Add A record: `@` → `[Railway IP address]`
2. Add CNAME: `www` → `scrollintel.com`

**Result: https://scrollintel.com will be live!**

---

## 🥈 OPTION 2: Vercel + Render (BEST PERFORMANCE)

### Frontend on Vercel:
1. Go to **https://vercel.com**
2. Click **"New Project"**
3. Import from GitHub: **stanleymay20/scrollintel-ai-system**
4. Set **Root Directory**: `frontend`
5. Add custom domain: **scrollintel.com**
6. Deploy!

### Backend on Render:
1. Go to **https://render.com**
2. Click **"New +"** → **"Web Service"**
3. Connect GitHub: **stanleymay20/scrollintel-ai-system**
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn scrollintel.api.simple_main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables (same as above)
6. Add custom domain: **api.scrollintel.com**

---

## 🥉 OPTION 3: DigitalOcean App Platform

1. Go to **https://cloud.digitalocean.com/apps**
2. Click **"Create App"**
3. Choose **GitHub** and select your repository
4. DigitalOcean will auto-configure
5. Add custom domain in settings
6. Deploy!

---

## 🔧 DNS Configuration (For All Options)

Add these records to your domain registrar:

```
Type    Name    Value                           TTL
A       @       [Your deployment IP]            300
CNAME   www     scrollintel.com                 300
CNAME   api     [Your API deployment URL]      300
```

## 🎉 After Deployment

Your ScrollIntel platform will be live at:
- **Main Site**: https://scrollintel.com
- **API**: https://api.scrollintel.com (if using separate backend)
- **API Docs**: https://scrollintel.com/docs or https://api.scrollintel.com/docs
- **Health Check**: https://scrollintel.com/health

## 🧪 Test Your Deployment

Run this command to test:
```bash
python test_scrollintel_domain.py
```

## 🚀 What You'll Have Live

✅ **AI-Powered CTO Platform**
✅ **Multiple AI Agents** (CTO, ML Engineer, Data Scientist, etc.)
✅ **File Processing** (Upload and analyze any data)
✅ **Visual Generation** (Create images and videos)
✅ **Real-time Chat** with AI agents
✅ **Dashboard** with system metrics
✅ **API Documentation** for developers
✅ **Enterprise Security** and monitoring

## 🎯 Recommended: Start with Railway

Railway is the easiest option:
1. One-click deployment
2. Automatic database setup
3. Easy custom domain configuration
4. Built-in monitoring

**Go to https://railway.app and deploy now!**

---

## 🆘 Need Help?

If you encounter any issues:
1. Check the deployment logs in your platform dashboard
2. Verify environment variables are set correctly
3. Ensure DNS records are configured properly
4. Test the health endpoint first

**ScrollIntel.com is ready to go live! 🌟**