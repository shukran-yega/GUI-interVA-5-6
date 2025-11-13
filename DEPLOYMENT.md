# 🚀 Deployment Guide for Render.com

This guide explains how to deploy the InterVA Analysis API to Render.com.

## 📋 Prerequisites

1. **GitHub Repository**: Your code must be in a Git repository
2. **Render.com Account**: Sign up at https://render.com
3. **Docker Installed** (for local testing): https://docker.com

## 🐳 Local Testing (Optional but Recommended)

Test your Docker container locally before deploying:

```powershell
# Build the Docker image
docker build -t interva-api .

# Run the container
docker run -p 8000:8000 interva-api

# Test the API
# Open browser: http://localhost:8000
# Or use curl: curl http://localhost:8000/health
```

## 🌐 Deploy to Render.com

### Method 1: Using render.yaml (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add Docker deployment files"
   git push origin master
   ```

2. **Connect to Render.com**:
   - Go to https://render.com/dashboard
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Select the repository with your code
   - Render will auto-detect `render.yaml`
   - Click "Apply" to deploy

3. **Wait for Build**:
   - Render will build your Docker image (5-10 minutes)
   - Monitor the build logs
   - Once complete, you'll get a URL like: `https://interva-api-xxx.onrender.com`

### Method 2: Manual Web Service Creation

1. **Create Web Service**:
   - Go to Render Dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**:
   - **Name**: `interva-analysis-api`
   - **Environment**: `Docker`
   - **Region**: Choose closest to your users
   - **Plan**: Free (or Starter for production)
   - **Dockerfile Path**: `./Dockerfile`

3. **Environment Variables** (optional):
   ```
   PORT=8000
   PYTHONUNBUFFERED=1
   ENVIRONMENT=production
   ```

4. **Deploy**:
   - Click "Create Web Service"
   - Wait for build to complete

## 🔧 Post-Deployment

### Update Frontend URL

Once deployed, update the API URL in `index.html`:

```javascript
// Change this line:
const API_BASE_URL = 'http://127.0.0.1:8000';

// To your Render URL:
const API_BASE_URL = 'https://your-service-name.onrender.com';
```

Then commit and redeploy.

### Test Your Deployment

1. **Health Check**:
   ```
   https://your-service-name.onrender.com/health
   ```

2. **Open Web Interface**:
   ```
   https://your-service-name.onrender.com/
   ```

## 📊 Render.com Plans

| Plan | Price | RAM | Features |
|------|-------|-----|----------|
| **Free** | $0 | 512MB | Spins down after 15min inactivity |
| **Starter** | $7/mo | 512MB | Always on, Custom domains |
| **Standard** | $25/mo | 2GB | Better performance |
| **Pro** | $85/mo | 4GB | High traffic |

**Recommendation**: 
- **Free tier** for testing/demo
- **Starter** for small production use (<100 users)
- **Standard** for larger datasets or more users

## ⚠️ Free Tier Limitations

- Service spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds (cold start)
- 512MB RAM (handles ~500 VA records comfortably)
- 750 hours/month free runtime

## 🔥 Troubleshooting

### Build Fails
```bash
# Check logs in Render dashboard
# Common issues:
# - Missing dependencies in requirements.txt
# - Wrong Python version
# - Large file sizes
```

### Service Won't Start
- Check environment variables
- Verify PORT is set correctly
- Check logs for Python errors

### Out of Memory
- Upgrade to Starter plan (more RAM)
- Or reduce file size limits in code

## 📝 Updating Your Deployment

After code changes:

```bash
git add .
git commit -m "Update feature X"
git push origin master
```

Render will auto-deploy (if auto-deploy enabled).

## 🌟 Production Checklist

- [ ] Updated API_BASE_URL in index.html
- [ ] Tested with real WHO VA data
- [ ] Set up custom domain (optional)
- [ ] Enable auto-deploy
- [ ] Monitor logs regularly
- [ ] Consider upgrading from Free tier

## 🆘 Support

- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- FastAPI Docs: https://fastapi.tiangolo.com

---

**Your app is now live! 🎉**

Share your URL: `https://your-service-name.onrender.com`
