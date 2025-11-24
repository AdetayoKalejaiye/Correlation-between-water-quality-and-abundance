# Deploying to Render.com

## 🚀 Quick Deployment Steps

### 1. Push Your Code to GitHub
```bash
git add .
git commit -m "Add Render deployment files"
git push origin master
```

### 2. Deploy on Render.com

1. **Sign up/Login** to [Render.com](https://render.com/)

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select repository: `Correlation-between-water-quality-and-abundance`

3. **Configure Settings**
   - **Name**: `water-quality-analysis` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free` (or paid for better performance)

4. **Click "Create Web Service"**
   - Render will automatically deploy your app
   - Your app will be live at: `https://water-quality-analysis.onrender.com`

### 3. Access Your App
Once deployed, visit your URL to see the live application!

## 📁 Files Created for Deployment

- ✅ **requirements.txt** - Python dependencies (updated with gunicorn)
- ✅ **Procfile** - Tells Render how to run the app
- ✅ **runtime.txt** - Specifies Python version

## ⚙️ Environment Variables (Optional)
If needed, you can add environment variables in Render dashboard:
- Go to your service → "Environment" tab
- Add any secrets or configuration

## 🔄 Automatic Deployments
Render will automatically redeploy when you push to GitHub:
```bash
git add .
git commit -m "Update feature"
git push
```

## 💡 Important Notes

1. **First deployment** takes 5-10 minutes
2. **Free tier** apps sleep after 15 min of inactivity (takes ~30s to wake up)
3. **Custom domain** available on paid plans
4. **View logs** in Render dashboard for debugging

## 🌐 Your Live URLs Will Be:
- Main app: `https://your-app-name.onrender.com/`
- Predictions: `https://your-app-name.onrender.com/`
- Correlations: `https://your-app-name.onrender.com/correlations`
- Models: `https://your-app-name.onrender.com/models`
- API endpoint: `https://your-app-name.onrender.com/api/predict`

## 🆘 Troubleshooting

If deployment fails, check Render logs:
1. Go to your service dashboard
2. Click "Logs" tab
3. Look for error messages

Common issues:
- Missing dependencies → Update requirements.txt
- Port issues → Render handles this automatically with gunicorn
- Build timeout → Use a paid plan for faster builds
