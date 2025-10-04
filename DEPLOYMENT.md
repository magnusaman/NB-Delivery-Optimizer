# Deploy to Render - Step by Step Guide

## 🚀 Quick Deployment

### 1. **Prepare Your Repository**
```bash
# Make sure all files are committed
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### 2. **Create New Render Project**

1. **Go to [Render Dashboard](https://dashboard.render.com)**
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repository**
4. **Select this repository**

### 3. **Configure Render Settings**

**Basic Settings:**
- **Name**: `nb-delivery-vrp-demo` (or your preferred name)
- **Environment**: `Python 3`
- **Region**: `Oregon (US West)` (closest to Canada)
- **Branch**: `main`

**Build & Deploy:**
- **Build Command**: `pip install -r demo/requirements.txt`
- **Start Command**: `cd demo && gunicorn app:app`

**Environment Variables:**
- **Key**: `GOOGLE_MAPS_API_KEY`
- **Value**: Your Google Maps API key

### 4. **Advanced Settings (Optional)**

**Auto-Deploy:**
- ✅ **Auto-Deploy**: Yes (deploys on every push)

**Health Check:**
- **Health Check Path**: `/api/data`

**Scaling:**
- **Instance Type**: Free tier (or upgrade for better performance)

### 5. **Deploy!**

1. **Click "Create Web Service"**
2. **Wait for build to complete** (2-3 minutes)
3. **Your app will be live at**: `https://your-app-name.onrender.com`

## 🔧 Troubleshooting

### **Build Fails?**
- Check that `demo/requirements.txt` exists
- Verify Python version (3.11+ recommended)
- Check build logs in Render dashboard

### **App Crashes?**
- Verify `GOOGLE_MAPS_API_KEY` is set correctly
- Check that all data files exist in `nb_vrp_dataset/`
- Review logs in Render dashboard

### **Performance Issues?**
- Upgrade to paid plan for better resources
- Consider adding caching
- Optimize marker rendering

## 📊 Expected Performance on Render

- ✅ **Faster loading** - 2-3x faster than local
- ✅ **Global CDN** - Static files served worldwide
- ✅ **Production optimization** - Better memory management
- ✅ **HTTPS** - Secure connections
- ✅ **Auto-scaling** - Handles traffic spikes

## 🔑 Environment Variables Required

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_MAPS_API_KEY` | Google Maps API key with Directions, Distance Matrix, and Places APIs enabled | ✅ Yes |

## 📱 Features After Deployment

- 🗺️ **Interactive Map** - All 200+ stores, 72 partners, 200+ charging stations
- 🛣️ **Real Routes** - Google Directions API integration
- ⚡ **EV Charging** - Charging station visualization
- 📱 **Mobile Friendly** - Responsive design
- 🌍 **Global Access** - Available worldwide

## 🎯 Next Steps After Deployment

1. **Test the application** at your Render URL
2. **Verify all markers load** (stores, partners, charging stations)
3. **Test order placement** by clicking on the map
4. **Check mobile responsiveness**
5. **Share the URL** with others to test

Your app will be much faster and more reliable on Render compared to local development!
