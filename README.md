# EV Adoption Forecasting 🚗⚡

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Why This Project Exists

Electric vehicles are everywhere now, and they're only getting more popular. But here's the thing - nobody wants to drive around looking for a charging station that doesn't exist. That's where this project comes in. I built this forecasting model to help city planners and policymakers figure out where we'll need charging infrastructure before we actually need it.

Think of it as a crystal ball for EV adoption, but powered by data instead of magic.

## The Problem I'm Solving

Cities are scrambling to keep up with EV growth. One day there are a few Teslas in the neighborhood, and suddenly everyone's driving electric. Without good predictions, we end up with either way too many charging stations in the wrong places, or not enough where people actually need them.

This project takes historical EV registration data and uses machine learning to predict future adoption patterns. It's like having a weather forecast, but for electric cars.

## What You'll Get

When you run this model, you'll get forecasts that help answer questions like:
- How many EVs will be on the road in my area next year?
- Which counties are likely to see the biggest growth?
- Should we be planning for more passenger cars or trucks?

## About the Data

I'm using real registration data from Washington State (they have some of the best EV data around). The dataset covers everything from 2017 to early 2024 and includes:

- **Monthly vehicle counts** - How many EVs were registered each month
- **Location data** - Broken down by county and state
- **Vehicle types** - Both fully electric (BEVs) and plug-in hybrids (PHEVs)
- **Usage patterns** - Whether they're passenger cars or trucks

Here's what the data looks like:

| What's In The Data | What It Means |
|-------------------|---------------|
| **Date** | When the count was taken (monthly snapshots) |
| **County** | Where the vehicle owner lives |
| **Vehicle Primary Use** | Passenger car or truck |
| **BEVs** | Pure electric vehicles (no gas engine) |
| **PHEVs** | Plug-in hybrids (electric + gas backup) |
| **EV Total** | All electric vehicles combined |
| **Percent Electric** | What percentage of all vehicles are electric |

## Getting Started

### What You'll Need
- Python 3.8 or newer
- About 10 minutes to set everything up

### Setting It Up

1. **Grab the code:**
   ```bash
   https://github.com/RajV231/EV_Adopt_Forecasting.git
   ```

2. **Create a clean Python environment** (trust me, this saves headaches later):
   ```bash
   python -m venv venv
   ```

3. **Activate your environment:**
   
   **Windows folks:**
   ```bash
   .\venv\Scripts\activate
   ```
   
   **Mac/Linux users:**
   ```bash
   source venv/bin/activate
   ```

4. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Model

1. Put your `EV_DataSet.csv` file in the same folder as `main.py`

2. Run it:
   ```bash
   python main.py
   ```

3. Grab some coffee while it thinks (shouldn't take long)

4. Check out your results and the fancy chart it creates!

## What You'll See

The model spits out three main things:

### How Well It's Working
- **Mean Absolute Error**: On average, how far off are the predictions?
- **R-squared**: How much of the pattern did the model actually capture? (Higher is better)

### Your Forecast
You'll get predictions for the next three years showing how many EVs to expect. The model looks at recent trends and makes educated guesses about what's coming.

### A Pretty Chart
The code creates `ev_adoption_forecast_rf.png` - a visual timeline showing where we've been and where we're headed. Perfect for presentations or just satisfying your curiosity.

## Why This Approach Works

I could have built a simple linear model, but EV adoption isn't linear. It's messy, seasonal, and varies wildly by location. So I went with a Random Forest model that can handle all that complexity:

**Smart Features**: The model doesn't just look at time - it considers location, seasonality, and vehicle types. Some counties love Teslas, others prefer trucks. The model learns these patterns.

**Handles Real-World Messiness**: Missing data? Outliers? The preprocessing steps clean things up so the model can focus on the real patterns.

**Actually Useful Predictions**: Instead of just saying "EVs will increase," it gives you specific numbers you can actually plan with.

## Want to Help Make It Better?

I'd love your contributions! Whether you're a data scientist, urban planner, or just someone who cares about EVs, there's probably something you can add.

**Easy ways to help:**
- Test it with data from your area
- Suggest new features that would be useful
- Fix bugs (there are always bugs)
- Improve the documentation

**Getting involved:**
1. Fork the repo
2. Make your changes
3. Submit a pull request
4. I'll take a look and we can make it happen

## The Fine Print

This project is MIT licensed, which basically means you can do whatever you want with it. Use it for your city, your research, your startup - just don't blame me if the predictions are wrong!

## Questions or Ideas?

Hit me up open an issue on GitHub. I'm always interested in hearing how people are using this or what would make it more useful.

## Thanks To

- The folks at Washington State DOL for making great EV data publicly available
- Everyone who's contributed to the open-source libraries that make this possible
- The data science community for sharing knowledge and keeping things interesting

---

**Found this useful?** Give it a star ⭐ - it helps other people find it too!
