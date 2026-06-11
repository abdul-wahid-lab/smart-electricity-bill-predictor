import pandas as pd
import numpy as np

np.random.seed(42)
n = 50000

fan     = np.random.randint(0, 11, n)   # 0-10
fridge  = np.random.randint(0, 4,  n)   # 0-3
ac      = np.random.randint(0, 4,  n)   # 0-3
tv      = np.random.randint(0, 5,  n)   # 0-4
monitor = np.random.randint(0, 4,  n)   # 0-3
pump    = np.random.randint(0, 3,  n)   # 0-2
month   = np.random.randint(1, 13, n)   # 1-12
daily_h = np.round(np.random.uniform(4, 18, n), 1)   # avg hours/day all appliances run
tariff  = np.round(np.random.uniform(5, 25, n), 1)   # PKR per kWh

# Appliance wattages (W): Fan=75, Fridge=150, AC=1500, TV=100, Monitor=50, Pump=750
watts_kw = (fan*75 + fridge*150 + ac*1500 + tv*100 + monitor*50 + pump*750) / 1000.0

# Monthly kWh = power (kW) × daily hours × 30 days
monthly_kwh = watts_kw * daily_h * 30

# Seasonal multiplier: summer (May-Sep) is 15% higher due to AC/fan usage
seasonal = np.where((month >= 5) & (month <= 9), 1.15, 1.0)
monthly_kwh = monthly_kwh * seasonal

# 5% random noise to make it realistic
monthly_kwh = monthly_kwh * np.random.normal(1.0, 0.05, n)
monthly_kwh = np.maximum(monthly_kwh, 0)

bill = np.round(monthly_kwh * tariff, 2)

df = pd.DataFrame({
    'Fan':            fan,
    'Refrigerator':   fridge,
    'AirConditioner': ac,
    'Television':     tv,
    'Monitor':        monitor,
    'MotorPump':      pump,
    'Month':          month,
    'DailyHours':     daily_h,
    'TariffRate':     tariff,
    'ElectricityBill': bill,
})

df = df[df['ElectricityBill'] > 0].reset_index(drop=True)
df.to_csv('electricity_bill_dataset.csv', index=False)
print(f"Saved {len(df)} records")
print(df[['Fan','AirConditioner','DailyHours','TariffRate','ElectricityBill']].describe().round(2))
