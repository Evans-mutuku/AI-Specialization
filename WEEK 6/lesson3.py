# Example: AI-powered heart rate monitoring
import random

class HeartRateMonitor:
    def __init__(self):
        self.normal_heart_rate = range(60, 100)  # Normal heart rate range

    def monitor(self):
        # Simulate heart rate data
        heart_rate = random.randint(50, 120)
        if heart_rate not in self.normal_heart_rate:
            print(f"Alert: Abnormal heart rate detected ({heart_rate} BPM)!")
        else:
            print(f"Heart rate is normal ({heart_rate} BPM).")

# Simulate monitoring
monitor = HeartRateMonitor()
import random

class HeartRateMonitor:
    def __init__(self):
        self.normal_heart_rate = range(60, 100)  # Normal heart rate range

    def monitor(self):
        # Simulate heart rate data
        heart_rate = random.randint(50, 120)
        if heart_rate not in self.normal_heart_rate:
            print(f"Alert: Abnormal heart rate detected ({heart_rate} BPM)!")
        else:
            print(f"Heart rate is normal ({heart_rate} BPM).")

# Simulate monitoring
monitor = HeartRateMonitor()
monitor.monitor()