import random

class HeartRateMonitor:
    def __init__(self):
        self.normal_heart_rate = range(60, 100)

    def monitor(self):
        heart_rate = random.randint(50, 120)
        if heart_rate not in self.normal_heart_rate:
            print(f"Alert: Abnormal heart rate detected ({heart_rate} BPM)!")
        else:
            print(f"Heart rate is normal ({heart_rate} BPM).")

monitor = HeartRateMonitor()
import random

class HeartRateMonitor:
    def __init__(self):
        self.normal_heart_rate = range(60, 100)

    def monitor(self):
        heart_rate = random.randint(50, 120)
        if heart_rate not in self.normal_heart_rate:
            print(f"Alert: Abnormal heart rate detected ({heart_rate} BPM)!")
        else:
            print(f"Heart rate is normal ({heart_rate} BPM).")

monitor = HeartRateMonitor()
monitor.monitor()