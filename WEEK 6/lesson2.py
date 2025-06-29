# Example: AI-powered IoT device for temperature control
import random

class SmartThermostat:
    def __init__(self):
        self.preferred_temperature = 22  # Default preferred temperature

    def learn_preferences(self, user_behavior):
        # Simulate learning user preferences
        self.preferred_temperature = sum(user_behavior) / len(user_behavior)

    def adjust_temperature(self):
        # Simulate temperature adjustment
        current_temperature = random.randint(18, 26)
        if current_temperature != self.preferred_temperature:
            print(f"Adjusting temperature to {self.preferred_temperature}°C")
        else:
            print("Temperature is optimal.")

# Simulate user behavior (e.g., preferred temperatures over a week)
user_behavior = [22, 23, 22, 21, 22, 23, 22]
thermostat = SmartThermostat()
thermostat.learn_preferences(user_behavior)
thermostat.adjust_temperature()