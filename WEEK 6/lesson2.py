import random

class SmartThermostat:
    def __init__(self):
        self.preferred_temperature = 22

    def learn_preferences(self, user_behavior):
        self.preferred_temperature = sum(user_behavior) / len(user_behavior)

    def adjust_temperature(self):
        current_temperature = random.randint(18, 26)
        if current_temperature != self.preferred_temperature:
            print(f"Adjusting temperature to {self.preferred_temperature}°C")
        else:
            print("Temperature is optimal.")

user_behavior = [22, 23, 22, 21, 22, 23, 22]
thermostat = SmartThermostat()
thermostat.learn_preferences(user_behavior)
thermostat.adjust_temperature()