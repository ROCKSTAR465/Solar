import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import paho.mqtt.client as mqtt
import time
import json
from datetime import datetime, timedelta

# =====================================
# 1. SOLAR TRACKING SYSTEM SIMULATION
# =====================================
class SolarTracker:
    def __init__(self):
        self.panel_azimuth = 0  # Horizontal angle (0-360°)
        self.panel_elevation = 30  # Vertical angle (0-90°)
        self.dust_level = 0  # 0-100% dust coverage
        self.temperature = 25  # Panel temperature in °C
        
    def calculate_sun_position(self, dt):
        """Calculate sun position based on time and date"""
        hour = dt.hour + dt.minute/60
        day_of_year = dt.timetuple().tm_yday
        
        # Solar position algorithm
        solar_azimuth = 180 + 15 * (hour - 12)  # Simplified model
        solar_elevation = 80 - abs(day_of_year - 172) * 0.4  # Higher in summer
        
        # Apply seasonal variation
        solar_elevation = max(10, min(solar_elevation, 90))
        return solar_azimuth, solar_elevation
    
    def optimize_position(self, solar_azimuth, solar_elevation):
        """Adjust panel position to track the sun"""
        # Dual-axis tracking logic
        self.panel_azimuth = solar_azimuth
        self.panel_elevation = solar_elevation
        
        # Calculate misalignment (0 = perfect alignment)
        azimuth_diff = abs(self.panel_azimuth - solar_azimuth)
        elevation_diff = abs(self.panel_elevation - solar_elevation)
        misalignment = (azimuth_diff + elevation_diff) / 2
        
        return misalignment
    
    def calculate_power_output(self, irradiance):
        """Calculate power output considering various factors"""
        # Base efficiency
        efficiency = 0.18
        
        # Temperature effect (-0.5% per °C above 25°C)
        temp_effect = max(0, 1 - 0.005 * (self.temperature - 25))
        
        # Dust effect (-0.8% per 1% dust)
        dust_effect = 1 - (self.dust_level * 0.008)
        
        # Alignment effect
        alignment_effect = 1 - (self.get_misalignment() / 180)
        
        # Calculate final output
        effective_irradiance = irradiance * alignment_effect * dust_effect
        power_output = effective_irradiance * efficiency * temp_effect
        
        return power_output
    
    def get_misalignment(self):
        """Get current misalignment from optimal position"""
        current_time = datetime.now()
        sun_azimuth, sun_elevation = self.calculate_sun_position(current_time)
        return abs(self.panel_azimuth - sun_azimuth) + abs(self.panel_elevation - sun_elevation)
    
    def accumulate_dust(self):
        """Simulate dust accumulation (1% per day)"""
        self.dust_level = min(100, self.dust_level + 1)
    
    def clean_panels(self):
        """Reset dust level after cleaning"""
        self.dust_level = 0

# ===================================
# 2. MPPT ALGORITHM IMPLEMENTATION
# ===================================
class MPPTController:
    def __init__(self):
        self.voltage = 0
        self.current = 0
        self.power = 0
        self.best_voltage = 0
        self.step_size = 0.5
        
    def perturb_and_observe(self, voltage, current):
        """Perturb and Observe MPPT algorithm"""
        new_power = voltage * current
        
        if new_power > self.power:
            # Continue in same direction
            if voltage > self.voltage:
                self.voltage += self.step_size
            else:
                self.voltage -= self.step_size
        else:
            # Change direction
            if voltage > self.voltage:
                self.voltage -= self.step_size
            else:
                self.voltage += self.step_size
        
        self.power = new_power
        self.current = current
        
        # Track best voltage point
        if new_power > self.best_voltage * current:
            self.best_voltage = voltage
        
        return self.voltage

# =====================================
# 3. ENERGY FORECASTING AI MODEL
# =====================================
class EnergyForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.weather_data = pd.DataFrame()
        
    def load_historical_data(self):
        """Generate synthetic historical data"""
        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        data = {
            'date': dates,
            'irradiance': [max(0, 1000 + 500 * np.sin(i/30) + random.uniform(-100, 100)) for i in range(365)],
            'temperature': [25 + 10 * np.sin(i/30) + random.uniform(-3, 3) for i in range(365)],
            'cloud_cover': [random.randint(0, 100) for _ in range(365)],
            'power_output': [0] * 365
        }
        
        # Calculate power output (simplified)
        for i in range(365):
            tracker = SolarTracker()
            tracker.temperature = data['temperature'][i]
            data['power_output'][i] = tracker.calculate_power_output(data['irradiance'][i])
        
        self.weather_data = pd.DataFrame(data)
    
    def train_model(self):
        """Train forecasting model"""
        if self.weather_data.empty:
            self.load_historical_data()
        
        # Prepare features
        self.weather_data['day_of_year'] = self.weather_data['date'].dt.dayofyear
        self.weather_data['month'] = self.weather_data['date'].dt.month
        
        X = self.weather_data[['day_of_year', 'irradiance', 'temperature', 'cloud_cover', 'month']]
        y = self.weather_data['power_output']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Model trained! MAE: {mae:.2f} Watts")
    
    def predict_energy(self, date):
        """Predict energy output for a specific date"""
        if self.weather_data.empty:
            self.load_historical_data()
        
        # Get weather forecast (simulated)
        day_of_year = date.timetuple().tm_yday
        month = date.month
        
        # Simulate weather forecast
        irradiance = 1000 + 500 * np.sin(day_of_year/30) + random.uniform(-100, 100)
        temperature = 25 + 10 * np.sin(day_of_year/30) + random.uniform(-3, 3)
        cloud_cover = random.randint(0, 100)
        
        # Create feature vector
        features = [[day_of_year, irradiance, temperature, cloud_cover, month]]
        
        # Predict
        prediction = self.model.predict(features)
        return prediction[0]

# =====================================
# 4. IOT COMMUNICATION AND MONITORING
# =====================================
class IoTMonitor:
    def __init__(self):
        self.client = mqtt.Client("solar_inverter")
        self.broker = "broker.hivemq.com"
        self.port = 1883
        self.topic = "smart_solar/inverter_data"
        
    def connect(self):
        self.client.connect(self.broker, self.port)
        self.client.loop_start()
    
    def publish(self, data):
        """Publish data to MQTT broker"""
        json_data = json.dumps(data)
        result = self.client.publish(self.topic, json_data)
        if result[0] == 0:
            print(f"Data published: {json_data}")
        else:
            print("Failed to publish data")
    
    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

# =====================================
# 5. AUTOMATED CLEANING SCHEDULER
# =====================================
class CleaningScheduler:
    def __init__(self, tracker):
        self.tracker = tracker
        self.last_cleaned = datetime.now() - timedelta(days=2)
        
    def check_cleaning_needed(self):
        """Determine if cleaning is needed based on dust and efficiency"""
        days_since_clean = (datetime.now() - self.last_cleaned).days
        efficiency_loss = self.tracker.dust_level * 0.008
        
        # Clean if dust > 15% or every 3 days
        if self.tracker.dust_level > 15 or days_since_clean >= 3:
            return True
        return False
    
    def perform_cleaning(self):
        """Perform cleaning operation"""
        self.tracker.clean_panels()
        self.last_cleaned = datetime.now()
        return {"status": "cleaned", "dust_before": self.tracker.dust_level, "dust_after": 0}

# =====================================
# 6. MAIN SYSTEM INTEGRATION
# =====================================
class SmartSolarSystem:
    def __init__(self):
        self.tracker = SolarTracker()
        self.mppt = MPPTController()
        self.forecaster = EnergyForecaster()
        self.monitor = IoTMonitor()
        self.cleaner = CleaningScheduler(self.tracker)
        
        # Initialize components
        self.forecaster.train_model()
        self.monitor.connect()
        
    def run(self, hours=24):
        """Run system simulation for specified hours"""
        start_time = datetime.now()
        
        for i in range(hours):
            current_time = start_time + timedelta(hours=i)
            print(f"\n--- Hour {i+1} @ {current_time.strftime('%Y-%m-%d %H:%M')} ---")
            
            # 1. Update sun position and adjust tracker
            sun_azimuth, sun_elevation = self.tracker.calculate_sun_position(current_time)
            misalignment = self.tracker.optimize_position(sun_azimuth, sun_elevation)
            
            # 2. Simulate environmental conditions
            irradiance = max(0, 1000 + 500 * np.sin(current_time.hour/24*2*np.pi) - 
                             current_time.hour * 10 + random.uniform(-50, 50))
            self.tracker.temperature = 25 + current_time.hour * 0.5
            
            # 3. Calculate power output
            power_output = self.tracker.calculate_power_output(irradiance)
            
            # 4. Apply MPPT optimization
            optimal_voltage = self.mppt.perturb_and_observe(48 + random.uniform(-5, 5), 
                                                            power_output / 48)
            
            # 5. Check cleaning schedule
            if self.cleaner.check_cleaning_needed():
                cleaning_result = self.cleaner.perform_cleaning()
                print(f"Cleaning performed: {cleaning_result}")
            else:
                # Accumulate dust
                self.tracker.accumulate_dust()
            
            # 6. Create monitoring data
            system_data = {
                "timestamp": current_time.isoformat(),
                "panel_azimuth": self.tracker.panel_azimuth,
                "panel_elevation": self.tracker.panel_elevation,
                "sun_azimuth": sun_azimuth,
                "sun_elevation": sun_elevation,
                "misalignment": misalignment,
                "irradiance": irradiance,
                "temperature": self.tracker.temperature,
                "dust_level": self.tracker.dust_level,
                "power_output": power_output,
                "optimal_voltage": optimal_voltage,
                "mppt_efficiency": power_output / (irradiance * 0.18)
            }
            
            # 7. Add energy forecast for next 24 hours
            forecast = []
            for j in range(1, 25):
                forecast_time = current_time + timedelta(hours=j)
                forecast.append({
                    "hour": forecast_time.hour,
                    "predicted_power": self.forecaster.predict_energy(forecast_time)
                })
            system_data["energy_forecast"] = forecast
            
            # 8. Publish to IoT dashboard
            self.monitor.publish(system_data)
            
            # 9. Print status
            print(f"Power: {power_output:.2f}W | Temp: {self.tracker.temperature:.1f}°C")
            print(f"Dust: {self.tracker.dust_level}% | MPPT Eff: {system_data['mppt_efficiency']:.2%}")
            
            # Simulate real-time interval
            time.sleep(1)
        
        self.monitor.disconnect()

# =====================================
# 7. DASHBOARD VISUALIZATION (SIMPLIFIED)
# =====================================
def visualize_system(system_data):
    """Visualize key system metrics"""
    df = pd.DataFrame(system_data)
    
    plt.figure(figsize=(15, 10))
    
    # Power output
    plt.subplot(2, 2, 1)
    plt.plot(df['timestamp'], df['power_output'], 'b-')
    plt.title('Power Output')
    plt.ylabel('Watts')
    plt.grid(True)
    
    # Solar alignment
    plt.subplot(2, 2, 2)
    plt.plot(df['timestamp'], df['misalignment'], 'r-')
    plt.title('Tracking Misalignment')
    plt.ylabel('Degrees')
    plt.grid(True)
    
    # Environmental factors
    plt.subplot(2, 2, 3)
    plt.plot(df['timestamp'], df['irradiance'], 'y-', label='Irradiance')
    plt.plot(df['timestamp'], df['temperature'], 'r-', label='Temperature')
    plt.title('Environmental Conditions')
    plt.ylabel('W/m² / °C')
    plt.legend()
    plt.grid(True)
    
    # Efficiency factors
    plt.subplot(2, 2, 4)
    plt.plot(df['timestamp'], df['dust_level'], 'g-', label='Dust Level')
    plt.plot(df['timestamp'], df['mppt_efficiency']*100, 'b-', label='MPPT Efficiency')
    plt.title('System Efficiency')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =====================================
# 8. RUN THE SYSTEM
# =====================================
if __name__ == "__main__":
    print("Starting Smart Solar Inverter System...")
    solar_system = SmartSolarSystem()
    
    # Run simulation for 72 hours (3 days)
    solar_system.run(hours=72)
    
    # For visualization, you would collect the data during the run
    # and pass it to visualize_system()
    print("Simulation complete!")
