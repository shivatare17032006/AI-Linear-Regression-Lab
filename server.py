from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from io import StringIO
import os

app = Flask(__name__)
CORS(app)

class LinearRegressionModel:
    def __init__(self):
        self.slope = 0
        self.intercept = 0
        self.x_data = []
        self.y_data = []
        self.x_param = ""
        self.y_param = ""
        self.column_headers = []
    
    def set_data(self, dataset, x_column, y_column, headers):
        self.x_param = x_column
        self.y_param = y_column
        self.column_headers = headers
        self.x_data = []
        self.y_data = []
        
        # Convert column names to indices
        try:
            x_col = int(x_column)  # If it's already an index
        except:
            x_col = headers.index(x_column)  # If it's a column name
        
        try:
            y_col = int(y_column)  # If it's already an index
        except:
            y_col = headers.index(y_column)  # If it's a column name
        
        print(f"X column: {x_col}, Y column: {y_col}")
        print(f"Dataset shape: {len(dataset)} rows")
        
        for row in dataset:
            if len(row) > max(x_col, y_col):
                try:
                    x_val = float(row[x_col])
                    y_val = float(row[y_col])
                    self.x_data.append(x_val)
                    self.y_data.append(y_val)
                except (ValueError, TypeError):
                    # Skip rows with invalid data
                    continue
        
        print(f"Successfully loaded {len(self.x_data)} valid data points")
        print(f"X data range: {min(self.x_data) if self.x_data else 'N/A'} to {max(self.x_data) if self.x_data else 'N/A'}")
        print(f"Y data range: {min(self.y_data) if self.y_data else 'N/A'} to {max(self.y_data) if self.y_data else 'N/A'}")
    
    def train(self):
        if not self.x_data or not self.y_data:
            raise ValueError("No valid data available for training")
        
        if len(self.x_data) < 2:
            raise ValueError("Need at least 2 data points for regression")
        
        x = np.array(self.x_data)
        y = np.array(self.y_data)
        
        # Calculate slope and intercept using least squares
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        
        denominator = (n * sum_xx - sum_x * sum_x)
        if denominator == 0:
            raise ValueError("Cannot perform regression: X values are constant")
        
        self.slope = (n * sum_xy - sum_x * sum_y) / denominator
        self.intercept = (sum_y - self.slope * sum_x) / n
        
        print(f"Trained model: slope={self.slope:.4f}, intercept={self.intercept:.4f}")
    
    def predict(self, x):
        return self.slope * x + self.intercept
    
    def calculate_mse(self):
        predictions = [self.predict(x) for x in self.x_data]
        mse = np.mean([(p - y) ** 2 for p, y in zip(predictions, self.y_data)])
        return mse
    
    def calculate_r_squared(self):
        y_mean = np.mean(self.y_data)
        ss_total = np.sum([(y - y_mean) ** 2 for y in self.y_data])
        ss_residual = np.sum([(y - self.predict(x)) ** 2 for x, y in zip(self.x_data, self.y_data)])
        
        if ss_total == 0:
            return 1.0  # Perfect fit if all y values are the same
        
        return 1 - (ss_residual / ss_total)
    
    def get_results(self):
        # Get data points for visualization (limit to 100 for performance)
        display_points = min(100, len(self.x_data))
        data_points = [{"x": self.x_data[i], "y": self.y_data[i]} for i in range(display_points)]
        
        # Generate regression line points
        if self.x_data:
            x_min, x_max = min(self.x_data), max(self.x_data)
            regression_line = [
                {"x": x_min, "y": self.predict(x_min)},
                {"x": x_max, "y": self.predict(x_max)}
            ]
        else:
            regression_line = []
        
        return {
            "equation": f"y = {self.slope:.6f} * x + {self.intercept:.6f}",
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "mse": float(self.calculate_mse()),
            "r_squared": float(self.calculate_r_squared()),
            "x_param": self.x_param,
            "y_param": self.y_param,
            "x_param_name": self.column_headers[int(self.x_param)] if self.x_param.isdigit() and int(self.x_param) < len(self.column_headers) else self.x_param,
            "y_param_name": self.column_headers[int(self.y_param)] if self.y_param.isdigit() and int(self.y_param) < len(self.column_headers) else self.y_param,
            "data_points": data_points,
            "regression_line": regression_line,
            "data_count": len(self.x_data)
        }

model = LinearRegressionModel()
current_dataset = []
column_headers = []

@app.route('/')
def home():
    return jsonify({
        "status": "Universal Linear Regression Server is running",
        "endpoints": ["/upload", "/train", "/predict", "/columns", "/set_parameters"],
        "message": "Accepts any CSV dataset with any columns"
    })

@app.route('/status')
def status():
    return jsonify({"status": "Server is running", "dataset_loaded": len(current_dataset) > 0})

@app.route('/columns', methods=['GET'])
def get_columns():
    columns = [{"index": i, "name": name} for i, name in enumerate(column_headers)]
    return jsonify(columns)

@app.route('/upload', methods=['POST'])
def upload_dataset():
    global current_dataset, column_headers
    
    try:
        data = request.get_json()
        if not data or 'csv_data' not in data:
            return jsonify({"error": "No CSV data provided"}), 400
        
        csv_text = data['csv_data']
        
        # Parse CSV
        df = pd.read_csv(StringIO(csv_text))
        column_headers = df.columns.tolist()
        current_dataset = df.values.tolist()
        
        print(f"Uploaded dataset: {len(current_dataset)} rows, {len(column_headers)} columns")
        print(f"Columns: {column_headers}")
        
        # Show sample of data
        if len(current_dataset) > 0:
            print(f"First row sample: {current_dataset[0]}")
        
        return jsonify({
            "status": "Dataset uploaded successfully",
            "rows": len(current_dataset),
            "columns": len(column_headers),
            "column_names": column_headers,
            "message": f"Loaded {len(current_dataset)} rows with {len(column_headers)} columns: {', '.join(column_headers)}"
        })
    
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/set_parameters', methods=['POST'])
def set_parameters():
    try:
        data = request.get_json()
        if not data or 'x_param' not in data or 'y_param' not in data:
            return jsonify({"error": "Missing parameters"}), 400
        
        x_param = data['x_param']
        y_param = data['y_param']
        
        print(f"Setting parameters: X={x_param}, Y={y_param}")
        print(f"Available columns: {column_headers}")
        
        model.set_data(current_dataset, x_param, y_param, column_headers)
        
        return jsonify({
            "status": "Parameters set successfully",
            "x_param": x_param,
            "y_param": y_param,
            "x_param_name": column_headers[int(x_param)] if x_param.isdigit() and int(x_param) < len(column_headers) else x_param,
            "y_param_name": column_headers[int(y_param)] if y_param.isdigit() and int(y_param) < len(column_headers) else y_param,
            "data_points": len(model.x_data)
        })
    
    except Exception as e:
        print(f"Parameter setting error: {str(e)}")
        return jsonify({"error": f"Parameter setting failed: {str(e)}"}), 500

@app.route('/train', methods=['POST'])
def train_model():
    try:
        model.train()
        results = model.get_results()
        results["status"] = "Model trained successfully"
        
        print(f"Model trained successfully: {results['equation']}")
        print(f"Performance - MSE: {results['mse']:.4f}, R²: {results['r_squared']:.4f}")
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'x' not in data:
            return jsonify({"error": "Missing x value for prediction"}), 400
        
        x_value = float(data['x'])
        prediction = model.predict(x_value)

        # Optional unit handling (kept minimal and backward compatible)
        x_unit = data.get('x_unit') if isinstance(data, dict) else None
        y_unit = data.get('y_unit') if isinstance(data, dict) else None

        # Light sanitization: trim and limit length to avoid noisy UI
        def _clean_unit(u):
            try:
                if u is None:
                    return None
                u = str(u).strip()
                if not u:
                    return None
                return u[:20]  # keep first 20 chars only
            except Exception:
                return None

        x_unit = _clean_unit(x_unit)
        y_unit = _clean_unit(y_unit)
        
        print(f"Prediction: {x_value}{(' ' + x_unit) if x_unit else ''} -> {prediction}{(' ' + y_unit) if y_unit else ''}")
        
        response = {
            "prediction": float(prediction),
            "input": float(x_value),
            "equation": model.get_results()["equation"]
        }
        # Include units only if provided to keep API backward compatible
        if x_unit is not None:
            response["x_unit"] = x_unit
        if y_unit is not None:
            response["y_unit"] = y_unit

        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/dataset_info', methods=['GET'])
def dataset_info():
    return jsonify({
        "rows": len(current_dataset),
        "columns": len(column_headers),
        "column_names": column_headers,
        "has_data": len(current_dataset) > 0
    })

if __name__ == '__main__':
    print("=== UNIVERSAL LINEAR REGRESSION SERVER ===")
    print("Server running at: http://localhost:3002")
    print("This server accepts ANY CSV dataset!")
    print("\nEndpoints:")
    print("  GET  /status       - Server status")
    print("  GET  /columns      - Get dataset columns")
    print("  GET  /dataset_info - Get dataset information")
    print("  POST /upload       - Upload CSV dataset")
    print("  POST /train        - Train regression model")
    print("  POST /predict      - Make prediction")
    print("  POST /set_parameters - Set X and Y parameters")
    print("\nWorkflow:")
    print("1. Upload any CSV file")
    print("2. Select any columns for X and Y")
    print("3. Train model")
    print("4. Make predictions and visualize")
    
    app.run(host='0.0.0.0', port=3002, debug=True)