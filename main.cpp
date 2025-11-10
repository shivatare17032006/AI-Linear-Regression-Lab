#include <iostream>
#include <cpprest/http_listener.h>
#include <cpprest/json.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;

class LinearRegressionModel {
private:
    double slope;
    double intercept;
    std::vector<double> x_data;
    std::vector<double> y_data;
    std::string x_param;
    std::string y_param;

public:
    LinearRegressionModel() : slope(0), intercept(0) {}

    void setData(const std::vector<std::vector<double>>& dataset, 
                 const std::string& x_column, const std::string& y_column) {
        x_param = x_column;
        y_param = y_column;
        x_data.clear();
        y_data.clear();
        
        int x_col = std::stoi(x_column);
        int y_col = std::stoi(y_column);
        
        for (const auto& row : dataset) {
            if (row.size() > std::max(x_col, y_col)) {
                x_data.push_back(row[x_col]);
                y_data.push_back(row[y_col]);
            }
        }
        
        std::cout << "Set data: " << x_data.size() << " points, X col: " << x_col << ", Y col: " << y_col << std::endl;
    }

    void train() {
        if (x_data.empty() || y_data.empty()) {
            throw std::runtime_error("No data available for training");
        }

        int n = x_data.size();
        double sum_x = std::accumulate(x_data.begin(), x_data.end(), 0.0);
        double sum_y = std::accumulate(y_data.begin(), y_data.end(), 0.0);
        double sum_xy = 0.0;
        double sum_xx = 0.0;

        for (int i = 0; i < n; i++) {
            sum_xy += x_data[i] * y_data[i];
            sum_xx += x_data[i] * x_data[i];
        }

        double x_mean = sum_x / n;
        double y_mean = sum_y / n;

        slope = (sum_xy - n * x_mean * y_mean) / (sum_xx - n * x_mean * x_mean);
        intercept = y_mean - slope * x_mean;
        
        std::cout << "Trained model: slope=" << slope << ", intercept=" << intercept << std::endl;
    }

    double predict(double x) const {
        return slope * x + intercept;
    }

    double calculateMSE() const {
        double mse = 0;
        int n = x_data.size();
        
        for (int i = 0; i < n; i++) {
            double prediction = predict(x_data[i]);
            double error = prediction - y_data[i];
            mse += error * error;
        }
        
        return mse / n;
    }

    double calculateRSquared() const {
        double y_mean = std::accumulate(y_data.begin(), y_data.end(), 0.0) / y_data.size();
        double ss_total = 0;
        double ss_residual = 0;
        
        for (int i = 0; i < y_data.size(); i++) {
            double prediction = predict(x_data[i]);
            ss_total += std::pow(y_data[i] - y_mean, 2);
            ss_residual += std::pow(y_data[i] - prediction, 2);
        }
        
        return 1 - (ss_residual / ss_total);
    }

    json::value getResults() const {
        json::value result;
        result[U("equation")] = json::value::string(U("y = ") + 
            utility::conversions::to_string_t(std::to_string(slope)) + 
            U(" * x + ") + utility::conversions::to_string_t(std::to_string(intercept)));
        result[U("slope")] = json::value::number(slope);
        result[U("intercept")] = json::value::number(intercept);
        result[U("mse")] = json::value::number(calculateMSE());
        result[U("r_squared")] = json::value::number(calculateRSquared());
        result[U("x_param")] = json::value::string(utility::conversions::to_string_t(x_param));
        result[U("y_param")] = json::value::string(utility::conversions::to_string_t(y_param));
        
        // Return data points for visualization
        json::value data_points = json::value::array();
        for (int i = 0; i < x_data.size() && i < 100; i++) { // Limit to 100 points for performance
            json::value point;
            point[U("x")] = json::value::number(x_data[i]);
            point[U("y")] = json::value::number(y_data[i]);
            data_points[i] = point;
        }
        result[U("data_points")] = data_points;
        
        return result;
    }
};

class RegressionServer {
private:
    http_listener listener;
    LinearRegressionModel model;
    std::vector<std::vector<double>> current_dataset;
    std::vector<std::string> column_headers;

public:
    RegressionServer(const utility::string_t& url) : listener(url) {
        listener.support(methods::GET, std::bind(&RegressionServer::handle_get, this, std::placeholders::_1));
        listener.support(methods::POST, std::bind(&RegressionServer::handle_post, this, std::placeholders::_1));
        listener.support(methods::PUT, std::bind(&RegressionServer::handle_put, this, std::placeholders::_1));
        listener.support(methods::DEL, std::bind(&RegressionServer::handle_delete, this, std::placeholders::_1));
    }

    void handle_get(http_request message) {
        auto path = http::uri::decode(message.relative_uri().path());
        
        std::cout << "GET request: " << utility::conversions::to_utf8string(path) << std::endl;
        
        if (path == U("/") || path == U("/status")) {
            json::value response;
            response[U("status")] = json::value::string(U("Linear Regression Server is running"));
            response[U("endpoints")] = json::value::string(U("/upload, /train, /predict, /columns"));
            message.reply(status_codes::OK, response);
        }
        else if (path == U("/columns")) {
            json::value response = json::value::array();
            for (int i = 0; i < column_headers.size(); i++) {
                json::value column;
                column[U("index")] = json::value::number(i);
                column[U("name")] = json::value::string(utility::conversions::to_string_t(column_headers[i]));
                response[i] = column;
            }
            message.reply(status_codes::OK, response);
        }
        else {
            message.reply(status_codes::NotFound, U("Not Found"));
        }
    }

    void handle_post(http_request message) {
        auto path = http::uri::decode(message.relative_uri().path());
        
        std::cout << "POST request: " << utility::conversions::to_utf8string(path) << std::endl;
        
        message.extract_json().then([=](json::value request) {
            if (path == U("/upload")) {
                handle_upload(message, request);
            }
            else if (path == U("/train")) {
                handle_train(message, request);
            }
            else if (path == U("/predict")) {
                handle_predict(message, request);
            }
            else if (path == U("/set_parameters")) {
                handle_set_parameters(message, request);
            }
            else {
                message.reply(status_codes::NotFound, U("Not Found"));
            }
        }).catch([=](std::exception_ptr e) {
            message.reply(status_codes::BadRequest, U("Invalid JSON"));
        });
    }

    void handle_put(http_request message) {
        message.reply(status_codes::MethodNotAllowed, U("Method Not Allowed"));
    }

    void handle_delete(http_request message) {
        message.reply(status_codes::MethodNotAllowed, U("Method Not Allowed"));
    }

    void handle_upload(http_request message, json::value request) {
        try {
            current_dataset.clear();
            column_headers.clear();
            
            if (request.has_field(U("csv_data"))) {
                auto csv_data = request[U("csv_data")].as_string();
                std::string csv_string = utility::conversions::to_utf8string(csv_data);
                std::stringstream ss(csv_string);
                std::string line;
                
                // Read headers
                if (std::getline(ss, line)) {
                    std::stringstream header_ss(line);
                    std::string cell;
                    while (std::getline(header_ss, cell, ',')) {
                        column_headers.push_back(cell);
                    }
                }
                
                // Read data rows
                int row_count = 0;
                while (std::getline(ss, line) && row_count < 1000) { // Limit to 1000 rows
                    if (line.empty()) continue;
                    
                    std::stringstream row_ss(line);
                    std::string cell;
                    std::vector<double> row;
                    
                    while (std::getline(row_ss, cell, ',')) {
                        try {
                            row.push_back(std::stod(cell));
                        } catch (...) {
                            row.push_back(0.0);
                        }
                    }
                    
                    if (!row.empty()) {
                        current_dataset.push_back(row);
                        row_count++;
                    }
                }
                
                json::value response;
                response[U("status")] = json::value::string(U("Dataset uploaded successfully"));
                response[U("rows")] = json::value::number(current_dataset.size());
                response[U("columns")] = json::value::number(column_headers.size());
                response[U("message")] = json::value::string(U("Loaded ") + 
                    utility::conversions::to_string_t(std::to_string(current_dataset.size())) + 
                    U(" rows with ") + utility::conversions::to_string_t(std::to_string(column_headers.size())) + 
                    U(" columns"));
                
                std::cout << "Uploaded dataset: " << current_dataset.size() << " rows, " 
                          << column_headers.size() << " columns" << std::endl;
                
                message.reply(status_codes::OK, response);
            } else {
                message.reply(status_codes::BadRequest, U("No CSV data provided"));
            }
        } catch (const std::exception& e) {
            json::value error;
            error[U("error")] = json::value::string(U("Upload failed: ") + 
                utility::conversions::to_string_t(e.what()));
            message.reply(status_codes::InternalError, error);
        }
    }

    void handle_set_parameters(http_request message, json::value request) {
        try {
            if (!request.has_field(U("x_param")) || !request.has_field(U("y_param"))) {
                message.reply(status_codes::BadRequest, U("Missing parameters"));
                return;
            }
            
            std::string x_param = utility::conversions::to_utf8string(request[U("x_param")].as_string());
            std::string y_param = utility::conversions::to_utf8string(request[U("y_param")].as_string());
            
            model.setData(current_dataset, x_param, y_param);
            
            json::value response;
            response[U("status")] = json::value::string(U("Parameters set successfully"));
            response[U("x_param")] = request[U("x_param")];
            response[U("y_param")] = request[U("y_param")];
            response[U("data_points")] = json::value::number(model.getResults()[U("data_points")].as_array().size());
            
            message.reply(status_codes::OK, response);
        } catch (const std::exception& e) {
            json::value error;
            error[U("error")] = json::value::string(U("Parameter setting failed: ") + 
                utility::conversions::to_string_t(e.what()));
            message.reply(status_codes::InternalError, error);
        }
    }

    void handle_train(http_request message, json::value request) {
        try {
            model.train();
            json::value results = model.getResults();
            results[U("status")] = json::value::string(U("Model trained successfully"));
            
            std::cout << "Model trained: " << utility::conversions::to_utf8string(results[U("equation")].as_string()) << std::endl;
            
            message.reply(status_codes::OK, results);
        } catch (const std::exception& e) {
            json::value error;
            error[U("error")] = json::value::string(U("Training failed: ") + 
                utility::conversions::to_string_t(e.what()));
            message.reply(status_codes::InternalError, error);
        }
    }

    void handle_predict(http_request message, json::value request) {
        try {
            if (!request.has_field(U("x"))) {
                message.reply(status_codes::BadRequest, U("Missing x value for prediction"));
                return;
            }
            
            double x_value = request[U("x")].as_double();
            double prediction = model.predict(x_value);
            
            json::value response;
            response[U("prediction")] = json::value::number(prediction);
            response[U("input")] = json::value::number(x_value);
            response[U("equation")] = model.getResults()[U("equation")];
            
            std::cout << "Prediction: " << x_value << " -> " << prediction << std::endl;
            
            message.reply(status_codes::OK, response);
        } catch (const std::exception& e) {
            json::value error;
            error[U("error")] = json::value::string(U("Prediction failed: ") + 
                utility::conversions::to_string_t(e.what()));
            message.reply(status_codes::InternalError, error);
        }
    }

    void start() {
        listener.open().wait();
        std::wcout << U("Server started at: ") << listener.uri().to_string() << std::endl;
    }

    void stop() {
        listener.close().wait();
    }
};

// Enable CORS
void handle_options(http_request message) {
    http_response response(status_codes::OK);
    response.headers().add(U("Access-Control-Allow-Origin"), U("*"));
    response.headers().add(U("Access-Control-Allow-Methods"), U("GET, POST, OPTIONS"));
    response.headers().add(U("Access-Control-Allow-Headers"), U("Content-Type"));
    message.reply(response);
}

int main() {
    utility::string_t url = U("http://localhost:3002");
    RegressionServer server(url);
    
    try {
        server.start();
        std::wcout << U("=== Linear Regression Server ===") << std::endl;
        std::wcout << U("Server running at: http://localhost:3002") << std::endl;
        std::wcout << U("Endpoints:") << std::endl;
        std::wcout << U("  GET  /status   - Server status") << std::endl;
        std::wcout << U("  GET  /columns  - Get dataset columns") << std::endl;
        std::wcout << U("  POST /upload   - Upload CSV dataset") << std::endl;
        std::wcout << U("  POST /train    - Train regression model") << std::endl;
        std::wcout << U("  POST /predict  - Make prediction") << std::endl;
        std::wcout << U("Press Enter to stop the server...") << std::endl;
        
        std::string line;
        std::getline(std::cin, line);
        
        server.stop();
        std::wcout << U("Server stopped.") << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}