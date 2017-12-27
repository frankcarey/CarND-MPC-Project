#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/Geometry"
#include "MPC.h"
#include "json.hpp"

using namespace Eigen;
using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }

double deg2rad(double x) { return x * pi() / 180; }

double rad2deg(double x) { return x * 180 / pi(); }

// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;
// 100ms of latency + ~50ms for the solver.
const double latency = 0.150;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}


std::vector<double> convertPt(double global_x, double global_y, double px, double py, double psi) {
  // Convert world space coordinates into car space coordinates.
  double vehicle_x = (global_x - px) * cos(psi) + (global_y - py) * sin(psi);
  double vehicle_y = -(global_x - px) * sin(psi) + (global_y - py) * cos(psi);
  return {vehicle_x, vehicle_y};
}

Eigen::VectorXd predict(Eigen::VectorXd state, double delay,
                        double current_steer, double current_throttle, Eigen::VectorXd coeffs, double Lf) {

  // This discussion was helpful in figuring out how to implement the latency.
  // https://discussions.udacity.com/t/how-to-incorporate-latency-into-the-model/257391/63?u=fcarey

  // current state
  double x0 = state[0];
  double y0 = state[1];
  double psi0 = state[2];
  double v0 = state[3];
  // state after delay
  double x = x0 + v0 * cos(psi0) * delay;
  double y = y0 + v0 * sin(psi0) * delay;
  double psi = psi0 - v0 * current_steer * delay / Lf;
  double v = v0; // Assume a constant velocity considering the short delay.
  double epsi = -atan(coeffs[1]) + psi;
  double cte = polyeval(coeffs, 0) - y0 + v0 * sin(epsi) * delay;

  Eigen::VectorXd return_state(6);
  return_state << x, y, psi, v, cte, epsi;

  return return_state;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          // Convert miles per hour into meters per second!
          v *= 0.44704;
          double current_steering = j[1]["steering_angle"];
          double current_throttle = j[1]["throttle"];

          /*
          * TODO: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          double steer_value;
          double throttle_value;

          Eigen::VectorXd x_way_pts(ptsx.size());
          Eigen::VectorXd y_way_pts(ptsy.size());

          Eigen::VectorXd state(6);

          // Transform the way points from world space to the future
          // car-local space to make calculations easier.
          for (unsigned int i = 0; i < ptsx.size(); i++) {
            auto transformed = convertPt(ptsx[i], ptsy[i], px, py, psi);
            x_way_pts[i] = transformed[0];
            y_way_pts[i] = transformed[1];
          }

          // Fit the waypoints to a line.
          auto coeffs = polyfit(x_way_pts, y_way_pts, 3);

          // Get the CTE as the y offset at 0.
          // Note: the CTE in car space is relative to the car, so we don't need to
          // subtract py from the value returned.
          double cte = polyeval(coeffs, 0);
          // Similarly, in car space, psi is equal to zero, so it doesnt' need to
          // be added to epsi.
          double epsi = -atan(coeffs[1]);

          state << 0, // px for car-space is zero.
              0, // py for car-space is zero.
              0, // psi for car-space is zero.
              v,
              cte,
              epsi;

          // Handle latency by predicting what the state will be at "latency" amount of time in the future.
          auto future_state = predict(state, latency, current_steering, current_throttle, coeffs, Lf);
          auto future_solution = mpc.Solve(future_state, coeffs);

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          steer_value = future_solution[0] / deg2rad(25);
          throttle_value = future_solution[1];
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
          int N = int(future_solution.size() / 2 - 1);
          for (int i = 0; i < N; i++) {
            mpc_x_vals.push_back(future_solution[i + 2]);
            mpc_y_vals.push_back(future_solution[i + N + 2]);
          }


          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line
          double x_increment = 5;
          int num_points = 10;
          for (double i = 0; i < num_points; i++) {
            next_x_vals.push_back(i * x_increment);
            next_y_vals.push_back(polyeval(coeffs, i * x_increment));
          }
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          //std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
