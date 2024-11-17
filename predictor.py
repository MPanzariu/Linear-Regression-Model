import numpy as np
import pickle
try:
    with open("linear_regression_params.pkl", "rb") as f:
        saved_params = pickle.load(f)
        final_slope = saved_params["slope"]
        final_bias = saved_params["bias"]
        mean_G1 = saved_params["mean_G1"]
        mean_G2 = saved_params["mean_G2"]
        mean_studytime = saved_params["mean_studytime"]
        mean_failures = saved_params["mean_failures"]
        mean_absences = saved_params["mean_absences"]
        print("Loaded saved molde parameters.")
except:
    raise ExceptionType("No parameters")
def predict(G1, G2, studytime, failures, absences):
    p_G1 = mean_G1 if G1 is None else G1
    p_G2 = mean_G2 if G2 is None else G2
    p_studytime = mean_studytime if studytime is None else studytime
    p_failures = mean_failures if failures is None else failures
    p_absences = mean_absences if absences is None else absences
    x = [p_G1, p_G2, p_studytime, p_failures, p_absences]
    return np.dot(x, final_slope) + final_bias
