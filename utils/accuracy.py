from sklearn.metrics import mean_absolute_error

# Example
predicted_speeds = [110.2, 104.5, 115.3]
ground_truth_speeds = [111.0, 105.0, 113.0]

mae = mean_absolute_error(ground_truth_speeds, predicted_speeds)
print("MAE:", mae)
