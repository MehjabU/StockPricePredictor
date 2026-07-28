import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(dates, actual, actual_5, actual_20, prediction_1, prediction_5, prediction_20, test_rmse, test5_rmse, test20_rmse, ticker):
    # Convert (n, 1) arrays to (n,) for plotting
    actual = np.asarray(actual).reshape(-1)
    actual_5 = np.asarray(actual_5).reshape(-1)
    actual_20 = np.asarray(actual_20).reshape(-1)
    prediction_1 = np.asarray(prediction_1).reshape(-1)
    prediction_5 = np.asarray(prediction_5).reshape(-1)
    prediction_20 = np.asarray(prediction_20).reshape(-1)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(4, 1)

    # Price graph
    ax1 = fig.add_subplot(gs[:3, 0])

    ax1.plot(dates, actual, color="blue", label="Actual Price",)
    ax1.plot( dates, prediction_1, color="green", label="One-Day-Ahead Prediction",)
    ax1.plot( dates, prediction_5, color="purple", label="Five-Day-Ahead Prediction",)
    ax1.plot( dates, prediction_20, color="orange", label="Twenty-Day-Ahead Prediction",)

    ax1.set_title(f"{ticker} Stock Price Prediction")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(alpha=0.25)

    # Prediction-error graph
    prediction_error = np.abs(actual - prediction_1)
    prediction5_error = np.abs(actual_5 - prediction_5)
    prediction20_error = np.abs(actual_20 - prediction_20)

    ax2 = fig.add_subplot(gs[3, 0])
    ax2.axhline( test_rmse, color="blue", linestyle="--", label=f"RMSE: {test_rmse:.2f}",)
    ax2.plot( dates, prediction_error, color="red", label="One-Day Prediction Error",)

    ax2.axhline(test5_rmse, color='purple', linestyle='-.', label=f'5-Day RMSE: {test5_rmse:.2f}')
    ax2.plot(dates, prediction5_error, color='pink', label='Five-Day Prediction Error')

    ax2.axhline(test20_rmse, color='orange', linestyle=':', label=f'20-Day RMSE: {test20_rmse:.2f}')
    ax2.plot(dates, prediction20_error, color='black', label='Twenty-Day Prediction Error')

    ax2.set_title("Prediction Error")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Error")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    plt.show()
