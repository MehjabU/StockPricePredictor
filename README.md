# Stock Price Predictor

A PyTorch stock-price forecasting project that uses Long Short-Term Memory (LSTM)
neural networks to compare one-day, five-day, and twenty-day prediction horizons.
Historical market data is downloaded from Yahoo Finance, transformed into rolling
sequences, and divided chronologically into training and testing sets.

The application plots each model's predictions against actual closing prices and
displays the absolute prediction error and root mean squared error (RMSE).

## Acknowledgements

This project was heavily inspired by the following YouTube tutorial. The overall 
design and implementation approach are based on the concepts presented in the 
video.

- https://www.youtube.com/watch?v=IJ50ew8wi-0

## Features

- Downloads current historical stock data with `yfinance`
- Uses separate LSTM models for 1-day, 5-day, and 20-day forecasts
- Uses 90-day rolling data windows
- Preserves time order with an 80/20 training and testing split
- Automatically uses CUDA when a compatible GPU is available
- Converts normalized predictions back into stock-price values
- Compares model performance using RMSE
- Displays predictions and errors in a Matplotlib figure

## Project Structure

```text
StockPricePredictor/
├── Main.py       # Data preparation, model training, and evaluation
├── Model.py      # PyTorch LSTM model
├── Plot.py       # Prediction and error visualization
└── README.md
```

## Technologies

- Python
- NumPy
- pandas
- yfinance
- PyTorch
- scikit-learn
- Matplotlib

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/StockPricePredictor.git
   cd StockPricePredictor
   ```

2. Create and activate a virtual environment:

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install numpy pandas yfinance matplotlib torch scikit-learn
   ```

## Usage

Run the project from its root directory:

```bash
python Main.py
```

An internet connection is required to download market data. Training begins after
the data is downloaded, and the prediction graph appears when all three models
finish training.

### Changing the stock

Change the ticker in `Main.py`:

```python
ticker = "MSFT"
```

For example:

```python
ticker = "AAPL"
```

### Changing the starting date

Change the date passed to `yf.download`:

```python
df = yf.download(ticker, "2023-01-01")
```

### Changing the training time

Adjust the number of training epochs:

```python
num_epochs = 20
```

More epochs can increase training time and do not guarantee better performance.

## How It Works

1. Historical closing prices are downloaded for the selected ticker.
2. Prices are standardized with `StandardScaler`.
3. The data is converted into overlapping 90-day windows.
4. Separate datasets are created for the three forecast horizons.
5. Three LSTM models are trained using mean squared error and the Adam optimizer.
6. Predictions are converted back to their original price scale.
7. RMSE and absolute prediction errors are calculated.
8. Actual prices, predictions, and errors are plotted.

## Model Architecture

Each forecasting model contains:

- Two LSTM layers
- 32 hidden units
- A dropout rate of 0.15
- A fully connected output layer

The model uses the final LSTM timestep to produce one predicted closing price.

## Possible Improvements

- Fit the scaler only on the training period to prevent test-data leakage
- Add naive and moving-average baselines
- Use walk-forward validation
- Include volume and technical indicators as model inputs
- Save trained models instead of retraining on every run
- Add confidence intervals around predictions
- Accept ticker, date, and epoch settings as command-line arguments
- Record directional accuracy in addition to RMSE

## Disclaimer

This project is for educational purposes only. Its predictions are experimental
and should not be treated as financial or investment advice. Historical
performance does not guarantee future results.
