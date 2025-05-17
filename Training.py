# 📈 Enhanced Stock Prediction System v2.0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import talib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import lightgbm as lgb
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuration
SEQ_LENGTH = 21  # Optimal for market patterns
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.4

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True  # Ensure CUDA operations are deterministic
torch.backends.cudnn.benchmark = False
# ---------------------------
# 1. Advanced Feature Engineering
# ---------------------------
def feature_analysis(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")

def add_market_features(df):
    """Generate non-leaking technical features"""
    df = df.copy()
    close = df['Close'].values

    def compute_bb(group):
       upper, middle, lower = talib.BBANDS(group['Close'], timeperiod=20)
       group['Upper_BB'] = upper
       group['Middle_BB'] = middle
       group['Lower_BB'] = lower
       group['BB_Width'] = (upper - lower) / middle
       return group

    df = df.groupby('ticker', group_keys=False).apply(compute_bb)
    
    # Price Transformations
    df['RSI_14'] = talib.RSI(close, timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(close)
    df['ATR_14'] = talib.ATR(df['High'], df['Low'], close, 14)
    df['Returns_3d'] = df['Close'].pct_change(3).shift(1)
    
    # Market Structure Features
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(21).mean()
    df['Overnight_Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['Volatility_21d'] = df.groupby('ticker')['Close'].transform(
    lambda x: x.pct_change().rolling(21, min_periods=15).std().shift(1)
)
   
    
    # Sentiment Enhancements
    df['combined_sentiment'] = df['sentiment_score'] * 0.6 + df['tweet_sentiment_score'] * 0.4
    df['sentiment_ma5'] = df['combined_sentiment'].rolling(5).mean()
    df['sentiment_trend'] = (df['combined_sentiment'] > df['sentiment_ma5']).astype(int)
    df['sentiment_velocity'] = df['combined_sentiment'] - df['combined_sentiment'].rolling(3).mean()
    
    # 4. Volume-Weighted Overnight Gap
    df['Gap_Volume_Adj'] = df['Overnight_Gap'] * df['Relative_Volume']
    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    return df

    
def create_technical_features(df):
    # Lagged features (prevents lookahead)
    df['EMA_20'] = ta.EMA(df['Close'], timeperiod=20).shift(1)
    df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14).shift(1)
    
    # Candlestick patterns (using talib abstract functions)
    df['CDL_ENGULFING'] = ta.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close']).shift(1)
    df['CDL_MORNINGSTAR'] = ta.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close']).shift(1)
    
    # Volume-based features
    df['Volume_ROC'] = df['Volume'].pct_change(3).shift(1)
    
    # Clean data pipeline
    return df.dropna()

# ---------------------------
# 2. Enhanced LSTM with Attention
# ---------------------------
class MarketLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, num_layers=1, batch_first=True)
        self.attention = nn.MultiheadAttention(64, 2)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1))
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.permute(1, 0, 2)
        attn_out, _ = self.attention(out, out, out)
        return torch.sigmoid(self.fc(attn_out[-1]))

    def predict(self, df, features, sequence_length=10, device=None):
        
        import numpy as np
        import torch
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        X = df[features].values.astype(np.float32)
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i+sequence_length])
        if not sequences:
            return np.array([])
        X_seq = np.stack(sequences)
        X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = self.forward(X_seq)
            preds = outputs.cpu().numpy().flatten()
        # Pad the beginning with NaNs to align with original df length
        pad = [np.nan] * (sequence_length - 1)
        preds_full = np.concatenate([pad, preds])
        return preds_full

# ---------------------------
# 3. Hybrid Modeling Pipeline
# ---------------------------
def create_temporal_features(df):
    """Create temporal-aware features"""
    df = add_market_features(df)
    
    # Ensure 'Date' is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Target: Next day return direction
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Remove raw price features
    features = [
        'Relative_Volume', 'Overnight_Gap', 'RSI_14',
        'MACD', 'ATR_14', 'Returns_3d','Volatility_21d',
        'combined_sentiment', 'sentiment_ma5', 'sentiment_trend',
        'Upper_BB','BB_Width','sentiment_velocity','Gap_Volume_Adj'
    ]
    # Ensure price columns are retained for downstream analysis
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    keep_cols = features + price_cols + ['Target', 'Date']
    keep_cols = [col for col in keep_cols if col in df.columns]
    return df[keep_cols].dropna()

def temporal_train_test_split(df, test_days=60):
    """Time-aware data splitting"""
    dates = pd.to_datetime(df['Date'].unique())
    cutoff = dates[-test_days]
    return (
        df[df['Date'] < cutoff],
        df[df['Date'] >= cutoff]
    )

# ---------------------------
# 4. Optimized Training Process
# ---------------------------
def train_hybrid_model(train_df, val_df):
    """Train optimized hybrid model"""

    # Separate technical and sentiment features
    technical_features = [
        'Relative_Volume', 'Overnight_Gap', 'RSI_14', 'MACD', 
        'ATR_14', 'Volatility_21d', 'Returns_3d', 'BB_Width', 'Gap_Volume_Adj'
    ]
    sentiment_features = [
        'combined_sentiment', 'sentiment_ma5', 'sentiment_trend', 'sentiment_velocity'
    ]

    # LightGBM with Feature Selection (only on technical features)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'num_leaves': 127,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 50,
        'seed': 42,
        'num_threads': 1 
    }

    selector = RFECV(
        estimator=lgb.LGBMClassifier(**lgb_params),
        step=1,
        cv=TimeSeriesSplit(3),
        scoring='roc_auc'
    )
    selector.fit(train_df[technical_features], train_df['Target'])
    selected_technical = train_df[technical_features].columns[selector.support_]
    print(f"Selected technical features ({len(selected_technical)}): {list(selected_technical)}")

    # Always include all sentiment features
    final_features = list(selected_technical) + sentiment_features
    print(f"Final features used for training: {final_features}")

    # Train Final LGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(train_df[final_features], train_df['Target'])

    # LSTM Training
    def create_sequences(data, features, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[features].iloc[i:i+seq_length].values)
            y.append(data['Target'].iloc[i+seq_length])
        return np.array(X), np.array(y)

    X_train_seq, y_train_seq = create_sequences(train_df, final_features, SEQ_LENGTH)
    X_val_seq, y_val_seq = create_sequences(val_df, final_features, SEQ_LENGTH)

    pos_weight = torch.tensor([(len(y_train_seq) - sum(y_train_seq)) / sum(y_train_seq)]).to(DEVICE)
    criterion = nn.BCELoss()
    lstm_model = MarketLSTM(input_size=len(final_features))
    lstm_model = lstm_model.to(DEVICE)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    # Train loop
    for epoch in range(15):
        lstm_model.train()
        for i in range(0, len(X_train_seq), 64):
            batch_x = torch.FloatTensor(X_train_seq[i:i+64]).to(DEVICE)
            batch_y = torch.FloatTensor(y_train_seq[i:i+64]).unsqueeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = lstm_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        lstm_model.eval()
        with torch.no_grad():
            val_preds = lstm_model(torch.FloatTensor(X_val_seq).to(DEVICE)).cpu().numpy()
            val_acc = accuracy_score(y_val_seq, val_preds > 0.5)
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
    
    return lgb_model, lstm_model, final_features

# ---------------------------
# 5. Enhanced Evaluation
# ---------------------------
def backtest_strategy(df, model, features, sequence_length=10,threshold=0.2):
    """
    Backtest trading strategy using model predictions.
    """
    preds = model.predict(df, features, sequence_length=sequence_length)
    df['Prediction'] = preds
    # Optionally, handle NaNs in 'Prediction' column as needed
    df['Returns'] = (df['Close'] - df['Open']) / df['Open']   
    df['Strategy'] = df['Prediction'].shift(1) * df['Returns']
    df['Cumulative_Market'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy']).cumprod()
    df['Signal'] = (df['Prediction'] > threshold).astype(int)

    
    print("\n💰 Strategy Performance:")
    print(f"Sharpe Ratio: {df['Strategy'].mean()/df['Strategy'].std()*np.sqrt(252):.2f}")
    print("Number of trades:", df['Signal'].sum())

    
    plt.figure(figsize=(12,6))
    plt.title("Cumulative Returns")
    plt.plot(df['Date'], (1 + df['Returns']).cumprod() - 1, label='Buy & Hold')
    plt.plot(df['Date'], (1 + df['Strategy']).cumprod() - 1, label='Strategy')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model(model, X, y):
    """Evaluate classification model performance"""
    preds = model.predict(X)
    if hasattr(preds, "toarray"):  # For LGBM, sometimes output is sparse
        preds = preds.toarray().flatten()
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, preds)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(classification_report(y, preds))

# ---------------------------
# Modified Main Execution Flow
# ---------------------------
if __name__ == "__main__":
    # Load and prepare data
    tickers = ['aapl', 'goog', 'intc', 'mstr', 'nvda', 'tsla']
    full_df = pd.concat(
        [pd.read_csv(f'Results/{ticker}/{ticker}_train.csv').assign(ticker=ticker)
         for ticker in tickers]
    ).sort_values('Date')
    
    # Convert to datetime and filter date range
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df = full_df[(full_df['Date'] >= '2020-01-01') & 
                      (full_df['Date'] <= pd.to_datetime('today'))]
    
    # Feature Engineering
    processed_df = create_temporal_features(full_df)
    
    # Get sorted unique dates
    all_dates = processed_df['Date'].unique()
    all_dates.sort()
    
    # Walk-Forward Parameters
    initial_train_days = 252 * 4  # 3 years initial training
    retrain_interval = 21  # Days between model retraining
    
    # Storage for results
    all_preds = []
    all_targets = []
    portfolio_values = [1.0]  # Starting with $1
    
    # Walk-Forward Loop
    for day in range(initial_train_days, len(all_dates)):
        current_date = all_dates[day]
        
        # 1. Get training data (expanding window)
        train_mask = processed_df['Date'] < current_date
        train_data = processed_df[train_mask]
        
        # 2. Get test data (next trading day)
        test_mask = processed_df['Date'] == current_date
        test_data = processed_df[test_mask]
        
        # Only retrain periodically to reduce computation
        if (day - initial_train_days) % retrain_interval == 0:
            # Split recent data for validation
            
            val_cutoff = current_date - pd.Timedelta(days=45)
            recent_train = train_data[train_data['Date'] < val_cutoff]
            val_data = train_data[train_data['Date'] >= val_cutoff]
            
            # Retrain models
            print(f"\n🔁 Retraining models for {pd.Timestamp(current_date).strftime('%Y-%m-%d')}")
            lgb_model, lstm_model, features = train_hybrid_model(recent_train, val_data)
        
        # 3. Make predictions
        # LightGBM predictions
        lgb_preds = lgb_model.predict_proba(test_data[features])[:,1]
        
        # LSTM predictions (requires sequence data)
        lstm_input = train_data.append(test_data).tail(SEQ_LENGTH*2)
        lstm_preds = lstm_model.predict(lstm_input, features, SEQ_LENGTH)
        lstm_preds = lstm_preds[-len(test_data):]
        
        # Ensemble with threshold
        lgb_signal = (lgb_preds > THRESHOLD).astype(int)
        lstm_signal = (lstm_preds > THRESHOLD).astype(int)
        final_signal = ((0.2 * lstm_signal + 0.8 * lgb_signal) > THRESHOLD).astype(int)
        
        # Store results
        all_preds.extend(final_signal.astype(int))
        all_targets.extend(test_data['Target'].values)
        
        # 4. Update portfolio value (align logic with backtest)
        # Use the same logic as in backtest_strategy for position and returns
        for idx, row in test_data.iterrows():
            entry_price = row['Open']
            exit_price = row['Close']
            daily_return = (exit_price - entry_price) / entry_price
            position = final_signal[test_data.index.get_loc(idx)]
            if np.any(position == 1):
                portfolio_values.append(portfolio_values[-1] * (1 + daily_return - 0.001))
            else:
                portfolio_values.append(portfolio_values[-1])
        
        print(f"Date: {pd.Timestamp(current_date).strftime('%Y-%m-%d')} | "
              f"Accuracy: {accuracy_score(all_targets, all_preds):.2f} | "
              f"Portfolio: {portfolio_values[-1]:.2f}x | Trades so far: {np.sum(np.array(all_preds)==1)}")

    def save_production_models():
        """Add this to your existing training code"""
        torch.save(lstm_model.state_dict(), 'models/lstm.pth')
        joblib.dump(lgb_model, 'models/lgb.pkl')    
    save_production_models()
    # Calculate baseline accuracy for both targets
    print("Interday Random Baseline:", processed_df['Close'].pct_change().shift(-1).gt(0).mean())
    print("Intraday Random Baseline:", (processed_df['Close'] > processed_df['Open']).mean())   

    # Final Evaluation
    print("\n🎯 Final Walk-Forward Performance (Training):")
    print(f"Total Portfolio Growth: {portfolio_values[-1]-1:.1%}")
    print(classification_report(all_targets, all_preds))

    print("\n📊 Final Model Analysis:")
    final_test_period = processed_df[processed_df['Date'] >= all_dates[initial_train_days]]
    backtest_strategy(final_test_period, lstm_model, features, SEQ_LENGTH, threshold=THRESHOLD)
    
    # Add benchmark comparison
    benchmark_return = (final_test_period['Close'].iloc[-1] / 
                       final_test_period['Close'].iloc[0] - 1) * 100
    print(f"\n📉 Buy & Hold Return: {benchmark_return:.1f}%")

    feature_analysis(lgb_model, train_data[features])

    # Plot performance
    plt.figure(figsize=(12,6))
    # Ensure the lengths of all_dates and portfolio_values match
    if len(all_dates[initial_train_days:]) != len(portfolio_values[1:]):
        # Adjust portfolio_values to match the length of all_dates
        portfolio_values = portfolio_values[:len(all_dates[initial_train_days:])+1]
    
    plt.plot(pd.to_datetime(all_dates[initial_train_days:]), portfolio_values[1:])
    plt.title("Walk-Forward Portfolio Growth")
    plt.xlabel("Date")
    plt.ylabel("Value (Multiple of Initial)")
    plt.show()

# 1. Add special occasion features
# 1. Data loading and filtering for 2025 onwards
sales_df = pd.read_csv('dataset_script/sales_data.csv')
sales_df['Order Date'] = pd.to_datetime(sales_df['Order Date'])
sales_df = sales_df[(sales_df['Order Date'] >= '2025-01-01') & (sales_df['Order Date'] <= pd.to_datetime('today'))]

# 2. (Optional) Filter for the specific toy if needed
# sales_df = sales_df[sales_df['Product Name'].str.contains('popmart', case=False, na=False)]

# 3. Special occasion features (update for 2025)
SPECIAL_DATES = [
    '2025-01-01',  # New Year
    '2025-02-14',  # Valentine's Day
    '2025-03-03', '2025-04-04', '2025-05-05', # Double number days
    # Add more: Mother's Day, Malaysia-specific holidays, etc.
]
def add_special_occasion_features(df):
    df = df.copy()
    df['is_special_occasion'] = df['Order Date'].dt.strftime('%Y-%m-%d').isin(SPECIAL_DATES).astype(int)
    return df
sales_df = add_special_occasion_features(sales_df)

# 4. Feature engineering for sales prediction
sales_df = sales_df.sort_values('Order Date')
sales_df['sales_lag_1'] = sales_df['Total Amount'].shift(1)
sales_df['sales_lag_3'] = sales_df['Total Amount'].rolling(3).sum().shift(1)
sales_df['sales_lag_5'] = sales_df['Total Amount'].rolling(5).sum().shift(1)
sales_df['sales_lag_7'] = sales_df['Total Amount'].rolling(7).sum().shift(1)
sales_df['day_of_week'] = sales_df['Order Date'].dt.dayofweek
sales_df['month'] = sales_df['Order Date'].dt.month

# 5. Multi-horizon targets
sales_df['sales_next_1d'] = sales_df['Total Amount'].shift(-1)
sales_df['sales_next_3d'] = sales_df['Total Amount'].rolling(3).sum().shift(-3)
sales_df['sales_next_5d'] = sales_df['Total Amount'].rolling(5).sum().shift(-5)
sales_df['sales_next_7d'] = sales_df['Total Amount'].rolling(7).sum().shift(-7)
sales_df = sales_df.dropna()

# 6. Adjust train/validation split for short time range (e.g., 60-90 days for validation)
def temporal_train_test_split(df, test_days=60):
    dates = pd.to_datetime(df['Order Date'].unique())
    cutoff = dates[-test_days]
    return (
        df[df['Order Date'] < cutoff],
        df[df['Order Date'] >= cutoff]
    )

# 7. (Optional) If you want to focus on recent trends, use a rolling window for training (e.g., last 90-180 days)
# Example:
# train_df = sales_df[sales_df['Order Date'] >= (sales_df['Order Date'].max() - pd.Timedelta(days=180))]

# ... rest of your model code (LSTM, training, etc.) ...

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import timedelta

# --- LSTM for Sales Forecasting ---

class SalesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_sales_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    # Aggregate sales per day
    daily_sales = df.groupby('Date')['UnitsSold'].sum().sort_index()
    return daily_sales

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

def train_sales_lstm(series, seq_length=30, epochs=30):
    data = series.values.astype(np.float32)
    xs, ys = create_sequences(data, seq_length)
    xs = torch.tensor(xs).unsqueeze(-1)
    ys = torch.tensor(ys).unsqueeze(-1)
    model = SalesLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(xs)
        loss = loss_fn(output, ys)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return model

def predict_future(model, series, seq_length, days_ahead):
    model.eval()
    data = series.values.astype(np.float32)
    input_seq = torch.tensor(data[-seq_length:]).unsqueeze(0).unsqueeze(-1)
    preds = []
    for _ in range(days_ahead):
        with torch.no_grad():
            pred = model(input_seq)
        preds.append(pred.item())
        input_seq = torch.cat([input_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(-1)], dim=1)
    return preds

# --- Usage Example ---

if __name__ == "__main__":
    sales_series = load_sales_data("./data/Sales.csv")
    seq_len = 30
    model = train_sales_lstm(sales_series, seq_length=seq_len, epochs=30)
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3, 5, 7]:
        preds = predict_future(model, sales_series, seq_len, days)
        print(f"Predicted sales for next {days} day(s): {preds}")
    for days in [1, 3
