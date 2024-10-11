from flask import Flask, request, render_template
from models import stock_prediction_and_analysis  
import yfinance as yf

app = Flask(__name__)

def validate_ticker(ticker):
    try:
        stock_data = yf.Ticker(ticker)
        if stock_data.history(period="1d").empty:
            raise ValueError("Invalid ticker")
        return stock_data
    except Exception as e:
        raise ValueError("Invalid ticker. Please enter a correct ticker.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form.get('ticker')  

        if not ticker:
            return render_template('form.html', error="Please enter a stock ticker.") 

        try:
            stock_data = validate_ticker(ticker)
        except ValueError as e:
            return render_template('form.html', error=str(e))  

        try:
            main_graph, last_day_prediction, next_7_days_graph, buy_percentage, sell_percentage, hold_percentage, decisions = stock_prediction_and_analysis(ticker)

            return render_template('form.html', 
                                   prediction=f"Prediction for {ticker}: ${last_day_prediction:.2f}",
                                   buy_percentage=f"{buy_percentage:.2f}",
                                   sell_percentage=f"{sell_percentage:.2f}",
                                   hold_percentage=f"{hold_percentage:.2f}",
                                   main_graph=main_graph, 
                                   next_7_days_graph=next_7_days_graph,
                                   decisions=decisions)
        except Exception as e:
            return render_template('form.html', error=f"An error occurred while processing your request: {str(e)}")

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)