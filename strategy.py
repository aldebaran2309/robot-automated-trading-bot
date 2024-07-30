import json
import pprint
import pandas as pd
import operator

from datetime import datetime 
from datetime import timedelta
from configparser import ConfigParser

from pyrobot.trades import Trade
from pyrobot.robot import PyRobot
from pyrobot.indicators import Indicators
from samples.trading_robot_indicators import ACCOUNT_NUMBER

config = ConfigParser()
config.read("configs/config.ini")

CLIENT_ID = config.get("main " , "CLIENT_ID")
REDIRECT_URI =  config.get("main" , "REDIRECT_URI")
CREDENTIALS_PATH = config.get("main", "CREDENTIALS_PATH")
ACCOUNT_NUMBER =  config.get("main", "ACCOUNT_NUMBER")

trading_robot =  PyRobot(
    client_id=  CLIENT_ID,
    redirect_uri= REDIRECT_URI,
    credentials_path= CREDENTIALS_PATH,
    trading_account= ACCOUNT_NUMBER,
    paper_trading= False
    
    )
 
trading_robot_portfolio =  trading_robot.create_portfolio()

trading_symbol = 'FCEL'

trading_robot_portfolio.add_position(
    symbol =  trading_symbol,
    asset_type= 'equity'
   )

#grab the historical prices , define start and end date
start_date  =  datetime.today()
end_date =  start_date -  timedelta(days = 30)

historical_prices =  trading_robot.grab_historical_prices(
    start =  start_date,
    end =  end_date,
    bar_size =  1,
    bar_type= 'minute'
    )

print(historical_prices)