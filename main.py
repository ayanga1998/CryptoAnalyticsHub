from functions.crypto_analyst import CryptoAnalyst
from functions.crypto_analyst import CustomModel
from PyQt5.QtGui import *
from PyQt5 import QtCore
import PyQt5.QtWidgets as qtw
import pandas as pd
import numpy as np
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mplfinance as mpf
from PyQt5 import QtCore


class App(qtw.QMainWindow):
    def __init__(self):
        super().__init__()

        # Add a title
        self.title = "Crypto Advisor"
        self.left = 0
        self.top = 0
        self.width = 1100
        self.height = 700
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top, self.width, self.height)
        self.tab_widget = MyTabWidget(self)

        self.setCentralWidget(self.tab_widget)

        self.show()


class MyTabWidget(qtw.QWidget):
    def __init__(self, parent):
        super(qtw.QWidget, self).__init__(parent)

        self.ca = CryptoAnalyst()
        self.get_crypto_names = self.ca.get_coin_names()
        self.layout = qtw.QVBoxLayout(self)

        # Initialize the tab screen
        self.tabs = qtw.QTabWidget()
        self.tab1 = qtw.QWidget()
        self.tab2 = qtw.QWidget()
        self.tab3 = qtw.QWidget()
        self.tab4 = qtw.QWidget()
        #self.tabs.resize(1000, 600)

        #add tabs
        self.tabs.addTab(self.tab1, "Home Page")
        self.tabs.addTab(self.tab2, "Price History")
        self.tabs.addTab(self.tab3, "Modeling Tools")
        self.tabs.addTab(self.tab4, "Candlestick Chart")

        ########## Tab 1 ###########
        self.tab1.layout = qtw.QVBoxLayout(self)
        self.title = qtw.QLabel()
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setText("Welcome to Crypto Analytics Hub")
        self.title.setFont(QFont('Helvetica', 24))

        self.headline = qtw.QLabel()
        self.headline.setText('''Mission Statement''')
        self.headline.setFont(QFont('Helvetica', 20))

        self.description = qtw.QLabel()
        self.description.setText('''
        The mission of Crypto Analytics Hub is to introduce crypto enthusiasts and those who are new to the crypto market to 
        the limitless tools and capabilities of Deep Learning/Machine learning models for price analysis. This application in its
        current stage is a proof of concept/demonstration of how users can generate valuable insights on market trends and build
        a general understanding of price movement for thier favorite cryptocurrencies.
        ''')
        self.description.setFont(QFont('Helvetica', 16))

        self.headline = qtw.QLabel()
        self.headline.setText('''Current Features''')
        self.headline.setFont(QFont('Helvetica', 20))

        self.description2 = qtw.QLabel()
        self.description2.setText('''
                Using the dropdown list at the bottom of the homepage window, users have the option to select a currency of their 
                choice. The list of available cryptocurrencies are based on all active cryptos recognized by the CoinGecko API. 
                After selecting a coin, the user has access to several features such as plots of historical price data, candlestick
                charts, and results for price predictions. The current model used is a GRU Neural Network that is trained on the 
                historical data of the selected currency. In the future, our goal is to incorporate a wide variety of price prediction
                methods such as Stochastic Modelling, ARIMA Forecasting, Linear Regression, etc. 
                ''')
        self.description2.setFont(QFont('Helvetica', 16))

        self.l = qtw.QLabel()
        self.l.setText("Please select a cryptocurrency")
        self.tab1.layout.addWidget(self.title)
        self.tab1.layout.addWidget(self.description)
        self.tab1.layout.addWidget(self.headline)
        self.tab1.layout.addWidget(self.description2)
        self.tab1.layout.addWidget(self.l)
        self.my_combo = qtw.QComboBox(self)
        for item in self.get_crypto_names:
            self.my_combo.addItem(item)
        self.tab1.layout.addWidget(self.my_combo)
        self.my_button = qtw.QPushButton('Press Me!', clicked=lambda: press_it())
        self.tab1.layout.addWidget(self.my_button)
        self.tab1.setLayout(self.tab1.layout)

        ########## Tab 2 ###########
        self.figure = plt.figure(1)
        self.figure.patch.set_facecolor("gray")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, parent)
        self.tab2.layout = qtw.QVBoxLayout(self)
        self.tab2.layout.addWidget(self.canvas)
        self.tab2.layout.addWidget(self.toolbar)
        self.tab2.setLayout(self.tab2.layout)

        ########## Tab 3 ###########
        self.figure2 = plt.figure(2)
        self.figure2.patch.set_facecolor("gray")
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, parent)
        self.tab3.layout = qtw.QVBoxLayout(self)
        self.tab3.layout.addWidget(self.canvas2)
        self.tab3.layout.addWidget(self.toolbar2)
        self.tab3.setLayout(self.tab3.layout)

        ########## Tab 4 ###########
        self.figure3 = mpf.figure()
        self.figure3.patch.set_facecolor("gray")
        self.canvas3 = FigureCanvas(self.figure3)
        self.toolbar3 = NavigationToolbar(self.canvas2, parent)
        self.tab4.layout = qtw.QVBoxLayout(self)
        self.tab4.layout.addWidget(self.canvas3)
        self.tab4.layout.addWidget(self.toolbar3)
        self.tab4.setLayout(self.tab4.layout)

        # Add tabs to UI
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        def press_it():
            self.selected_coin = str(self.my_combo.currentText())
            selected_id = self.ca.get_id_from_name(self.selected_coin)
            self.ca = CryptoAnalyst(selected_id)

            self.plot()
            self.plot_model()
            self.plot_candlesticks()

    def plot(self):
        data = self.ca.get_historical_price_data()
        price_change = self.ca.get_historical_price_change()
        name = self.ca.coin.capitalize()

        self.figure.clf()
        self.figure.suptitle('Historical ' + name + ' Market Data', fontsize=16)
        ax = self.figure.add_subplot(221)
        ax.plot(data.index, data.price)
        ax.axes.xaxis.set_visible(False)
        ax.axes.set_ylabel('Price [USD]')
        ax.set_title('Price', fontsize=16)

        ax1 = self.figure.add_subplot(222, sharex = ax)
        ax1.plot(data.index, data.volume)
        ax1.axes.set_ylabel('Volume')
        ax1.axes.xaxis.set_visible(False)
        ax1.set_title('Volume', fontsize=16)

        ax2 = self.figure.add_subplot(223, sharex = ax)
        ax2.plot(data.index, data.market_cap)
        ax2.axes.set_ylabel('Market Cap')
        ax2.tick_params(axis='x', labelrotation=45)
        ax2.set_title('Market Cap ', fontsize=16)

        ax3 = self.figure.add_subplot(224, sharex = ax)
        ax3.plot(price_change[0], price_change[1])
        ax3.axes.set_ylabel('Price Change')
        ax3.tick_params(axis='x', labelrotation=45)
        ax3.set_title('Historical Price Change', fontsize=16)

        self.figure.tight_layout()

        self.canvas.draw()

    def plot_model(self):
        '''
        this function affectively plots data and resets based on home tab button clicks
        - next step is to plot the model predictions
        :return:
        '''
        name = self.ca.coin.capitalize()
        cm = CustomModel(self.ca.coin)
        pred, y_test = cm.run_model()
        range_future = len(pred)

        self.figure2.clf()
        self.figure2.suptitle('GRU NN Predictions for ' + name, fontsize=16)
        ax = self.figure2.add_subplot(111)
        ax.plot(np.arange(range_future), np.array(y_test), label = 'True Future')
        ax.plot(np.arange(range_future), np.array(pred), label = 'Prediction')
        ax.legend()
        ax.axes.set_ylabel('Price [USD]')

        self.figure2.tight_layout()

        self.canvas2.draw()

    def plot_candlesticks(self):
        data = self.ca.get_OHLC()
        name = self.ca.coin.capitalize()
        self.figure3.clf()
        self.figure3.suptitle(name + ' Candlestick Chart (Last 30 Days)', fontsize = 16)
        ax = self.figure3.add_subplot(111)

        mpf.plot(data, type='candle', style='yahoo', ax = ax, mav = (5, 10, 15))

        self.figure3.tight_layout()
        self.canvas3.draw()



if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
