# Crypto Analytics Hub

The purpose of this project is to create a dashboard for users to toggle through various charts to analyze price behavior of their favorite cryptocurrency. The interface has four main tabs: the home page, price history, modeling tools, and candlestick charts. In its current state, Crypto Analytics Hub utilizes PyQt for the interface and is connected to CoinGecko API using the PyCoinGecko wrapper to access price data in real time. In the modeling tab, I deployed a GRU Network using Tensorflow to generate predictions for the next tick. This project in its current stage in development is not meant to be used for financial advice. I will continue to add new features and increase the overall quality of the interface. 

In particular, I would love to incorporate various deep learning/machine learning models to generate indicators into the dashboard. I was quite new to front-end development when I first made this code (and admittedly still am) so this repository is more of a sandbox for me to build upon these areas. There are several criticisms I have of the initial design, such as the utility of my modelling tools tab. I have included my initial poster for this project that I presented for my Cognitive Computing course for reference.
