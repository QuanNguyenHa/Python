import pandas as pd
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.now()

df = web.DataReader("XOM", "yahoo", start, end)

#reset to the normal index
df.reset_index(inplace=True)
df.set_index("Date", inplace=True)

print(df.head(5))

df['High'].plot()
plt.legend()
plt.show()
