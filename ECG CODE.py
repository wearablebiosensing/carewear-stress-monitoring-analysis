import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_excel ("P2 Data.xlsx")

fig, axs=plt.subplots (2,1)

axs[0].plot(data['BELT'])
axs[0].set_title('BELT')
axs[1].plot(data['Biopac'])
axs[1].set_title('BIOPAC')

plt.show()