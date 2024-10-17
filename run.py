import matplotlib.pyplot as plt
import pandas as pd

from electrolyzer import electrolyzer

# Import der Eingangsleistung
ts = pd.read_csv('wind_energy.csv', sep=',', decimal='.', parse_dates=[0], index_col=0)

ts= ts.loc["2015-01-01 00:00:00+00:00 ":"2015-01-03 02:00:00+00:00"]

# Leistungsanpassung

electrolyzer = electrolyzer("1000","15", "min", "750")  # Elektrolyseur-Größe,Einheit Elektrolyseur,  dt, Einheit zeit, Druck in bar


# Auführen des Elektrolyseurs
ts = electrolyzer.calculate_data_table(ts)
ts.reset_index(inplace=True)
print(ts)

# CSV-Datei
ts.to_csv('Electrolyzer_output.csv', index=False)

#plt.plot( ts['hydrogen production [(Kg/h)/dt]'], linestyle='-', color='b')  # Markierung und Linienstil bestimmen
#
# plt.title('Effizienz in Abhängigkeit von P_in')  # Titel des Diagramms
# plt.xlabel('P_in')  # Bezeichnung der X-Achse
# plt.ylabel('Effizienz [%]')  # Bezeichnung der Y-Achse
# plt.grid(True)  # Raster anzeigen für eine bessere Lesbarkeit
#
# plt.show()  # Anzeigen des Diagramms