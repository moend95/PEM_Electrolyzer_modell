import pandas as pd
from electrolyzer import electrolyzer
import matplotlib.pyplot as plt

# Import der Eingangsleistung


ts = pd.read_csv('wind_energy.csv', sep=',', decimal='.')

#ts = ts*1000
#ts = ts.resample('1min').interpolate(method='linear')

ts= ts.loc[0:200]

# Leistungsanpassung
#ts['P_ac'] = round(ts['P_ac'] / 100, 2)

electrolyzer = electrolyzer("1000", "kw", "1", "M", "750", "0",
                                  "kg")  # Elektrolyseur-Größe,Einheit Elektrolyseur,  dt, Einheit zeit, Druck in bar, benötigte Wasserstoffmenge, Einheit Wasserstoffmenge

# Auführen des Elektrolyseurs


electrolyzer.run(ts)

# CSV-Datei
ts.to_csv('Electrolyzer_output.csv', index=False)

print(ts)
# EXCEL-Datei
# excel_file_path = r'C:\Users\Anwender\Documents\Masterprojekt\12345\vpplib\vpplib\a_output.xlsx'
# ts.to_excel(excel_file_path, index=False)

# timestamp_int = 10
# timestamp_str = "2015-01-01 02:30:00+00:00"
#
#
# def test_value_for_timestamp(electrolyzer, timestamp):
#     timestepvalue = electrolyzer.value_for_timestamp(timestamp)
#     print("\nvalue_for_timestamp:\n", timestepvalue)


# def test_observations_for_timestamp(electrolyzer, timestamp):
#     print("observations_for_timestamp:")
#
#     observation = electrolyzer.observations_for_timestamp(timestamp)
#     print(observation, "\n")
#
#
# test_value_for_timestamp(electrolyzer, timestamp_int)
# test_value_for_timestamp(electrolyzer, timestamp_str)
#
# test_observations_for_timestamp(electrolyzer, timestamp_int)
# test_observations_for_timestamp(electrolyzer, timestamp_str)

plt.plot(ts['time'], ts['hydrogen production [Kg/dt]'], linestyle='-', color='b')  # Markierung und Linienstil bestimmen
#
# plt.title('Effizienz in Abhängigkeit von P_in')  # Titel des Diagramms
# plt.xlabel('P_in')  # Bezeichnung der X-Achse
# plt.ylabel('Effizienz [%]')  # Bezeichnung der Y-Achse
# plt.grid(True)  # Raster anzeigen für eine bessere Lesbarkeit
#
plt.show()  # Anzeigen des Diagramms