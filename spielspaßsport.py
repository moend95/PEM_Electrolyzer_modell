import numpy as np
import pandas as pd


df = pd.read_csv('wind_energy.csv', sep=',', decimal='.')

P_min = 100  # Minimalleistung für die Wasserstoffproduktion, Beispielwert

# Status und Zeit seit Statuswechsel initialisieren
df['Status'] = 'cold standby'
df['time_since_status_change'] = 0

# Vorigen Status für den Fall des Zurückfallens von "booting" speichern
prev_status_before_booting = ""

for i in range(1, len(df)):
    power = df.loc[i, 'P_ac']
    prev_status = df.loc[i-1, 'Status']
    time_since_change = df.loc[i-1, 'time_since_status_change'] + 1

    # Logik für Statuswechsel
    if power >= P_min:
        if prev_status in ['cold standby', 'hot standby']:
            prev_status_before_booting = prev_status
            df.at[i, 'Status'] = 'booting'
            df.at[i, 'time_since_status_change'] = 1
        elif prev_status == 'booting' and \
            ((prev_status_before_booting == 'cold standby' and time_since_change >= 30) or
             (prev_status_before_booting == 'hot standby' and time_since_change >= 15)):
            df.at[i, 'Status'] = 'h2 production'
            df.at[i, 'time_since_status_change'] = 0
        elif prev_status == 'hot' or prev_status == 'h2 production':
            df.at[i, 'Status'] = 'h2 production'
            df.at[i, 'time_since_status_change'] = 0 if prev_status == 'hot' else time_since_change
        else:
            df.at[i, 'Status'] = prev_status
            df.at[i, 'time_since_status_change'] = time_since_change
    else:
        if prev_status == 'booting':
            # Fall zurück zu cold standby, wenn während des Bootings die Mindestleistung unterschritten wird
            df.at[i, 'Status'] = 'cold standby'
            df.at[i, 'time_since_status_change'] = 0
        elif prev_status == 'h2 production':
            df.at[i, 'Status'] = 'hot'
            df.at[i, 'time_since_status_change'] = 0
        elif prev_status == 'hot' and time_since_change > 4:
            df.at[i, 'Status'] = 'hot standby'
            df.at[i, 'time_since_status_change'] = 0
        elif prev_status == 'hot standby' and time_since_change >= 55:
            df.at[i, 'Status'] = 'cold standby'
            df.at[i, 'time_since_status_change'] = 0
        else:
            # Bleibt im aktuellen Status, wenn keine andere Bedingung erfüllt ist
            df.at[i, 'Status'] = prev_status
            df.at[i, 'time_since_status_change'] = time_since_change if prev_status != 'cold standby' else 0
df = df.drop(['time_since_status_change'], axis=1)


print(df)
df.to_csv('test_output.csv', index=False)