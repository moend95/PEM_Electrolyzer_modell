import math
import numpy as np
import scipy
import pandas as pd
from scipy.signal import tf2ss, cont2discrete
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.optimize import fsolve


class electrolyzer:

    def __init__(self, P_elektrolyseur_, unit_P, dt_1, unit_dt, p2, production_H2_, unit_production_H2):

        # P_elektrolyseur_       :Electrolyzer power not yet converted.
        # unit_P                 :Unit Elektrolyzer
        # dt_1                   :Timestep not yet converted
        # unit_dt                :Unit Timestep
        # p2                     :Compression pressure
        # production_H2_         :Hydrogen mass to be produced (time calculation).
        # unit_production_H2     : Unit Hydrogen mass to be produced (time calculation)

        # Constants
        self.F = 96485.34  # Faraday's constant [C/mol]
        self.R = 8.314  # ideal gas constant [J/(mol*K)]
        self.n = 2  # number of electrons transferred in reaction
        self.gibbs = 237.24e3
        self.E_th_0 = 1.481  # thermoneutral voltage at standard state
        self.M = 2.016  # molecular weight [g/mol]
        self.lhv = 33.33  # lower heating value of H2 [kWh/kg]
        self.hhv = 39.41  # higher heating value of H2 [kWh/kg]
        self.roh_H2 = 0.08988  # Density in kg/m3
        self.roh_O = 1.429  # Density kg/m3
        self.T = 50  # Grad Celsius
        self.p2 = p2  # bar compression

        self.P_elektrolyseur_ = P_elektrolyseur_  # kW
        self.P_elektrolyseur = P_elektrolyseur_  # kW
        self.unit_P = unit_P  # kW
        self.dt_1 = dt_1  # time(s/h/d)
        self.unit_dt = unit_dt  # (s/h/d)
        self.p2 = p2  # bar
        self.production_H2_ = production_H2_  # weight
        self.unit_production_H2 = unit_production_H2  # weight (g,kg,t)

        self.kontrolle()  # Checks the input and provides an error message if necessary

        self.P_nominal = self.P_elektrolyseur  # kW
        self.P_min = self.P_nominal * 0.1  # kW   minimum power Elektrolyzer
        self.P_max = self.P_nominal  # kW   maximum power Elektrolyzer

        # Stack parameters
        self.n_cells = 56  # Number of cells
        self.cell_area = 2500  # [cm^2] Cell active area
        self.max_current_density = 2.1  # [A/cm^2] max. current density #2 * self.cell_area
        self.temperature = 50  # [C] stack temperature
        self.n_stacks = self.P_nominal / self.stack_nominal()  # number of stacks

        self.p_atmo = 101325  # 2000000  # (Pa) atmospheric pressure / pressure of water
        self.p_anode = self.p_atmo  # (Pa) pressure at anode, assumed atmo
        self.dt = 1.0
    def kontrolle(self):
        # the various inputs are checked
        # If incorrect inputs are provided, error messages will be issued
        # ----------------------------------------------------------------
        # Check whether the performance of the electrolyzer and the time step are a number or a decimal. If not, an error message will be displayed
        try:
            self.P_elektrolyseur = float(self.P_elektrolyseur)
        except ValueError:
            raise ValueError("Please check the input for the electrolyzer size. It should be numbers or decimals.")

        try:
            self.dt_1 = float(self.dt_1)
        except ValueError:
            raise ValueError("Please check the input for the time step. It should be numbers or decimals.")

        # ---------------------------------------------------------------------
        # Here, the unit of the electrolyzer is being verified
        # units_P: W, KW, MW, GW  Units of the ELektrolyzer

        if self.unit_P.lower() == "w":  # W
            self.P_elektrolyseur = self.P_elektrolyseur / 1000
            self.unit_P_2 = "W"
        elif self.unit_P.lower() == "kw":  # kW
            self.P_elektrolyseur = self.P_elektrolyseur
            self.unit_P_2 = "KW"
        elif self.unit_P.lower() == "mw":  # mW
            self.P_elektrolyseur = self.P_elektrolyseur * 1000
            self.unit_P_2 = "MW"
        elif self.unit_P.lower() == "gw":  # gW
            self.P_elektrolyseur = self.P_elektrolyseur * 1000 * 1000
            self.unit_P_2 = "GW"
        else:
            raise ValueError("Please check the unit of the electrolyzer! Currently, the options are W, kW, MW, GW.")
            # ----------------------------------------------------------------------
        # Here, the unit of the time step is being verified.
        # Units_dt: S, M, H,D     Units of the time step.

        if self.unit_dt.lower() == "s":  # second
            self.dt = self.dt_1 / 60
            self.dt_2 = "seconds"

        elif self.unit_dt.lower() == "m":  # minute
            self.dt = self.dt_1
            self.dt_2 = "Minutes"

        elif self.unit_dt.lower() == "h":  # hour
            self.dt = self.dt_1 * 60
            self.dt_2 = "hours"

        elif self.unit_dt.lower() == "d":  # days
            self.dt = self.dt_1 * 60 * 24
            self.dt_2 = "Days"

        else:
            raise ValueError(
                "Please verify the unit of time! Currently, the options are S (seconds), M (minutes), H (hours), D (days)")
            # ------------------------------------------------------------------------------
        # Pressure verification.

        if self.p2 == "":
            self.p2 = 0
        elif not self.p2.isdigit():  # If p2 is not a numerical value, for example, a letter.
            self.p2 = 0
        elif int(self.p2) < 30:  # bar
            raise ValueError("The value of the pressure to be compressed must be greater than 30 bar!")
        elif int(self.p2) >= 30:  # bar
            self.p2 = self.p2  # bar
        else:
            raise ValueError("Please check the unit of the pressure to be compressed.")

        # ---------------------------------------------------------------------------------
        # Verification of the time calculation for hydrogen generation
        # units_H2 G, KG, T,

        self.production_H2 = self.production_H2_

        # If no input is provided for the required hydrogen amount, it will be set to 0 and not displayed. This is part of the function within the 'h2_production_calc' function.

        if self.production_H2_ == "":
            self.production_H2 = 0
            self.production_H2_ = 0
        else:
            try:
                self.production_H2_ = float(self.production_H2_)
                self.production_H2 = int(self.production_H2_)
            except ValueError:
                self.production_H2 = 0
                self.production_H2_ = 0

        if self.unit_production_H2.lower() == "g":  # gramm
            self.unit_production = self.production_H2
            self.unit_H2 = "Gramm"
            self.production_H2 = self.production_H2 / 1000
        elif self.unit_production_H2.lower() == "kg":  # kilogramm
            self.production_H2 = self.production_H2
            self.unit_H2 = "Kilogramm"
        elif self.unit_production_H2.lower() == "t":  # Tonnen
            self.production_H2 = self.production_H2 * 1000
            self.unit_H2 = "Tonnen"
        elif self.unit_production_H2 == "":
            self.production_H2 = 0
            self.production_H2_ = 0
            self.unit_H2 = ""
        else:
            raise ValueError(
                "Please check the unit of the generated hydrogen! Currently, the options are g (grams), kg (kilograms), T (tonnes).")
        # ----------------------------------------------------------------------------------------------------------------------------------
        # If the power of the electrolyzer is not divisible by 500, an error message will be issued. Reason: Stack size is fixed at 500 kW.
        if self.P_elektrolyseur % 500 != 0:
            raise ValueError(
                "The power of the electrolyzer must be in the 500 series, as one stack of the electrolyzer is 500 kW")
            # ------------------------------------------------------------------

    def state_codes(self, df):

        # state und Zeit seit statewechsel initialisieren
        df['state'] = 'cold standby'
        df['time_since_state_change'] = 0

        # Vorigen state für den Fall des Zurückfallens von "booting" speichern
        prev_state_before_booting = ""

        for i in range(1, len(df)):
            power = df.loc[i, 'P_ac']
            prev_state = df.loc[i - 1, 'state']
            time_since_change = df.loc[i - 1, 'time_since_state_change'] + 1

            # Logik für statewechsel
            if power >= self.P_min:
                if prev_state in ['cold standby', 'hot standby']:
                    prev_state_before_booting = prev_state
                    df.at[i, 'state'] = 'booting'
                    df.at[i, 'time_since_state_change'] = 1
                elif prev_state == 'booting' and \
                        ((prev_state_before_booting == 'cold standby' and time_since_change >= 30) or
                         (prev_state_before_booting == 'hot standby' and time_since_change >= 15)):
                    df.at[i, 'state'] = 'h2 production'
                    df.at[i, 'time_since_state_change'] = 0
                elif prev_state == 'hot' or prev_state == 'h2 production':
                    df.at[i, 'state'] = 'h2 production'
                    df.at[i, 'time_since_state_change'] = 0 if prev_state == 'hot' else time_since_change
                else:
                    df.at[i, 'state'] = prev_state
                    df.at[i, 'time_since_state_change'] = time_since_change
            else:
                if prev_state == 'booting':
                    # Fall zurück zu cold standby, wenn während des Bootings die Mindestleistung unterschritten wird
                    df.at[i, 'state'] = 'cold standby'
                    df.at[i, 'time_since_state_change'] = 0
                elif prev_state == 'h2 production':
                    df.at[i, 'state'] = 'hot'
                    df.at[i, 'time_since_state_change'] = 0
                elif prev_state == 'hot' and time_since_change > 4:
                    df.at[i, 'state'] = 'hot standby'
                    df.at[i, 'time_since_state_change'] = 0
                elif prev_state == 'hot standby' and time_since_change >= 55:
                    df.at[i, 'state'] = 'cold standby'
                    df.at[i, 'time_since_state_change'] = 0
                else:
                    # Bleibt im aktuellen state, wenn keine andere Bedingung erfüllt ist
                    df.at[i, 'state'] = prev_state
                    df.at[i, 'time_since_state_change'] = time_since_change if prev_state != 'cold standby' else 0
        df.drop(['time_since_state_change'], axis=1, inplace=True)

        return df

        # long_gap_threshold = 60
        # short_gap_threshold = 5
        # # create a mask for power values below P_min
        # below_threshold_mask = df['P_ac'] < self.P_min
        #
        # # find short gaps (up to 4 steps) where power is below P_min
        # short_gaps = below_threshold_mask.rolling(window=short_gap_threshold).sum()
        # hot_mask = (short_gaps <= 4) & below_threshold_mask
        # df.loc[hot_mask, 'status'] = 'hot'
        #
        # # find middle gaps (between 5 and 60 steps) where power is below P_min
        # middle_gaps = below_threshold_mask.rolling(window=long_gap_threshold).sum()
        # hot_standby_mask = ((5 <= middle_gaps) & (middle_gaps < 60)) & below_threshold_mask
        # df.loc[hot_standby_mask, 'status'] = 'hot standby'
        # # find long gaps (over 60 steps) where power is below P_min
        # long_gaps = below_threshold_mask.rolling(window=long_gap_threshold).sum()
        # cold_standby_mask = (long_gaps >= 60) & below_threshold_mask
        # df.loc[cold_standby_mask, 'status'] = 'cold standby'
        #
        # # mark production periods (above P_min)
        # production_mask = df['P_ac'] >= self.P_min
        # df.loc[production_mask, 'status'] = 'production'
        #
        # # add status codes
        # df['status codes'] = df['status'].replace({
        #     'cold standby': 0,
        #     'hot standby': 1,
        #     'hot': 2,
        #     'production': 4
        # })
        #
        # # add 'booting' status
        # booting_mask = pd.Series(False, index=df.index)
        # # Identify rows where production is True and previous row is hot standby or cold standby
        # booting_mask |= (df['status'].eq('production') & df['status'].shift(1).isin(['hot standby', 'cold standby']))
        #
        # # Identify rows where production is True and status is cold standby for up to 5 rows before
        #
        # booting_mask |= (df['status'].eq('production') & df['status'].shift(30).eq('cold standby'))
        #
        # # Identify rows where production is True and status is hot standby for up to 30 rows before
        # for i in range(1, 15):
        #     booting_mask |= (df['status'].eq('production') & df['status'].shift(i).eq('hot standby'))
        #
        # df.loc[booting_mask, 'status'] = 'booting'
        #
        # # add status codes
        # df['status codes'] = df['status'].replace({
        #     'cold standby': 0,
        #     'hot standby': 1,
        #     'hot': 2,
        #     'production': 4,
        #     'booting': 3
        # })



    def calc_cell_voltage(self, I, T):
        """
        I [Adc]: stack current
        T [degC]: stack temperature
        return : V_cell [Vdc/cell]: cell voltage
        """
        T_K = T + 273.15

        # Cell reversible voltage:
        E_rev_0 = self.gibbs / (self.n * self.F)  # Reversible cell voltage at standard state

        p_anode = 200000
        p_cathode = 3000000  # pressure applied by the pumps

        # Arden Buck equation T=C, https://www.omnicalculator.com/chemistry/vapour-pressure-of-water#vapor-pressure-formulas
        p_h2O_sat = (0.61121 * np.exp((18.678 - (T / 234.5)) * (T / (257.14 + T)))) * 1e3  # (Pa)
        p_atmo = 101325
        # General Nernst equation
        E_rev = E_rev_0 + ((self.R * T_K) / (self.n * self.F)) * (
            np.log(((p_anode - p_h2O_sat) / p_atmo) * np.sqrt((p_cathode - p_h2O_sat) / p_atmo)))

        # E_rev = E_rev_0 - ((E_rev_0* 10**-3 * T_K) + 9.523 * 10**-5 * np.log(T_K) + 9.84*10**-8* T_K**2) #empirical equation

        T_anode = T_K
        T_cathode = T_K


        alpha_a = 2 # anode charge transfer coefficient
        alpha_c = 0.5 # cathode charge transfer coefficient

        # anode exchange current density
        i_0_a = 2 * 10 ** (-7)

        # cathode exchange current density
        i_0_c = 10 ** (-3)

        i = I / self.cell_area

        z_a = 4  # stoichiometric coefficient of electrons transferred at anode
        z_c = 2  # stoichometric coefficient of electrons transferred at cathode
        i_0_a = 10 ** (-9)  # anode exchange current density
        i_0_c = 10 ** (-3)  # cathode exchange current density

        V_act_a = ((self.R * T_anode) / (alpha_a * z_a * self.F)) * np.log(i / i_0_a)
        V_act_c = ((self.R * T_cathode) / (alpha_c * z_c * self.F)) * np.log(i / i_0_c)

        # pulled from https://www.sciencedirect.com/science/article/pii/S0360319917309278?via%3Dihub
        lambda_nafion = 25
        t_nafion = 0.01  # cm

        sigma_nafion = ((0.005139 * lambda_nafion) - 0.00326) * np.exp(
            1268 * ((1 / 303) - (1 / T_K)))
        R_ohmic_ionic = t_nafion / sigma_nafion

        R_ohmic_elec = 50e-3

        V_ohmic = i * (R_ohmic_elec + R_ohmic_ionic)

        V_cell = E_rev + V_act_a + V_act_c + V_ohmic

        return V_cell

    def create_polarization(self):
        '''
        :return: dataframe with power, voltage and current values for the polarization curve
        '''
        currents = np.arange(1, 5500, 10)
        voltage = []
        for i in range(len(currents)):
            voltage.append(self.calc_cell_voltage(currents[i], self.T))
        df = pd.DataFrame({"current_A": currents, "voltage_U": voltage})
        df['power_W'] = df["current_A"] * df["voltage_U"]
        # df['current_A'] = df['current_A']/self.cell_area

        return df

    def plot_polarization(self):
        '''
        :return: plot of the polarization curve
        '''
        df = self.create_polarization()

        plt.plot((df['current_A'] / self.cell_area), df['voltage_U'])

        plt.title('Polarization curve')
        plt.xlabel('Current densitiy [A/cm2]')
        plt.ylabel('Cell Voltage [V]')
        plt.grid(True)

        plt.show()

    def stack_nominal(self):
        # Stack size is fixed at 500 kW.
        '''
        :return: stack nominal in kW
        '''
        P_stack_nominal = round((self.create_polarization().iloc[504, 0] * self.create_polarization().iloc[
            504, 1] * self.n_cells) / 1000)  # in kW

        return P_stack_nominal  # KW

    def calculate_cell_current(self, P_dc):
        '''
        P_dc: Power DC in Watt, after power electronics
        P_cell: Power each cell
        return I: Current each cell in Ampere
        '''
        P_cell = ((P_dc / self.n_stacks) / self.n_cells) * 1000  # in W
        df = self.create_polarization()
        x = df['power_W'].to_numpy()
        y = df['current_A'].to_numpy()

        f = interp1d(x, y, kind='linear')

        return f(P_cell)

    def calc_faradaic_efficiency(self, I):
        """
        I [A]: stack current
        return :: eta_F [-]: Faraday's efficiency
        """
        p = 20  # electrolyze pressure in bar
        i = I / self.cell_area

        a_1 = -0.0034
        a_2 = -0.001711
        b = -1
        c = 1

        eta_f = (a_1 * p + a_2) * ((i) ** b) + c    #Reference: https://res.mdpi.com/d_attachment/energies/energies-13-04792/article_deploy/energies-13-04792-v2.pdf

        return eta_f

    def power_electronics(self, P_ac, P_nominal):
        '''
        P_nominal: Electrolyzer Size in kW
        P_ac: P_ac
        P_electronics: Self-consumption power electronics in kW
        '''
        # definition of the
        relative_performance = [0.0, 0.09, 0.12, 0.15, 0.189, 0.209, 0.24, 0.3, 0.4, 0.54, 0.7, 1.001]
        eta = [0.86, 0.91, 0.928, 0.943, 0.949, 0.95, 0.954, 0.96, 0.965, 0.97, 0.973, 0.977]
        # Interpolationsfunktion erstellen
        f_eta = interp1d(relative_performance, eta)

        # Eigenverbrauch berechnen
        # print(P_ac)
        # print(P_nominal)
        if P_ac < P_nominal:  # Funktion hinzugefügt da probleme wenn die eingangsleistung gegen null geht
            P_ac = P_nominal
        eta_interp = f_eta(P_nominal / P_ac)  # Interpoliere den eta-Wert
        # print(eta_interp)
        P_electronics = P_nominal * (1 - eta_interp)  # Berechne den Eigenverbrauch

        return P_electronics

    def power_dc(self, P_ac):
        '''
        :param P_ac:
        :return:
        '''
        # print(self.stack_nominal()/100)
        P_dc = P_ac - self.power_electronics(P_ac, self.stack_nominal() / 100)

        # P_dc = P_ac - self.power_electronics(P_ac, self.P_nominal/100)

        return P_dc

    def calc_H2_mfr(self, P_dc):
        """
        P_in [KW] [kWdc]: stack power input
        return :: H2_mfr [kg/dt]: hydrogen mass flow rate
        """

        I = self.calculate_cell_current(P_dc)  # A
        V = self.calc_cell_voltage(I, self.temperature)  # V
        eta_F = self.calc_faradaic_efficiency(I)
        mfr = (eta_F * I * self.M * self.n_cells * self.n_stacks) / (self.n * self.F)  # self.n_stacks hinzugefügt

        H2_mfr = (mfr * self.dt*3600) / 1000  # kg/dt

        return H2_mfr

    def calc_O_mfr(self, H2_mfr):
        '''
        H2_mfr = massen flow rate H2 in kg/dt
        return: Oxygen flow rate in kg/dt
        '''
        roh_O = 1.429  # density Oxigen kg/m3
        O_mfr_m3 = (H2_mfr / self.roh_H2) / 2
        O_mfr = O_mfr_m3 * roh_O
        return O_mfr  # kg/dt

    def calc_H2O_mfr(self, H2_mfr):
        '''
        H2_mfr: Hydrogen mass flow in kg
        O_mfr: Oxygen mass flow in kg
        return: needed water mass flow in kg
        '''
        M_H2O = 18.010  # mol/g

        ratio_M = M_H2O / self.M  # (mol/g)/(mol/g)
        H2O_mfr = H2_mfr * ratio_M  # +40*(60/self.dt)

        return H2O_mfr  # kg/dt

    def gas_drying(self, H2_mfr):
        '''
        input n_h2: mass flow in kg/dt
        :param n_H2:
        :return:
        '''
        M_H2 = 2.016 * 10 ** -3  # kg/mol Molare Masse H2
        nH2 = (H2_mfr / 3600*(self.dt)) / M_H2  # kg/dt in kg/s in mol/s
        cp_H2 = 14300  # J/kg*K Wärmekapazität H2

        X_in = 0.1  # Mol H2O/Mol H2
        X_out = 1  # minimum needed
        n = (X_in / (X_out - X_in)) * nH2
        dT = 300 - 20  # Temperaturdifferenz zwischen Adsorbtion und Desorption

        P_hz = cp_H2 * M_H2 * n * dT

        Q_des = 48600 * n  # J/s
        # P_gasdrying = (P_hz + Q_des)/1000/H2_mfr #in kW/kg
        P_gasdrying = (P_hz + Q_des) / 1000  # in kW
        return P_gasdrying  # kw

    def compression(self, H2_mfr):
        '''
        :param p2: needed pressure in bar
        :param T: electrolyze temperature
        :return: needed Power for compression in kW
        '''
        # bei 1min zeitschritt
        # 500kw
        # 100bar ca. 5kw
        # 200bar ca. 8.5kw
        # 750bar ca. 17kw

        # w_isotherm = R * T * Z * ln(p2 / p1)

        T2 = 273.15 + 30  # k
        p1 = 30  # bar
        Z = 0.95
        k = 1.4
        kk = k / (k - 1)
        kkk = (k - 1) / k
        # eta_Ver = 0.75 #woher kommt der
        eta_Ver = 1

        # If no pressure is specified, compression will not take place.
        if self.p2 == 0:
            w_isentrop = 0
        else:
            w_isentrop = kk * (self.R / self.M) * T2 * Z * (((int(self.p2) / p1) ** (kkk)) - 1)  # j/g

        P_compression = (((w_isentrop) / eta_Ver) * H2_mfr / (self.dt))  # kw
        # print(P_compression)
        return P_compression  # kw

    def heat_cell(self, P_dc):
        '''
        P_dc: in kW
        return: q cell in kW
        '''
        V_th = self.E_th_0
        I = self.calculate_cell_current(P_dc)
        U_cell = self.calc_cell_voltage(I, self.temperature)
        q_cell = (self.n_stacks * (self.n_cells * (U_cell - V_th) * I)) / 1000

        return q_cell  # kW

    def heat_sys(self, q_cell, H2O_mfr):
        '''
        q_cell: in kW
        H20_mfr: in kg/dt
        return: q_loss in kW
                q_H20_fresh in kW
        '''
        # c_pH2O = 0.001162 #kWh/kg*k #alt
        c_pH2O = (0.001162) / (self.dt)  # kWh/kg*k
        dt = self.T - 20  # operate temp. - ambient temp.

        #q_H2O_fresh = - c_pH2O * H2O_mfr * dt * 1.5  # multyplied with 1.5 for transport water #kw
        #q_loss = -(q_cell + q_H2O_fresh) * 0.14  # hier war mal ein minus vor
        q_loss=q_cell
        return q_loss  # kw

    def calc_mfr_cool(self, q_loss):
        '''
        q_system in kW
        return: mfr cooling water in kg/dt
        '''

        q_system = q_loss / 1000  # kWh
        c_pH2O = 0.001162 / (self.dt)  # kWh/kg*k
        # operate temperature - should temperature
        mfr_cool = ((q_system) / (c_pH2O * (50 - 20)))

        return mfr_cool

    def calc_pump(self, H2O_mfr, P_in):
        '''
        mfr_H2o: in kg/h
        P_stack: kw
        P_dc:kw
        pressure: in Pa
        return: kW
        '''
        # Wirkungsgradkurve Kreiselpumpe: https://doi.org/10.1007/978-3-642-40032-2
        relative_performance_pump = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.011]
        eta_pump = [0.627,0.644,0.661,0.677,0.691,0.704,0.715,0.738,0.754,0.769,0.782,0.792,0.797,0.80]
        # Interpolationsfunktion erstellen
        f_eta_pump = interp1d(relative_performance_pump, eta_pump, kind='linear')
        # Wirkungsgrad berechnen für aktuellen
        eta_interp_pump = f_eta_pump(P_in/(self.P_nominal))  # Interpoliere den eta-Wert

        #Druckverlust Leitungen in Pa
        relative_performance_pressure = [0.0, 0.02, 0.07, 0.12, 0.16, 0.2, 0.25, 0.32, 0.36, 0.4, 0.47, 0.54, 0.59,
                                         0.63, 0.67, 0.71, 0.74, 0.77, 0.8, 0.83, 0.86, 0.89, 0.92, 0.95, 0.98, 1.015]
        dt_pressure = [0.0, 330, 1870, 3360, 5210, 8540, 12980, 21850, 27020, 32930, 44000, 59500, 70190, 80520, 90850,
                        100810,110400, 119990, 128840, 138420, 148010, 158330, 169760, 181190, 191890, 200000]
        # Interpolationsfunktion erstellen
        f_dt_pressure = interp1d(relative_performance_pressure, dt_pressure)
        # Eigenverbrauch berechnen
        dt_interp_pressure = f_dt_pressure(P_in/(self.P_nominal))  # Interpoliere den eta-Wert

        vfr_H2O = (H2O_mfr/997) #mass in volume with 997 kg/m3
        P_pump_fresh =  (vfr_H2O/3600) * (2000000) * (1-eta_interp_pump)
        P_pump_cool = (vfr_H2O / 3600) * (dt_interp_pressure) * (1 - eta_interp_pump)
        P_gesamt = (P_pump_fresh + P_pump_cool)  # kw
        return P_gesamt  # kw


    def run(self, df):


        df = self.state_codes(df)

        for i in range(len(df.index)):

            # Checks if the number 4 is present in the 'status' column (production).
            if df.loc[df.index[i], 'state'] == 'h2 production':

                # If the input power is less than the electrolyzer power.
                if df.loc[df.index[i], 'P_ac'] <= self.P_nominal:

                    # hydrogen [kg/dt]
                    df.loc[df.index[i], 'hydrogen production [Kg/dt]'] = self.calc_H2_mfr(df.loc[df.index[i], 'P_ac'])
                else:
                    # hydrogen [kg/dt]
                    df.loc[df.index[i], 'hydrogen production [Kg/dt]'] = self.calc_H2_mfr(self.P_nominal)

                # H20  [kg/dt] # Input
                df.loc[df.index[i], 'H20 [kg/dt]'] = self.calc_H2O_mfr(df.loc[df.index[i], 'hydrogen production [Kg/dt]'])

                # oxygen [kg/dt]
                df.loc[df.index[i], 'Oxygen [kg/dt]'] = self.calc_O_mfr(df.loc[df.index[i], 'hydrogen production [Kg/dt]'])

                # heat system [%]
                df.loc[df.index[i], 'heat system [kW]'] = self.heat_sys(self.heat_cell(df.loc[df.index[i], 'P_ac']),
                                                                        df.loc[df.index[i], 'H20 [kg/dt]'])
                df.loc[df.index[i], 'heat system [%]'] = (df.loc[df.index[i], 'heat system [kW]']/
                                                          df.loc[df.index[i], 'P_ac']) * 100

                # cooling_water [kg/dt]
                cooling_water = self.calc_mfr_cool(df.loc[df.index[i], 'heat system [kW]'])
                df.loc[df.index[i], 'cooling Water [kg/dt]'] = round(cooling_water, 2)

                # power electronics
                df.loc[df.index[i], 'electronics [kW]'] = self.power_electronics(self.P_nominal,
                                                                                     df.loc[df.index[i], 'P_ac'])
                df.loc[df.index[i], 'electronics [%]'] = (df.loc[df.index[i], 'electronics [kW]'] /
                                                              df.loc[df.index[i], 'P_ac']) * 100

                # gasdrying
                df.loc[df.index[i], 'gasdrying [kW]'] = self.gas_drying(
                        df.loc[df.index[i], 'hydrogen production [Kg/dt]'])
                df.loc[df.index[i], 'gasdrying [%]'] = (df.loc[df.index[i], 'gasdrying [kW]'] /
                                                            df.loc[df.index[i], 'P_ac']) * 100

                # pump
                df.loc[df.index[i], 'pump [kW]'] = self.calc_pump(df.loc[df.index[i], 'H20 [kg/dt]'],
                                                                      df.loc[df.index[i], 'P_ac'])
                df.loc[df.index[i], 'pump [%]'] = (df.loc[df.index[i], 'pump [kW]'] / df.loc[
                        df.index[i], 'P_ac']) * 100

                # efficency [%]
                df.loc[df.index[i], 'efficency [%]'] = ((df.loc[df.index[
                        i], 'hydrogen production [Kg/dt]'] * self.lhv) /
                                                        (df.loc[df.index[i], 'P_ac'] +
                                                         df.loc[df.index[i], 'pump [kW]'] +
                                                         df.loc[df.index[i], 'gasdrying [%]'] +
                                                         df.loc[df.index[i], 'electronics [kW]']))

        #             # compression [%]
        #             compression_KW = self.compression(df.loc[df.index[i], 'hydrogen production [Kg/dt]'])
        #             df.loc[df.index[i], 'compression [%]'] = round(
        #                 (compression_KW / df.loc[df.index[i], 'P_ac']) * 100, 2)
        #
        #             # efficency with compression [%]
        #             efficency_with_compression = (((df.loc[df.index[i], 'hydrogen production [Kg/dt]'] * self.lhv * (
        #                         60 / self.dt)) / df.loc[df.index[i], 'P_ac']) * 100) - df.loc[
        #                                              df.index[i], 'gasdrying {%]'] - df.loc[df.index[i], 'pump [%]'] - \
        #                                          df.loc[df.index[i], 'compression [%]'] - df.loc[
        #                                              df.index[i], 'electronics [%]']
        #             df.loc[df.index[i], 'efficency _c [%]'] = round(efficency_with_compression, 2)
        #             # ---------------------------------------------------------------------------------------------------------------------------------------------
        #
        #         # If the input power is greater than the electrolyzer power
        #         else:
                    # ----------------------------------------------------------------------------------------------------------------
                    # output
        #
                    # hydrogen [kg/dt]
        #             hydrogen_production = self.run(self.P_nominal)
        #             df.loc[df.index[i], 'hydrogen production [Kg/dt]'] = round(hydrogen_production, 2)
        #
        #             # surplus electricity [kW]
        #             surplus_electricity = df.loc[df.index[i], 'P_ac'] - self.P_nominal
        #             df.loc[df.index[i], 'surplus electricity [kW]'] = round(surplus_electricity, 2)
        #
        #             # H20  [kg/dt]  #Input
        #             H2O = self.calc_H2O_mfr(df.loc[df.index[i], 'hydrogen production [Kg/dt]'])
        #             df.loc[df.index[i], 'H20 [kg/dt]'] = round(H2O, 2)
        #
        #             # oxygen [kg/dt]
        #             Oxygen = self.calc_O_mfr(df.loc[df.index[i], 'hydrogen production [Kg/dt]'])
        #             df.loc[df.index[i], 'Oxygen [kg/dt]'] = round(Oxygen, 2)
        #
        #             # ------------------------------------------------------------------------------------------------------------------------------
        #             # losses
        #
        #             # electrolyzer [%]
        #             electrolyzer = 100 - (((df.loc[df.index[i], 'hydrogen production [Kg/dt]'] * self.lhv * (
        #                         60 / self.dt)) / self.P_nominal) * 100)
        #             df.loc[df.index[i], 'electrolyzer {%]'] = round(electrolyzer)
        #
        #             # gasdrying [%]
        #             gasdrying_KW = self.gas_drying(df.loc[df.index[i], 'hydrogen production [Kg/dt]'])
        #             df.loc[df.index[i], 'gasdrying {%]'] = round((gasdrying_KW / self.P_nominal) * 100, 2)
        #
        #             # pump  [%]
        #             pump_KW = self.calc_pump(df.loc[df.index[i], 'H20 [kg/dt]'], self.P_nominal)
        #             df.loc[df.index[i], 'pump [%]'] = round((pump_KW / self.P_nominal) * 100, 2)
        #
        #             # electronics [%]
        #             electronics = self.power_electronics(self.P_nominal, self.P_nominal)
        #             df.loc[df.index[i], 'electronics [%]'] = round((electronics / self.P_nominal) * 100, 2)
        #
        #             # heat_cell [%]
        #             Heat_Cell_kW = self.heat_cell(self.P_nominal)
        #             df.loc[df.index[i], 'Heat Cell [%]'] = round((Heat_Cell_kW / self.P_nominal) * 100, 2)
        #
        #             # heat system [%]
        #             heat_system_KW = self.heat_sys(Heat_Cell_kW, df.loc[df.index[i], 'H20 [kg/dt]'])
        #             df.loc[df.index[i], 'heat system [%]'] = round((heat_system_KW / self.P_nominal) * 100, 2)
        #
        #             # compression [%]
        #             compression_KW = self.compression(df.loc[df.index[i], 'hydrogen production [Kg/dt]'])
        #             df.loc[df.index[i], 'compression [%]'] = round((compression_KW / self.P_nominal) * 100, 2)
        #
        #             # cooling_water [kg/dt]
        #             cooling_water = self.calc_mfr_cool(heat_system_KW)
        #             df.loc[df.index[i], 'cooling Water [kg/dt]'] = round(cooling_water, 2)
        #             # ---------------------------------------------------------------------------------------------------------------------------------------
        #             # efficiency [%]
        #             efficiency = (((df.loc[df.index[i], 'hydrogen production [Kg/dt]'] * self.lhv * (
        #                         60 / self.dt)) / self.P_nominal) * 100) - df.loc[df.index[i], 'gasdrying {%]'] - df.loc[
        #                              df.index[i], 'pump [%]'] - df.loc[df.index[i], 'electronics [%]']
        #             df.loc[df.index[i], 'efficiency [%]'] = round(efficiency, 2)
        #
        #             # efficency with compression [%]
        #             efficency_with_compression = (((df.loc[df.index[i], 'hydrogen production [Kg/dt]'] * self.lhv * (
        #                         60 / self.dt)) / self.P_nominal) * 100) - df.loc[df.index[i], 'gasdrying {%]'] - df.loc[
        #                                              df.index[i], 'pump [%]'] - df.loc[df.index[i], 'compression [%]'] - \
        #                                          df.loc[df.index[i], 'electronics [%]']
        #             df.loc[df.index[i], 'efficency _c [%]'] = round(efficency_with_compression, 2)
        #         # ---------------------------------------------------------------------------------------------------------------------------------------------
        #         # required_power [kW/dt]
        #
        #         if df.loc[df.index[i], 'P_ac'] <= self.P_nominal:
        #             df.loc[df.index[i], 'Electrolyzer'] = round(df.loc[df.index[i], 'P_in without losses [KW]'], 2)
        #
        #         else:
        #             df.loc[df.index[i], 'Electrolyzer'] = round(self.P_nominal + (
        #                         df.loc[df.index[i], 'P_in without losses [KW]'] - df.loc[df.index[i], 'P_ac']), 2)
        #
        #         # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #         # Deducdf pump, gas drying, and compression from the next time step.
        #         if i < (len(df.index) - 1):
        #
        #             df.loc[df.index[i + 1], 'P_ac'] = round(
        #                 df.loc[df.index[i + 1], 'P_ac'] - (pump_KW + compression_KW + gasdrying_KW), 2)
        #
        #             if df.loc[df.index[i + 1], 'P_ac'] <= 0:
        #                 df.loc[df.index[i + 1], 'P_ac'] = 0
        #     # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        #     # Startup
        #     elif df.loc[df.index[i], 'state'] == 'booting':
        #         df.loc[df.index[i], 'surplus electricity [kW]'] = df.loc[df.index[
        #             i], 'P_ac'] - 0.0085 * self.P_nominal
        #         # df.loc[df.index[i], 'Electrolyzer' ]=round((df.loc[df.index[i], 'P_in without losses [KW]']-df.loc[df.index[i], 'P_ac']),2)
        #
        #
        #
        #
        #     else:
        #         df.loc[df.index[i], 'surplus electricity [kW]'] = df.loc[df.index[i], 'P_ac']

        # -----------------------------------------------------------------------------------------------------------------------
        # Hydrogen production/volume calculation of compressed hydrogen.
        # -------------------------------------------------------------------------------------------------------------------------
        # Sedf efficiency_c to 0 if p2 equals 0.
        df.fillna(0.0, inplace=True)
        return df