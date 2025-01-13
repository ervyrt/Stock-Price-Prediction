import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import time
import numpy as np
import os

output_directory='project/datass'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

API_KEY = 'L7ES73WQXQWTDMRQ'

app=TimeSeries(key=API_KEY,output_format='pandas')
ti = TechIndicators(key=API_KEY, output_format='pandas')

symbol='MSFT'
interval='1min'


# Initialize the TechIndicators class with your API key
'''
snapshot, meta_data = app.get_intraday('MSFT',interval='1min',outputsize='full')
snapshot.to_csv(os.path.join(output_directory, 'MSFT_1min_snapshot.csv'))

print(snapshot.head())
print(snapshot.shape)
sma_data, _ = ti.get_sma(symbol=symbol, interval=interval, time_period=20, series_type='close')
sma_data.to_csv(os.path.join(output_directory, 'MSFT_SMA.csv'))

ema_data, _ = ti.get_ema(symbol=symbol, interval=interval, time_period=20, series_type='close')
ema_data.to_csv(os.path.join(output_directory, 'MSFT_EMA.csv'))

wma_data, _ = ti.get_wma(symbol=symbol, interval=interval, time_period=20, series_type='close')
wma_data.to_csv(os.path.join(output_directory, 'MSFT_WMA.csv'))

dema_data, _ = ti.get_dema(symbol=symbol, interval=interval, time_period=20, series_type='close')
dema_data.to_csv(os.path.join(output_directory, 'MSFT_DEMA.csv'))

tema_data, _ = ti.get_tema(symbol=symbol, interval=interval, time_period=20, series_type='close')
tema_data.to_csv(os.path.join(output_directory, 'MSFT_TEMA.csv'))

trima_data, _ = ti.get_trima(symbol=symbol, interval=interval, time_period=30, series_type='close')
trima_data.to_csv(os.path.join(output_directory, 'MSFT_TRIMA.csv'))

kama_data, _ = ti.get_kama(symbol=symbol, interval=interval, time_period=20, series_type='close')
kama_data.to_csv(os.path.join(output_directory, 'MSFT_KAMA.csv'))

mama_data, _ = ti.get_mama(symbol=symbol, interval=interval, series_type='close')
mama_data.to_csv(os.path.join(output_directory, 'MSFT_MAMA.csv'))'''

'''vwap_data, _ = ti.get_vwap(symbol=symbol, interval=interval)
vwap_data.to_csv(os.path.join(output_directory, 'MSFT_VWAP.csv'))

t3_data, _ = ti.get_t3(symbol=symbol, interval=interval, time_period=20, series_type='close')
t3_data.to_csv(os.path.join(output_directory, 'MSFT_T3.csv'))

macd_data, _ = ti.get_macd(symbol=symbol, interval=interval, series_type='close')
macd_data.to_csv(os.path.join(output_directory, 'MSFT_MACD.csv'))

macdext_data, _ = ti.get_macdext(symbol=symbol, interval=interval, series_type='close')
macdext_data.to_csv(os.path.join(output_directory, 'MSFT_MACDEXT.csv'))

stoch_data, _ = ti.get_stoch(symbol=symbol, interval=interval)
stoch_data.to_csv(os.path.join(output_directory, 'MSFT_STOCH.csv'))

stochf_data, _ = ti.get_stochf(symbol=symbol, interval=interval)
stochf_data.to_csv(os.path.join(output_directory, 'MSFT_STOCHF.csv'))

rsi_data, _ = ti.get_rsi(symbol=symbol, interval=interval, time_period=14, series_type='close')
rsi_data.to_csv(os.path.join(output_directory, 'MSFT_RSI.csv'))

stochrsi_data, _ = ti.get_stochrsi(symbol=symbol, interval=interval, time_period=14, series_type='close')
stochrsi_data.to_csv(os.path.join(output_directory, 'MSFT_STOCHRSI.csv'))

willr_data, _ = ti.get_willr(symbol=symbol, interval=interval, time_period=14)
willr_data.to_csv(os.path.join(output_directory, 'MSFT_WILLR.csv'))

adx_data, _ = ti.get_adx(symbol=symbol, interval=interval, time_period=14)
adx_data.to_csv(os.path.join(output_directory, 'MSFT_ADX.csv'))

adxr_data, _ = ti.get_adxr(symbol=symbol, interval=interval, time_period=14)
adxr_data.to_csv(os.path.join(output_directory, 'MSFT_ADXR.csv'))

apo_data, _ = ti.get_apo(symbol=symbol, interval=interval, series_type='close')
apo_data.to_csv(os.path.join(output_directory, 'MSFT_APO.csv'))

ppo_data, _ = ti.get_ppo(symbol=symbol, interval=interval, series_type='close')
ppo_data.to_csv(os.path.join(output_directory, 'MSFT_PPO.csv'))

mom_data, _ = ti.get_mom(symbol=symbol, interval=interval, time_period=10, series_type='close')
mom_data.to_csv(os.path.join(output_directory, 'MSFT_MOM.csv'))

bop_data, _ = ti.get_bop(symbol=symbol, interval=interval)
bop_data.to_csv(os.path.join(output_directory, 'MSFT_BOP.csv'))

cci_data, _ = ti.get_cci(symbol=symbol, interval=interval, time_period=20)
cci_data.to_csv(os.path.join(output_directory, 'MSFT_CCI.csv'))

cmo_data, _ = ti.get_cmo(symbol=symbol, interval=interval, time_period=14, series_type='close')
cmo_data.to_csv(os.path.join(output_directory, 'MSFT_CMO.csv'))

roc_data, _ = ti.get_roc(symbol=symbol, interval=interval, time_period=10, series_type='close')
roc_data.to_csv(os.path.join(output_directory, 'MSFT_ROC.csv'))

rocr_data, _ = ti.get_rocr(symbol=symbol, interval=interval, time_period=10, series_type='close')
rocr_data.to_csv(os.path.join(output_directory, 'MSFT_ROCR.csv'))

aroon_data, _ = ti.get_aroon(symbol=symbol, interval=interval, time_period=25)
aroon_data.to_csv(os.path.join(output_directory, 'MSFT_AROON.csv'))

aroonosc_data, _ = ti.get_aroonosc(symbol=symbol, interval=interval, time_period=25)
aroonosc_data.to_csv(os.path.join(output_directory, 'MSFT_AROONOSC.csv'))

mfi_data, _ = ti.get_mfi(symbol=symbol, interval=interval, time_period=14)
mfi_data.to_csv(os.path.join(output_directory, 'MSFT_MFI.csv'))

trix_data, _ = ti.get_trix(symbol=symbol, interval=interval, time_period=30, series_type='close')
trix_data.to_csv(os.path.join(output_directory, 'MSFT_TRIX.csv'))

ultosc_data, _ = ti.get_ultosc(symbol=symbol, interval=interval)
ultosc_data.to_csv(os.path.join(output_directory, 'MSFT_ULTOSC.csv'))

dx_data, _ = ti.get_dx(symbol=symbol, interval=interval, time_period=14)
dx_data.to_csv(os.path.join(output_directory, 'MSFT_DX.csv'))

minus_di_data, _ = ti.get_minus_di(symbol=symbol, interval=interval, time_period=14)
minus_di_data.to_csv(os.path.join(output_directory, 'MSFT_MINUS_DI.csv'))

plus_di_data, _ = ti.get_plus_di(symbol=symbol, interval=interval, time_period=14)
plus_di_data.to_csv(os.path.join(output_directory, 'MSFT_PLUS_DI.csv'))

minus_dm_data, _ = ti.get_minus_dm(symbol=symbol, interval=interval, time_period=14)
minus_dm_data.to_csv(os.path.join(output_directory, 'MSFT_MINUS_DM.csv'))

plus_dm_data, _ = ti.get_plus_dm(symbol=symbol, interval=interval, time_period=14)
plus_dm_data.to_csv(os.path.join(output_directory, 'MSFT_PLUS_DM.csv'))

bbands_data, _ = ti.get_bbands(symbol=symbol, interval=interval, time_period=20, series_type='close')
bbands_data.to_csv(os.path.join(output_directory, 'MSFT_BBANDS.csv'))

midpoint_data, _ = ti.get_midpoint(symbol=symbol, interval=interval, time_period=14)
midpoint_data.to_csv(os.path.join(output_directory, 'MSFT_MIDPOINT.csv'))

midprice_data, _ = ti.get_midprice(symbol=symbol, interval=interval, time_period=14)
midprice_data.to_csv(os.path.join(output_directory, 'MSFT_MIDPRICE.csv'))

sar_data, _ = ti.get_sar(symbol=symbol, interval=interval)
sar_data.to_csv(os.path.join(output_directory, 'MSFT_SAR.csv'))

trange_data, _ = ti.get_trange(symbol=symbol, interval=interval)
trange_data.to_csv(os.path.join(output_directory, 'MSFT_TRANGE.csv'))

atr_data, _ = ti.get_atr(symbol=symbol, interval=interval, time_period=14)
atr_data.to_csv(os.path.join(output_directory, 'MSFT_ATR.csv'))

natr_data, _ = ti.get_natr(symbol=symbol, interval=interval, time_period=14)
natr_data.to_csv(os.path.join(output_directory, 'MSFT_NATR.csv'))

ad_data, _ = ti.get_ad(symbol=symbol, interval=interval)
ad_data.to_csv(os.path.join(output_directory, 'MSFT_AD.csv'))

adosc_data, _ = ti.get_adosc(symbol=symbol, interval=interval, fastperiod=3, slowperiod=10)
adosc_data.to_csv(os.path.join(output_directory, 'MSFT_ADOSC.csv'))

obv_data, _ = ti.get_obv(symbol=symbol, interval=interval)
obv_data.to_csv(os.path.join(output_directory, 'MSFT_OBV.csv'))

ht_trendline_data, _ = ti.get_ht_trendline(symbol=symbol, interval=interval, series_type='close')
ht_trendline_data.to_csv(os.path.join(output_directory, 'MSFT_HT_TRENDLINE.csv'))

ht_sine_data, _ = ti.get_ht_sine(symbol=symbol, interval=interval, series_type='close')
ht_sine_data.to_csv(os.path.join(output_directory, 'MSFT_HT_SINE.csv'))

ht_trendmode_data, _ = ti.get_ht_trendmode(symbol=symbol, interval=interval, series_type='close')
ht_trendmode_data.to_csv(os.path.join(output_directory, 'MSFT_HT_TRENDMODE.csv'))

ht_dcperiod_data, _ = ti.get_ht_dcperiod(symbol=symbol, interval=interval, series_type='close')
ht_dcperiod_data.to_csv(os.path.join(output_directory, 'MSFT_HT_DCPERIOD.csv'))

ht_dcphase_data, _ = ti.get_ht_dcphase(symbol=symbol, interval=interval, series_type='close')
ht_dcphase_data.to_csv(os.path.join(output_directory, 'MSFT_HT_DCPHASE.csv'))

ht_phasor_data, _ = ti.get_ht_phasor(symbol=symbol, interval=interval, series_type='close')
ht_phasor_data.to_csv(os.path.join(output_directory, 'MSFT_HT_PHASOR.csv'))
time_period2 = 50
time_period3 = 100
sma_data, _ = ti.get_sma(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
sma_data.to_csv(os.path.join(output_directory, 'MSFT_SMA2.csv'))

ema_data, _ = ti.get_ema(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
ema_data.to_csv(os.path.join(output_directory, 'MSFT_EMA2.csv'))

wma_data, _ = ti.get_wma(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
wma_data.to_csv(os.path.join(output_directory, 'MSFT_WMA2.csv'))

dema_data, _ = ti.get_dema(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
dema_data.to_csv(os.path.join(output_directory, 'MSFT_DEMA2.csv'))

tema_data, _ = ti.get_tema(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
tema_data.to_csv(os.path.join(output_directory, 'MSFT_TEMA2.csv'))

trima_data, _ = ti.get_trima(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
trima_data.to_csv(os.path.join(output_directory, 'MSFT_TRIMA2.csv'))

kama_data, _ = ti.get_kama(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
kama_data.to_csv(os.path.join(output_directory, 'MSFT_KAMA2.csv'))

mama_data, _ = ti.get_mama(symbol=symbol, interval=interval, series_type='close')
mama_data.to_csv(os.path.join(output_directory, 'MSFT_MAMA2.csv'))

rsi_data, _ = ti.get_rsi(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
rsi_data.to_csv(os.path.join(output_directory, 'MSFT_RSI2.csv'))

stochrsi_data, _ = ti.get_stochrsi(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
stochrsi_data.to_csv(os.path.join(output_directory, 'MSFT_STOCHRSI2.csv'))

willr_data, _ = ti.get_willr(symbol=symbol, interval=interval, time_period=time_period2)
willr_data.to_csv(os.path.join(output_directory, 'MSFT_WILLR2.csv'))

adx_data, _ = ti.get_adx(symbol=symbol, interval=interval, time_period=time_period2)
adx_data.to_csv(os.path.join(output_directory, 'MSFT_ADX2.csv'))

adxr_data, _ = ti.get_adxr(symbol=symbol, interval=interval, time_period=time_period2)
adxr_data.to_csv(os.path.join(output_directory, 'MSFT_ADXR2.csv'))

apo_data, _ = ti.get_apo(symbol=symbol, interval=interval, series_type='close')
apo_data.to_csv(os.path.join(output_directory, 'MSFT_APO2.csv'))

ppo_data, _ = ti.get_ppo(symbol=symbol, interval=interval, series_type='close')
ppo_data.to_csv(os.path.join(output_directory, 'MSFT_PPO2.csv'))

mom_data, _ = ti.get_mom(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
mom_data.to_csv(os.path.join(output_directory, 'MSFT_MOM2.csv'))

sma_data, _ = ti.get_sma(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
sma_data.to_csv(os.path.join(output_directory, 'MSFT_SMA2.csv'))

ema_data, _ = ti.get_ema(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
ema_data.to_csv(os.path.join(output_directory, 'MSFT_EMA2.csv'))

wma_data, _ = ti.get_wma(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
wma_data.to_csv(os.path.join(output_directory, 'MSFT_WMA2.csv'))

dema_data, _ = ti.get_dema(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
dema_data.to_csv(os.path.join(output_directory, 'MSFT_DEMA2.csv'))

tema_data, _ = ti.get_tema(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
tema_data.to_csv(os.path.join(output_directory, 'MSFT_TEMA2.csv'))

trima_data, _ = ti.get_trima(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
trima_data.to_csv(os.path.join(output_directory, 'MSFT_TRIMA2.csv'))

kama_data, _ = ti.get_kama(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
kama_data.to_csv(os.path.join(output_directory, 'MSFT_KAMA2.csv'))

mama_data, _ = ti.get_mama(symbol=symbol, interval=interval, series_type='close')
mama_data.to_csv(os.path.join(output_directory, 'MSFT_MAMA2.csv'))

rsi_data, _ = ti.get_rsi(symbol=symbol, interval=interval, time_period=time_period2, series_type='close')
rsi_data.to_csv(os.path.join(output_directory, 'MSFT_RSI2.csv'))

sma_data, _ = ti.get_sma(symbol=symbol, interval=interval, time_period=time_period3, series_type='close')
sma_data.to_csv(os.path.join(output_directory, 'MSFT_SMA3.csv'))
ema_data, _ = ti.get_ema(symbol=symbol, interval=interval, time_period=time_period3, series_type='close')
ema_data.to_csv(os.path.join(output_directory, 'MSFT_EMA3.csv'))

wma_data, _ = ti.get_wma(symbol=symbol, interval=interval, time_period=time_period3, series_type='close')
wma_data.to_csv(os.path.join(output_directory, 'MSFT_WMA3.csv'))

dema_data, _ = ti.get_dema(symbol=symbol, interval=interval, time_period=time_period3, series_type='close')
dema_data.to_csv(os.path.join(output_directory, 'MSFT_DEMA3.csv'))

tema_data, _ = ti.get_tema(symbol=symbol, interval=interval, time_period=time_period3, series_type='close')
tema_data.to_csv(os.path.join(output_directory, 'MSFT_TEMA3.csv'))

trima_data, _ = ti.get_trima(symbol=symbol, interval=interval, time_period=time_period3, series_type='close')
trima_data.to_csv(os.path.join(output_directory, 'MSFT_TRIMA3.csv'))

kama_data, _ = ti.get_kama(symbol=symbol, interval=interval, time_period=time_period3, series_type='close')
kama_data.to_csv(os.path.join(output_directory, 'MSFT_KAMA3.csv'))

mama_data, _ = ti.get_mama(symbol=symbol, interval=interval, series_type='close')
mama_data.to_csv(os.path.join(output_directory, 'MSFT_MAMA3.csv'))

rsi_data, _ = ti.get_rsi(symbol=symbol, interval=interval, time_period=time_period3, series_type='close')
rsi_data.to_csv(os.path.join(output_directory, 'MSFT_RSI3.csv'))'''

indicators = [
    'SMA2', 'EMA2', 'WMA2', 'DEMA2', 'TEMA2', 'TRIMA2', 'KAMA2', 'MAMA2', 
    'VWAP', 'T3', 'MACD', 'MACDEXT', 'STOCH', 'STOCHF', 'RSI2', 'STOCHRSI2', 
    'WILLR2', 'ADX2', 'ADXR2', 'APO2', 'PPO2', 'MOM2', 'BOP', 'CCI', 'CMO', 
    'ROC', 'ROCR', 'AROON', 'AROONOSC', 'MFI', 'TRIX', 'ULTOSC', 'DX', 
    'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM', 'BBANDS', 'MIDPOINT', 
    'MIDPRICE', 'SAR', 'TRANGE', 'ATR', 'NATR', 'AD', 'ADOSC', 'OBV', 
    'HT_TRENDLINE', 'HT_SINE', 'HT_TRENDMODE', 'HT_DCPERIOD', 'HT_DCPHASE', 
    'HT_PHASOR', 'SMA3', 'EMA3', 'WMA3', 'DEMA3', 'TEMA3', 'TRIMA3', 'KAMA3', 'MAMA3', 'RSI3'
]

def merge_technical_data(output_directory, indicators):
    # Define the path to the snapshot file
    snapshot_path = os.path.join(output_directory, 'MSFT_1min_snapshot.csv')
    # Load the snapshot data
    snapshot_data = pd.read_csv(snapshot_path, index_col='date')
    snapshot_data.reset_index(inplace=True)
    snapshot_data.rename(columns={'index': 'date'}, inplace=True)
    # Initialize the final dataframe with the snapshot data
    final_data = snapshot_data

    # Iterate over each technical indicator to load and merge its data
    for indicator in indicators:
        file_path = os.path.join(output_directory, f'MSFT_{indicator}.csv')
        if os.path.exists(file_path):
            # Load the indicator data
            indicator_data = pd.read_csv(file_path, index_col='date')
            indicator_data.reset_index(inplace=True)
            indicator_data.rename(columns={'index': 'date'}, inplace=True)
            # Rename columns to include the indicator name
            # Merge the data using the 'date' column instead of indices
            final_data = final_data.merge(indicator_data, on='date', how='inner', suffixes=('', f'_{indicator}'))
        else:
            print(f"Warning: File for {indicator} not found at {file_path}")

    # Save the merged data to a new CSV file
    merged_path = os.path.join(output_directory, 'MSFT_merged_data.csv')
    final_data.to_csv(merged_path)
    print(f"Merged data saved to {merged_path}")

# Call the function to merge the data
merge_technical_data(output_directory, indicators)
