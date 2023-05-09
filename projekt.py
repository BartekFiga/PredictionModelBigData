#1 PRZYGOTOWAC DANE (2013-2020)
#2 WYSWIETLIC DANE
#3 PRZGOT0WAC MODELE PREDYKCJI
#ARIMA ITP Z LISTY
#
#4 WYBIERAMY NAJLEPSZĄ METODĘ
#5 POKAZUJEMY WYNIKI

#oddanie projektu 14.06

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Import data - testowe
val_m_test=pd.read_csv("mezczyzni1990-2019.csv",parse_dates=True,index_col=0)
val_k_test=pd.read_csv("kobiety1990-2019.csv",parse_dates=True,index_col=0)

#Import data - treningowe
val_m=pd.read_csv("mezczyzni_treningowe.csv",parse_dates=True,index_col=0)
val_k=pd.read_csv("kobiety_treningowe.csv",parse_dates=True,index_col=0)

lata = []
rok = 1990
for i in range(30):
    lata.append(rok)
    rok +=1

to_plot_test_m = []
to_plot_test_k = []

for [value] in val_m_test.values.tolist():
    to_plot_test_m.append(value)
for [value] in val_k_test.values.tolist():
    to_plot_test_k.append(value)




#Transform date index from string type to date type
val_m.index = pd.DatetimeIndex(val_m.index).to_period('Y')


mod_m = sm.tsa.SARIMAX(val_m, order=(1, 0, 0))
res_m = mod_m.fit()
#print(res_m.summary())
#print(res_m.forecast(3))

to_plot_m = []
for i in range(27):
    to_plot_m.append(None)
for value in res_m.forecast(3):
    to_plot_m.append(value)

#print(to_plot_m)


mod_m1 = sm.tsa.ARIMA(val_m, order=(1, 0, 0))
res_m1 = mod_m1.fit()
#print(res_m1.summary())
#print(res_m1.forecast(3))

to_plot_m1 = []
for i in range(27):
    to_plot_m1.append(None)
for value in res_m1.forecast(3):
    to_plot_m1.append(value)

#print(to_plot_m1)

#Transform date index from string type to date type
val_k.index = pd.DatetimeIndex(val_k.index).to_period('Y')

mod_k = sm.tsa.SARIMAX(val_k, order=(1, 0, 0))
res_k = mod_k.fit()
#print(res_k.summary())
#print(res_k.forecast(3))

to_plot_k = []
for i in range(27):
    to_plot_k.append(None)
for value in res_k.forecast(3):
    to_plot_k.append(value)

#print(to_plot_k)


mod_k1 = sm.tsa.ARIMA(val_k, order=(1, 0, 0))
res_k1 = mod_k1.fit()
#print(res_k1.summary())
#print(res_k1.forecast(3))


to_plot_k1 = []
for i in range(27):
    to_plot_k1.append(None)
for value in res_k1.forecast(3):
    to_plot_k1.append(value)

#print(to_plot_k1)



#print(to_plot_test_m)
#print(to_plot_test_k)
#print(lata)


from sklearn.metrics import mean_squared_error


def Q(prawdziwe, predykcja):
    return mean_squared_error(prawdziwe[-3:], predykcja[-3:])

print(f"błąd średnikwadratowy SARIMAX dla mężczyzn: {Q(to_plot_test_m, to_plot_m)}")
print(f"błąd średnikwadratowy ARIMA dla mężczyzn: {Q(to_plot_test_m, to_plot_m1)}")

print(f"błąd średnikwadratowy SARIMAX dla kobiet: {Q(to_plot_test_k, to_plot_k)}")
print(f"błąd średnikwadratowy ARIMA dla kobiet: {Q(to_plot_test_k, to_plot_k1)}")

#SARIMA MEZCZYZNI
fig,ax = plt.subplots()
ax.plot(lata, to_plot_m)
ax.plot(lata, to_plot_test_m)
plt.xlabel("Lata")
plt.ylabel("Wartość")
plt.title("Liczba zabójstw mężczyzn na 100 tys. w Polsce")
plt.legend(["Predykcja SARIMAX","Realne wartości"])
plt.show()
#ARIMA MEZCZYZNI
fig,ax = plt.subplots()
ax.plot(lata, to_plot_m1)
ax.plot(lata, to_plot_test_m)
plt.xlabel("Lata")
plt.ylabel("Wartość")
plt.title("Liczba zabójstw mężczyzn na 100 tys. w Polsce")
plt.legend(["Predykcja ARIMA","Realne wartości"])
plt.show()


#SARIMA KOBIETY
fig,ax = plt.subplots()
ax.plot(lata, to_plot_k)
ax.plot(lata, to_plot_test_k)
plt.xlabel("Lata")
plt.ylabel("Wartość")
plt.title("Liczba zabójstw kobiet na 100 tys. w Polsce")
plt.legend(["Predykcja SARIMAX","Realne wartości"])
plt.show()
#ARIMA KOBIETY
fig,ax = plt.subplots()
ax.plot(lata, to_plot_k1)
ax.plot(lata, to_plot_test_k)
plt.xlabel("Lata")
plt.ylabel("Wartość")
plt.title("Liczba zabójstw kobiet na 100 tys. w Polsce")
plt.legend(["Predykcja ARIMA","Realne wartości"])
plt.show()


print("Predykcja na przyszłe lata dla mężczyzn SARIMAX")
mod_m_test = sm.tsa.SARIMAX(val_m_test, order=(1, 0, 0))
res_m = mod_m_test.fit()
#print(res_m.summary())
print(res_m.forecast(10))

print("Predykcja na przyszłe lata dla kobiet ARIMA")
mod_k1_test = sm.tsa.ARIMA(val_k_test, order=(1, 0, 0))
res_k1 = mod_k1_test.fit()
#print(res_k1.summary())
print(res_k1.forecast(10))

for i in range(10):
    to_plot_test_m.append(None)
    to_plot_test_k.append(None)
    lata.append(rok)
    rok +=1

przyszlelata_mezczyzni_toplot = []
przyszlelata_kobiety_toplot = []

for i in range(30):
    przyszlelata_mezczyzni_toplot.append(None)
    przyszlelata_kobiety_toplot.append(None)

for value in res_m.forecast(10):
    przyszlelata_mezczyzni_toplot.append(value)

for value in res_k1.forecast(10):
    przyszlelata_kobiety_toplot.append(value)

#SARIMA MEZCZYZNI TRUE
fig,ax = plt.subplots()
ax.plot(lata, przyszlelata_mezczyzni_toplot)
ax.plot(lata, to_plot_test_m)
plt.xlabel("Lata")
plt.ylabel("Wartość")
plt.title("Liczba zabójstw mężczyzn na 100 tys. w Polsce, predykcja na przyszłość")
plt.legend(["Predykcja SARIMAX","Realne wartości"])
plt.show()


#ARIMA KOBIETY
fig,ax = plt.subplots()
ax.plot(lata, przyszlelata_kobiety_toplot)
ax.plot(lata, to_plot_test_k)
plt.xlabel("Lata")
plt.ylabel("Wartość")
plt.title("Liczba zabójstw kobiet na 100 tys. w Polsce, predykcja na przyszłość")
plt.legend(["Predykcja ARIMA","Realne wartości"])
plt.show()