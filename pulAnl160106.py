import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Funciones auxiliares - anl150807

def dataPathFile(mes, dia, acq):
    path= './data/'
    fNS= 'pul15'+ '{:02n}'.format(mes)+ '{:02n}'.format(dia)+ 'z'
    fNE= '.npy.npz'
    return path+ fNS+ '{:02n}'.format(acq)+fNE


def desvioEstandardCuasiNoSesgado(datos):
    '''
    Estimador cuasi no sesgado de la desviación estandar.
    https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    '''
    media= np.mean(datos)
    suma=0
    for registro in datos:
        aux= (registro- media)
#     for i in datos:
#         aux= (datos[i]- media)
        suma+= aux**2
    desv= np.sqrt(suma/( np.size(datos)- 1.5) ) 
    return desv


def errorEstandard(datos):
    '''
    Estimador de la desviación estandar del promedio.
    Usa la desviación estandar de la población y la divide por la raíz del número de elementos en la población.
    https://en.wikipedia.org/wiki/Standard_error
    '''
    return desvioEstandardCuasiNoSesgado(datos)/ np.sqrt(np.size(datos))


def baseMeas2(Meas2, messungenZahl, prop=0.25):
    '''
    (array, array, float) -> array
    
    De la primer fracción prop de ambos canales en Meas2,
    obtiene el promedio y dispersión que se asume como nivel de base
    grProm[0]: promedio ch1, grProm[1]: promedio ch2,
    grProm[2]: desviación estandard ch1, grProm[3]: desviación estandard ch2
    '''
    erstePunkten= int(messungenZahl[1]* prop)
    ch1Erste= Meas2[0][0:erstePunkten-1]
    ch2Erste= Meas2[1][0:erstePunkten-1]
    return np.array([ch1Erste.mean(), ch2Erste.mean(), ch1Erste.std(), ch2Erste.std() ] )


def tauCeti3eAreaMax(T, Meas2, messungenZahl, prop, conv1, rechazo=0.1):
    '''
    Calcula índices risetime como tiempo entre 10% y 90% de excursión entre nivel de base y máximo.
    Retorna area por sobre nivel de base de AMC entre 10% a izquierda y derecha.
   
    Meas2: array (2,:) con Meas2[0] canal 1, Meas2[0] canal 2, producto de np.vstack
    baseMeas2: base(Meas2)
    '''
    bMeas2= baseMeas2(Meas2, messungenZahl, prop) # niveles base
    # print(bMeas2[3],np.abs(bMeas2[1]* rechazo))
    
    if ((bMeas2[3])>np.abs(bMeas2[1]* rechazo)):
        # print('pasa tauCeti3eAreaMax')
        return 0,0,0,0,0,0,0
        # return np.array([0, 0 ])
    else:
        maxIx= (Meas2[0].argmax(), Meas2[1].argmax() ) # índices máximos ch1, ch2
        # máximo promediado cercano
        rangoCercano= 20 # tres lecturas a izq y derecha
        maxCh1= (Meas2[0][maxIx[0]- rangoCercano: maxIx[0]+ rangoCercano]).mean()
        maxCh2= (Meas2[1][maxIx[1]- rangoCercano: maxIx[1]+ rangoCercano]).mean()
        # diferencia máximo- base
        deltaV= np.array([maxCh1- bMeas2[0], maxCh2- bMeas2[1]]) # usa promediado cercano
        # Diferencias de potencial para 10%, 90% de excursión
        v10= bMeas2[0:2]+ 0.1* deltaV
        v90= bMeas2[0:2]+ 0.9* deltaV
        # Índices para primer potencial del pulso que exceda 10%, y último por debajo 90% 
        ixTau= np.array([maxIx[0], maxIx[1], maxIx[0], maxIx[1]])
        
        # ch1>10% sobre línea de base [ancho Pulser]
        while (Meas2[0,ixTau[2]]> v10[0]): # para ancho en Pulser
            ixTau[2]+= 1        
        #while (Meas2[0,ixTau[2]]> v90[0]): # para tau en FLUC
        #    ixTau[2]-= 1        
        while (Meas2[0,ixTau[0]]> v10[0]):
            ixTau[0]-= 1
        anchoPulser= T[ixTau[2]]- T[ixTau[0]]
        # ch1 area [area pulser]
        # sumo diferencia con línea de base bMeas2 por encima de 10%
        areaPulser= (Meas2[0][ixTau[0]: ixTau[2]] - bMeas2[0]).sum()
        puntoMedioPulser= np.int((ixTau[2]- ixTau[0])/2)
        maxPulser= (Meas2[0][puntoMedioPulser- rangoCercano: puntoMedioPulser+ rangoCercano]).mean()
        
        # 10->90% ch2 [tau AMC]
        while (Meas2[1,ixTau[1]]> v10[1]):
            ixTau[1]-= 1
        while (Meas2[1,ixTau[3]]> v90[1]):
            ixTau[3]-= 1
        tauAMC= T[ixTau[3]]- T[ixTau[1]]
            
        # ch2Area [AMC]
        ixAMC10ProzentRicht= maxIx[1] # empiezo por máximo
        while ((Meas2[1, ixAMC10ProzentRicht]> v10[1]) and (ixAMC10ProzentRicht< messungenZahl[1]-1)):
            ixAMC10ProzentRicht+= 1
        if (ixAMC10ProzentRicht== messungenZahl[1]):
            return 0,0,0,0,0,0,0 # Condición falla de área, pues llego desde el max hasta el final sin haber bajado
        # ixTau[1], maxIx[1], ixAMC10ProzentRicht
        # hay que sumar diferencia AMC con este nivel desde el índice 10% izquierda hasta derecha

        areaTotalAMC= (Meas2[1][ixTau[1]: ixAMC10ProzentRicht] - bMeas2[1]).sum() # suma alturas sobre bMeas2[1] (nivel base AMC)
        areaSubidaAMC= (Meas2[1][ixTau[1]: maxIx[1]] - bMeas2[1]).sum()
        maxAMC= maxCh2
        
        # verificación a través de graficación
        # plota2(conv1, T, Meas2, messungenZahl, prop, bMeas2, maxIx, ixTau)
        
        return anchoPulser, areaPulser, maxPulser, tauAMC, areaTotalAMC, areaSubidaAMC, maxAMC


# Tomado del exótico anl1501019
class conv(object):
    __slots__ = ['ch1Cero', 'ch1Paso', 'ch2Cero', 'ch2Paso']

    
def parametrosConversion(npzData):
    """ acq -> conv

    Da factores de conservión para de Meas2 con niveles en enteros obtener niveles en potencial eléctrico.
    e.g. potencialAMC= (Meas2[1,:]- ceroAMCPto)* pasoVerticalAMCVoltPto
    
    >>> cusi= conversionV(acq)
    >>> cusi.ch1Cero
    0.0013
    """
    # Constructivo del Tek 2002B
    divVert= 8 # numero de divisiones verticales
    divHoriz= 10
    ptosVert= 2**8 # 8 bits definición vertical
    ptosHoriz= 2500
    # npZData.items() # lista los elementos en el archivo de aquisición en memoria
    settingsList= npzData['settings'].tolist()
    
    # Vertical (potencial)
    conv1= conv()
    escalaCh1VoltsDiv= settingsList['SCALE1'] # [V/div]
    conv1.ch1Paso= escalaCh1VoltsDiv* divVert/ ptosVert # [V/div]
    escalaCh2VoltsDiv= settingsList['SCALE2'] # [V/div]
    conv1.ch2Paso= escalaCh2VoltsDiv* divVert/ ptosVert # [V/div]
    ceroCh1Div= settingsList['POSITION1'] # [div]
    conv1.ch1Cero= ceroCh1Div* divVert/ ptosVert # [ptos]
    ceroCh2Div= settingsList['POSITION2'] # [div]
    conv1.ch2Cero= ceroCh2Div* ptosVert/ divVert # [ptos]
    
    return conv1


def conversionV2(conv1, T, areaPulser, maxPulser, areaSubidaAMC, areaTotalAMC, maxAMC):
    # parametros área/máximo en volts
    pasoT= T[1]-T[0] # paso temporal en cada muestreo sumado área
    areaPulser*= pasoT* conv1.ch1Paso
    maxPulser= (maxPulser- conv1.ch1Cero)* conv1.ch1Paso
    areaSubidaAMC*= pasoT* conv1.ch2Paso
    areaTotalAMC*= pasoT* conv1.ch2Paso
    maxAMC= (maxAMC- conv1.ch2Cero)* conv1.ch2Paso

    # return conv
    return areaPulser, maxPulser, areaSubidaAMC, areaTotalAMC, maxAMC
    
def plota2(conv1, T, Meas2, messungenZahl, prop, bMeas2, maxIx, ixTau):

# def plota2(ceroCh2Pto, pasoVerticalCh2VoltPto, ceroCh1Pto, pasoVerticalCh1VoltPto, T, Meas2, messungenZahl, prop):
    '''
    Grafica en valores físicos de ambos canales, niveles de base (izquierda), máximos, 10% y 90% del risetime

    (T, Meas2) (maxCH1_index, maxCH2_index, iProcent10, iProcent90) -> 
    '''
    # ceroCh2Pto, pasoVerticalCh2VoltPto, ceroCh1Pto, pasoVerticalCh1VoltPto= conversionV(npzData)
    # bMeas2= baseMeas2(Meas2, messungenZahl, prop) # niveles base
    

    # Gráfica max, nivel cero, etc.
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    fig.set_size_inches(10,8)
    fig.subplots_adjust(hspace=0.000)
    # Axis 0
    ax0.plot(T, (Meas2[0,:]- conv1.ch1Cero)* conv1.ch1Paso, 'y')
    # Make the y-axis label and tick labels match the line color.
    ax0.set_ylabel('ch 1 [V]', color='y')
    ax0.grid(True)
    # Axis 1
    ax1.plot(T, (Meas2[1,:]- conv1.ch2Cero)* conv1.ch2Paso, 'c')
    ax1.set_ylabel('ch 2 [V]', color='c')
    ax1.set_xlabel('tiempo [s]')
    ax1.grid(True)
    ## zero level lines
    erstePunkten= int(messungenZahl[1]* prop) # auxiliar
    zeroT= T[0:(erstePunkten- 1)]
    zeroAx0= np.ones([erstePunkten- 1, 1])* (bMeas2[0]- conv1.ch1Cero)* conv1.ch1Paso
    zeroAx1= np.ones([erstePunkten- 1, 1])* (bMeas2[1]- conv1.ch2Cero)* conv1.ch2Paso
    ax0.plot(zeroT, zeroAx0, 'k-')
    ax1.plot(zeroT, zeroAx1, 'k-')
    ## marks maxima
    ax0.scatter(T[maxIx[0]], (Meas2[0,:][maxIx[0]]- conv1.ch1Cero)* conv1.ch1Paso, marker='o')
    ax1.scatter(T[maxIx[1]], (Meas2[1,:][maxIx[1]]- conv1.ch2Cero)* conv1.ch2Paso, marker='o')
    ## risetime ch1
    ch1Erste= (Meas2[0,:][ixTau[0]]- conv1.ch1Cero)* conv1.ch1Paso
    ch1Letzte= (Meas2[0,:][ixTau[2]]- conv1.ch1Cero)* conv1.ch1Paso
    ax0.scatter(T[ixTau[0]], ch1Erste, marker='o')
    ax0.scatter(T[ixTau[2]], ch1Letzte, marker='o')
    ## risetime ch2
    ch2percent10= (Meas2[1,:][ixTau[1]]- conv1.ch2Cero)* conv1.ch2Paso
    ch2percent90= (Meas2[1,:][ixTau[3]]- conv1.ch2Cero)* conv1.ch2Paso
    ax1.scatter(T[ixTau[1]], ch2percent10, marker='o')
    ax1.scatter(T[ixTau[3]], ch2percent90, marker='o')

    #plt.savefig('1090prozent.pdf')
    plt.show()


def tausAreasMaxSerieAdq(npzData, prop, rechazo):
    '''
    Vectores risetime canal AMC de archivo acq comprimido
    
    npzData= numpy.lib.npyio.NpzFile
    prop= porcentaje inferior del canal a promediar para obtener nivel de base     
    '''
    FLUC= npzData['ch1']
    AMC= npzData['ch2']
    messungenZahl= AMC.shape
    # tiempo
    T= npzData['zeit']

    anchoPulser= np.empty([0, 1])
    areaPulser= np.empty([0, 1])
    maxPulser= np.empty([0, 1])
    tauAMC= np.empty([0, 1])
    areaSubidaAMC= np.empty([0, 1])
    areaTotalAMC= np.empty([0, 1])
    maxAMC= np.empty([0, 1])

    conv1= parametrosConversion(npzData)
    
    i=0
    q=0
    cucho= (messungenZahl[0]- q- 1 )
    while (i< cucho ): # recorre cada forma de onda en medición npzData
        Meas2= np.array([FLUC[i+q], AMC[i+q] ] )
        anchoPulseri, areaPulseri, maxPulseri, tauAMCi, areaTotalAMCi, areaSubidaAMCi, maxAMCi= tauCeti3eAreaMax(T, Meas2, messungenZahl, prop, conv1, rechazo)
        # print('{:04n}'.format(i))
        # print (tauAMCi, areaTotalAMCi, areaSubidaAMCi, maxAMCi)
        if (tauAMCi!=0):
            i+=1
            anchoPulser= np.append(anchoPulser, anchoPulseri)
            areaPulser= np.append(areaPulser, areaPulseri)
            maxPulser= np.append(maxPulser, maxPulseri)
            tauAMC= np.append(tauAMC, tauAMCi)
            areaSubidaAMC= np.append(areaSubidaAMC, areaSubidaAMCi)
            areaTotalAMC= np.append(areaTotalAMC, areaTotalAMCi)
            maxAMC= np.append(maxAMC, maxAMCi)
        else:
            # print('pasa tausAreasMaxSerieAdq')
            q+=1
        cucho= (messungenZahl[0]- q- 1 )
        
    # conversión área/máximo -> volts*s/volts
    areaPulser, maxPulser, areaSubidaAMC, areaTotalAMC, maxAMC= conversionV2(conv1, T, areaPulser, maxPulser, areaSubidaAMC, areaTotalAMC, maxAMC)
    
    return anchoPulser, areaPulser, maxPulser, tauAMC, areaSubidaAMC, areaTotalAMC, maxAMC


# agregador area
def agregadorPulser(mes, dia, rango, prop= 0.25, rechazo= 0.09):
    
    anchosPulser= np.array([])
    areasPulser= np.array([])
    maximosPulser= np.array([])
    
    tausAMC= np.array([])
    areasTotalAMC= np.array([])
    areasSubidaAMC= np.array([])
    maximosAMC= np.array([])
    for i in rango:
        acq= np.load(dataPathFile(mes, dia, i))
        anchoPulser, areaPulser, maxPulser, tauAMC, areaSubidaAMC, areaTotalAMC, maxAMC= tausAreasMaxSerieAdq(acq, prop, rechazo)
        
        anchosPulser= np.append(anchosPulser,anchoPulser)
        areasPulser= np.append(areasPulser, areaPulser)
        maximosPulser= np.append(maximosPulser, maxPulser)
        
        tausAMC= np.append(tausAMC, tauAMC)
        areasTotalAMC= np.append(areasTotalAMC, areaTotalAMC)
        areasSubidaAMC= np.append(areasSubidaAMC, areaSubidaAMC)
        maximosAMC= np.append(maximosAMC, maxAMC)
        
    # Falta convertir areas y máximos reales con datos de 
        
    return anchosPulser, areasPulser, maximosPulser, tausAMC, areasTotalAMC, areasSubidaAMC, maximosAMC


# Enmascarado

def binner(taus, fracBinsHist, segmentos= 120):
    binsRecorte= np.int(fracBinsHist*segmentos)
    fig = plt.figure(figsize=(18,5))  # an empty figure with no axes
    ax_lst= fig.add_subplot(1,2,2)
    n, bins, patches= ax_lst.hist(taus, segmentos)
    ax_lst.set_ylabel('Cuentas')
    ax_lst.set_xlabel('Tau [ns]')
    intermedio = np.ma.masked_greater(taus, bins[binsRecorte])
    mascara= intermedio.mask
    primerCampana= intermedio.compressed()
    ax_lst2= fig.add_subplot(1,2,1)
    ax_lst2.set_ylabel('Cuentas')
    ax_lst2.set_xlabel('Tau [ns]')
    n_i, bins_i, patches_i= ax_lst2.hist(primerCampana, bins=np.int(segmentos/3))
    print('puntos campana {:d}'.format(primerCampana.size))
    return mascara, primerCampana
    #return primerCampana.size, primerCampana.mean(), errorEstandard(primerCampana), mascara

    
def enmascarador(datos, mascara):
    datosEnmascarados= np.ma.MaskedArray(datos,mascara)
    salida= datosEnmascarados.compressed() # datosComprimidos
    return salida # datosComprimidos


def todoEnmascarado(mascara, anchosPulser, areasPulser, maximosPulser, tausAMC, areasTotalAMC, areasSubidaAMC, maximosAMC):
    anchosPulserComprimidos= enmascarador(anchosPulser, mascara)
    print('anchoPulser= ', anchosPulserComprimidos.mean(), errorEstandard(anchosPulserComprimidos))

    areasPulserComprimidos= enmascarador(areasPulser, mascara)
    print('areaPulser= ', areasPulserComprimidos.mean(), errorEstandard(areasPulserComprimidos))

    maximosPulserComprimidos= enmascarador(maximosPulser, mascara)
    print('maxPulser= ', maximosPulserComprimidos.mean(), errorEstandard(maximosPulserComprimidos))

    #mTau= tausAMCComprimidos.mean(), emTau= errorEstandard(tausAMCComprimidos)
    # print('tau= '+'{:.03e}'.format(mTau)+ ' +/- '+ '{:.03e}'.format(emTau))
    print('tauAMC= ', tausAMCComprimidos.mean(), errorEstandard(tausAMCComprimidos))

    areasSubidaAMCComprimidos= enmascarador(areasSubidaAMC, mascara)
    print('areaSubidaAMC= ', areasSubidaAMCComprimidos.mean(), errorEstandard(areasSubidaAMCComprimidos))

    areasTotalAMCComprimidos= enmascarador(areasTotalAMC, mascara)
    print('areaTotalAMC= ', areasTotalAMCComprimidos.mean(), errorEstandard(areasTotalAMCComprimidos))

    maximosAMCComprimidos= enmascarador(maximosAMC, mascara)
    print('maximoAMC= ', maximosAMCComprimidos.mean(), errorEstandard(maximosAMCComprimidos))
    
    
class resultadosComprimido(object):
    __slots__ = ['anchoPulser', 'anchoPulserError'
                 , 'areaPulser', 'areaPulserError'
                 , 'maxPulser', 'maxPulserError'
                 , 'tauAMC', 'tauAMCError'
                 , 'areaSubidaAMC', 'areaSubidaAMCError'
                 , 'areaTotalAMC', 'areaTotalAMCError'
                 , 'maximoAMC', 'maximoAMCError'
                 ]


def todoEnmascarado2(mascara, anchosPulser, areasPulser, maximosPulser, tausAMCComprimidos, areasTotalAMC, areasSubidaAMC, maximosAMC):
    resultados=resultadosComprimido()

    anchosPulserComprimidos= enmascarador(anchosPulser, mascara)
    resultados.anchoPulser, resultados.anchoPulserError= anchosPulserComprimidos.mean(), errorEstandard(anchosPulserComprimidos) 
    print('anchoPulser= \t{:.03e} \t+/- {:.03e} \t s'.format(resultados.anchoPulser ,resultados.anchoPulserError ) )
    # print('anchoPulser= ', resultados.anchoPulser , '+/-', resultados.anchoPulserError )

    areasPulserComprimidos= enmascarador(areasPulser, mascara)
    resultados.areaPulser, resultados.areaPulserError= areasPulserComprimidos.mean(), errorEstandard(areasPulserComprimidos)
    print('areaPulser= \t{:.03e} \t+/- {:.03e} \t Vs'.format(resultados.areaPulser, resultados.areaPulserError ) )
    # print('areaPulser= ', resultados.areaPulser, '+/-',  resultados.areaPulserError )

    maximosPulserComprimidos= enmascarador(maximosPulser, mascara)
    resultados.maxPulser, resultados.maxPulserError= maximosPulserComprimidos.mean(), errorEstandard(maximosPulserComprimidos)
    print('maxPulser= \t{:.03e} \t+/- {:.03e} \t V'.format(resultados.maxPulser, resultados.maxPulserError) )
    # print('maxPulser= ', resultados.maxPulser ,'+/-' , resultados.maxPulserError)

    resultados.tauAMC, resultados.tauAMCError = tausAMCComprimidos.mean(), errorEstandard(tausAMCComprimidos)
    print('tauAMC= \t{:.03e} \t+/- {:.03e} \t s'.format(resultados.tauAMC, resultados.tauAMCError) )

    areasSubidaAMCComprimidos= enmascarador(areasSubidaAMC, mascara)
    resultados.areaSubidaAMC, resultados.areaSubidaAMCError= areasSubidaAMCComprimidos.mean(), errorEstandard(areasSubidaAMCComprimidos)
    print('areaSubidaAMC= \t{:.03e} \t+/- {:.03e} \t Vs'.format(resultados.areaSubidaAMC, resultados.areaSubidaAMCError) )

    areasTotalAMCComprimidos= enmascarador(areasTotalAMC, mascara)
    resultados.areaTotalAMC, resultados.areaTotalAMCError= areasTotalAMCComprimidos.mean(), errorEstandard(areasTotalAMCComprimidos)
    print('areaTotalAMC= \t{:.03e} \t+/- {:.03e} \t Vs'.format(resultados.areaTotalAMC, resultados.areaTotalAMCError ) )

    maximosAMCComprimidos= enmascarador(maximosAMC, mascara)
    resultados.maximoAMC, resultados.maximoAMCError= maximosAMCComprimidos.mean(), errorEstandard(maximosAMCComprimidos)
    print('maximoAMC=  \t{:.03e} \t+/- {:.03e} \t V'.format(resultados.maximoAMC, resultados.maximoAMCError ) )
    
    return resultados
