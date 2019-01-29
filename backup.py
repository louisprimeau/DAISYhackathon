import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

radius = pd.read_csv('track_1.csv').values.tolist()
for i in range(0,len(radius),1):
    radius[i] = radius[i][0]
radius = np.asarray(radius)

acceleration = 10
breaking = -30
topspeed = 20
gascap = 750
tiredur = 750
handling = 21

def maxvelocity(radius):#Calculate max velocity
    for i in range(0,radius.size,1):
        if radius[i] < 0:
            radius[i] = np.NaN
    maxvel = np.sqrt(radius * handling / 1000000)
    for i in range(0,maxvel.size,1):
        if maxvel[i] > 50:
            maxvel[i] = 50
    maxvel[np.isnan(maxvel)] = 50

    return(maxvel)

"""def getminimums(vector,indices):
    values = []
    indies = []
    for i in range(0,vector.size - 1,1):
        if(vector[i] < vector[i+1] and vector[i] < vector[i-1]):
            values += [vector[i]]
            indies += [indices[i]]
    return np.asarray(values), np.asarray(indies)"""

def getminimums(vector,num):
    idx = np.argpartition(vector, num)
    idx = idx[0:num]
    vec = vector[idx]
    vec = vec[np.argsort(idx)]
    idx = np.sort(idx)
    return vec,idx

def derivative(vec, ind):
	ax = np.zeros(vec.size)
	for i in range(0, ax.size - 1 ,1):
		ax[i] = (vec[i+1]**2 - vec[i]**2) / (2 * (ind[i+1] - ind[i]))
	return (ax)

def integrate(ax):
    vx = np.zeros(ax.size)
    for i in range(0,vx.size-1,1):
        if(vx[i]**2 + 2 * ax[i] * 1 < 0):
            vx[i+1] = 0
        else:
            vx[i + 1] = np.sqrt(vx[i]**2 + 2 * ax[i] * 1)
    return(vx)

def expand(vec, ind):
    append = []
    sum = 0
    append += int(ind[0] - 0) * [vec[0]]
    for i in range(0,ind.size - 1, 1):
        sum += ind[i+1] - ind[i]
        append += [vec[i]] * int(ind[i+1] - ind[i])
    append += (1000 - int(ind[ind.size - 1])) * [vec[vec.size-1]]
    return(np.asarray(append))

def contract(vec):
    last = -1
    new = []
    index = []
    for i in range(0,vec.size,1):
        if (i != last):
            index += [i]
            new += [vec[i]]
        else:
            last = vec[i]
    return np.asarray(new), np.asarray(index)

def valid(velocity,max):
    for i in range(0,velocity.size,1):
        if(velocity[i] > max[i]):
            print(i)
            return(False)
    return(True)

def time(ax,vx):
    time = 0
    for i in range(1,ax.size - 1,1):
        dt = 0
        if(ax[i] == 0):
            dt = 1 / vx[i]
        elif(ax[i] > 0):
            dt = (vx[i + 1] - vx[i]) / ax[i]
        time += dt
    return(time)

def limit(ax):
    for i in range(0,ax.size,1):
        if(ax[i] > acceleration):
            ax[i] = acceleration
        elif(ax[i] < breaking):
            ax[i] = breaking
    return(ax)

def pit_stop(ax,index):
	gas_used = 0
	break_used = 0
	stop = -1
	step = -1
	for i in range(0,ax.size,1):
		if ax[i] > 0:
			gas_used += 0.1*(ax[i]**2)
		else:
			break_used += 0.1*(ax[i]**2)
		if (break_used >= tiredur) or (gascap >= 1000):
			stop = i
			break
	if stop != -1:
		vx = integrate(ax)
		for r in range(1,stop,1):
			if vx[stop-r] < 2:
				vx[stop-r] = 0
				step = stop - r
				break
		ax = derivative(vx,index)
	return step,ax

def respread(ax,repeat):
    for i in range(0,repeat,1):
        ax = spread(ax)
    return(limit(ax))

def spread(ax):
    ax1 = np.copy(ax)
    spread_range = 10
    for i in range(spread_range,ax1.size,1):
        if ax1[i] < breaking:
            average = 0
            for j in range(0,-spread_range,-1):
                average += ax1[i+j]
            average /= spread_range
            for j in range(0,-spread_range,-1):
                ax1[i+j] = average
    for i in range(0,ax.size-spread_range,1):
        if ax1[i] > acceleration:
            average = 0
            for j in range(0,spread_range,1):
                average += ax1[i+j]
            average /= spread_range
            for j in range(0,spread_range,1):
                ax1[i+j] = average
    return(ax1)

def casezero(ax,vx,vmax):
    for i in range(0,vx.size,1):
        if(vx[i] == 0 and vx[i+1] == 0):
            ax[i] = 1
    return(ax)

"""def smooth(ax,index):
    rangesmooth = 5
    if(ax[index] < 0):
        average = 0
        for i in range(0,-rangesmooth, -1):
            average += ax[index+i]
        average /= rangesmooth
        for i in range(0,-rangesmooth, -1):
            ax[index+i] = average
    elif(ax[index] > 0):
        average = 0
        for i in range(0,rangesmooth, 1):
            average += ax[index+i]
        average /= rangesmooth
        for i in range(0,rangesmooth, 1):
            ax[index+i] = average
    return(ax)"""

def resafety(ax,vx,vmax,iterations):
    for i in range(0,iterations,1):
        ax = safety(ax,vx,vmax)
        vx = integrate(ax)
    return(ax)

def safety(ax,vx,targetvel):
    for i in range(0,vx.size,1):
        if(vx[i] > targetvel[i]):
            vx[i] = targetvel[i] - 0.000005
    ax = derivative(vx,np.linspace(0,ax.size,num=(ax.size+1)))
    return(ax)


#Generate acceleration
maxvel = maxvelocity(radius)
mins1,interval1 = getminimums(maxvel,30)
#mins2,interval2 = getminimums(mins1, interval1)
#mins1,interval1 = mins2,interval2
#mins2,interval2 = getminimums(mins2, interval2)

#mins2,interval2 = getminimums(mins2, interval2)

#Generate target velocity
"""interval2 = np.insert(interval2,0,0)
interval2 = np.insert(interval2,interval2.size,1000)
mins2 = np.insert(mins2,0,0)
mins2 = np.insert(mins2,mins2.size,0)"""

"""interval1 = np.insert(interval1,0,0)
interval1 = np.insert(interval1,interval1.size,1000)
mins1 = np.insert(mins1,0,0)
mins1 = np.insert(mins1,mins1.size,0)"""
average = mins1
series = interval1
mins2 = expand(mins1,series)
"""timeseries1 = expand(mins1, interval1)
timeseries2 = expand(mins2, interval2)
average = (timeseries1 + timeseries2)/2
average, series = contract(average)"""

#Initialize Acceleration
ax = derivative(average,series)
ax = expand(ax,series)
ax = respread(ax,5)

#Test for pit stops
stop,ax = pit_stop(ax,np.linspace(0, 999, num=1000))

#Check for special cases
vx = integrate(ax)
ax = resafety(ax,vx,mins2,10)
ax = respread(ax,100)
vx = integrate(ax)
ax = resafety(ax,vx,mins2,10)
vx = integrate(ax)
ax = casezero(ax,vx,maxvel)
vx = integrate(ax)
ax = resafety(ax,vx,mins2,10)
ax = limit(ax)
vx = integrate(ax)

#Check validity, time
print(valid(vx, maxvel))
print(time(ax,vx))

#Plot velocity data
plt.plot(np.linspace(0, 999, num=1000), maxvel)
plt.plot(series, average)
plt.plot(np.linspace(0, 999, num=1000),vx)
plt.legend(['maxvel','mins','car'])
plt.show()

#Plot acceleration data
plt.plot(np.linspace(0, 999, num=1000),ax)
plt.show()
