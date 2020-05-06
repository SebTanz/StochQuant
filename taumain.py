import subprocess
import threading
from queue import Queue

from io import BytesIO
from time import sleep, time

import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from sys import stdout

#potential = "harmosc"
#potential = "poeschlTeller"
potential = "double_well"
#potential = "quartic"


class dataThread (threading.Thread):
    def __init__(self, threadID, proc, q, e, cE):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.proc = proc
        self.q = q
        self.e = e
        self.cE = cE
        self.perc = 0.
        self.dtau = 0
    def run(self):
        now = time()
        while True:
            line = self.proc.stdout.readline()
            tmp=np.genfromtxt(BytesIO(line.strip()), delimiter="|")
            if tmp.size>1:
                self.dtau = tmp[-2]
                self.perc = tmp[-1]
            if line == b'' and self.proc.poll() is not None: 
                break
            elif not self.e.isSet():
                #print(tmp)
                #print("something")
                if tmp.size>1:
                    self.q.put(tmp[:-2])
                    if self.cE.isSet():
                        self.e.set()
            stdout.write("%6.2f" %self.perc+"%%| DeltaTau = %.2e"%self.dtau+"\r")
        stdout.write("100.00%"+"| DeltaTau = %.2e| "%self.dtau+"%.2f seconds\n"%(time()-now))
        #print("100.00%, "+str(time()-now)+" seconds")
        self.proc.stdout.close()
        self.proc.wait()

    
class animThread (threading.Thread):
    def __init__(self, threadID, frames, time, y, axlim, q, e, cE):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.frames = frames
        self.time = time
        self.y = y
        self.axlim = axlim
        self.q = q
        self.e = e
        self.cE = cE
    def run(self):
        fig, ax = plt.subplots()
        ln, = plt.plot(self.time, self.y, 'ro-', markersize=2)
        tx=plt.text(0, 0, str(0))

        def init():
            self.cE.set()
            ax.set_ylim(self.axlim[0], self.axlim[1])
            return ln,
        def update(frame):
            #print("something")
            if self.e.isSet():
                
                self.y = self.q.get()
                #print("something")
                #tempor = self.q.get()
                #nanidx = np.logical_not(np.isnan(tempor))
                #self.y[nanidx] = tempor[nanidx]
                self.e.clear()
            tx.set_text(str(self.y[0])+", "+str(self.y[-1]))
            #print(self.y)
            ln.set_data(self.time, self.y)
            return ln,
        ani = animation.FuncAnimation(fig, update, frames=self.frames, init_func=init, blit=False)

        plt.show()
        self.e.clear()
        self.cE.clear()

presets = {
    "harmosc": {
        "dtau": .3,
        "Nt": 100,
        "dt": .1,
        "potID":0,
        "theoVal":1.,
        "c":1.,
        "filename":"HarmOsc.txt"
    },
    "poeschlTeller": {
        "dtau": .1,
        "Nt": 100,
        "dt": 1.,
        "potID":1,
        "theoVal":.1315,
        "c":1.,
        "filename":"PoeschlTeller.txt"
    },
    "quartic": {
        "dtau": 0.01,
        "Nt": 50,
        "dt": 2,
        "potID":2,
        "theoVal":1.73,
        #"c":31.62277660168379332e-3,
        "c":1.,
        "filename":"Quartic.txt"
    },
    "double_well": {
        "dtau": .1,
        "Nt": 100,
        "dt": 1.,
        "potID":3,
        "theoVal":10,
        "c":1.,
        "filename":"tauDoubleWell4.txt"
    }
}
preset = presets[potential]
n = preset["Nt"]
deltat = preset["dt"]
deltatau = preset["dtau"]
h = 1e-5
parisi = 0
entw = 10000
potID = preset["potID"]
c = preset["c"]
strtval = preset["theoVal"]
mdpoint = 50.
device = 1
rpf = 1
intime = 0
loops = 10000
inputf = preset["filename"]
inputf = "0"
outputf = preset["filename"]
outputf = "0"
acco = 40

dmax=strtval+.5*strtval

p = subprocess.Popen(['./tauhost.o', str(n), str(deltat), str(deltatau), str(h), str(parisi), str(entw), str(potID), str(c), str(device), str(rpf), str(intime), str(loops), inputf, outputf, str(acco)], stdout=subprocess.PIPE, bufsize=1)
ev = threading.Event()
queue = Queue()
closeEv = threading.Event()

animt=animThread(1, entw, np.array(range(n-1))*deltat,np.zeros(n-1),[-dmax,dmax], queue, ev, closeEv)
datat=dataThread(2, p, queue, ev, closeEv)


ev.clear()
datat.start()

animt.start()














