import numpy as np

class RunningHist():
    def __init__(self,length=10):
        self.length = length
        self.history = []
        self.length = length
        self.halfWindow = 100 # delay from -100 to 100
        self.freq = np.array([0 for i in range(2*self.halfWindow+1)])

    def addNum(self,num):
        if abs(num) > self.halfWindow:
            return
        self.history.append(num)
        self.freq[num+self.halfWindow] += 1
        if len(self.history) > self.length:
            toDel = self.history.pop(0)
            self.freq[toDel+self.halfWindow] -= 1
        
    def freqFor(self,array):
        return self.freq[array + self.halfWindow]

if __name__ == "__main__":
    sut = RunningHist(5)
    sut.addNum(3)
    sut.addNum(3)
    sut.addNum(3)
    sut.addNum(3)
    sut.addNum(3)
    sut.addNum(4)
    print sut.freqFor(np.array([3]))
    print sut.freqFor(np.array([4]))
    print sut.freqFor(np.array([5]))
