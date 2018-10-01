import koishi
from tqdm import tqdm

class lr:

    def __init__(self, filename):
        self.km,self.price = self.readData(filename)
        self.initVariable = koishi.fillInitializer(0)
        self.initSize = koishi.fillInitializer(len(self.km))
        self.feedInputs = koishi.feedInitializer(self.km)
        self.feedOutputs = koishi.feedInitializer(self.price)
        two = koishi.Tensor(2)
        minus = koishi.Tensor(-1)
        
        self.inputs = koishi.Tensor([len(self.km),1], "inputs", self.feedInputs)
        self.outputs = koishi.Tensor([len(self.km),1], "outputs", self.feedOutputs)
        self.theta0 = koishi.Tensor("variable", self.initVariable)
        self.theta1 = koishi.Tensor([1,1], "variable", self.initVariable)
        self.m = koishi.Tensor("size", self.initSize)
        self.estimation = self.inputs.matmul(self.theta1).add(self.theta0)
        self.cost = self.estimation.add(self.outputs.multiply(minus)).pow(two).sum().multiply(self.m.multiply(two).inverse())

        koishi.initializeAll()

    def normalize(self, value, minv, rangev):
        return (value-minv) / rangev

    def readData(self, filename):
        with open(filename) as f:
            lines = f.readlines()
        lines = [l.strip("\n").split(",") for l in lines[1:]]
        km = [float(l[0]) for l in lines]
        price = [float(l[1]) for l in lines]
        self.kmRange = max(km) - min(km)
        self.priceRange = max(price) - min(price)
        self.kmMin = min(km)
        self.priceMin = min(price)
        km = [self.normalize(v, self.kmMin, self.kmRange) for v in km]
        price = [self.normalize(v, self.priceMin, self.priceRange) for v in price]
        return km, price
    
    def train(self, epoch, learningRate):
        for i in tqdm(range(epoch)):
            self.cost.gradientReinit()
            self.cost.gradientUpdate()
            koishi.gradientDescent("variable", learningRate)
        
