#!/usr/bin/env python3
"""Autoencoder neural network for dimensionality reduction."""
import sys, math, random

def sigmoid(x): return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
def sigmoid_d(x): return x * (1 - x)

class Layer:
    def __init__(self, n_in, n_out):
        self.W = [[random.gauss(0, math.sqrt(2.0/n_in)) for _ in range(n_in)] for _ in range(n_out)]
        self.b = [0.0]*n_out; self.out = []; self.inp = []
    def forward(self, x):
        self.inp = x; self.out = []
        for i in range(len(self.b)):
            s = self.b[i] + sum(self.W[i][j]*x[j] for j in range(len(x)))
            self.out.append(sigmoid(s))
        return self.out

class Autoencoder:
    def __init__(self, dims, lr=0.1):
        self.layers = []; self.lr = lr
        for i in range(len(dims)-1): self.layers.append(Layer(dims[i], dims[i+1]))
        self.bottleneck_idx = len(dims)//2 - 1

    def forward(self, x):
        for layer in self.layers: x = layer.forward(x)
        return x

    def encode(self, x):
        for i in range(self.bottleneck_idx + 1): x = self.layers[i].forward(x)
        return x

    def train_step(self, x):
        output = self.forward(x)
        deltas = [None]*len(self.layers)
        L = self.layers[-1]
        deltas[-1] = [(x[i]-output[i])*sigmoid_d(output[i]) for i in range(len(output))]
        for l in range(len(self.layers)-2, -1, -1):
            layer = self.layers[l]; nxt = self.layers[l+1]
            deltas[l] = [sum(deltas[l+1][j]*nxt.W[j][i] for j in range(len(nxt.b)))*sigmoid_d(layer.out[i]) for i in range(len(layer.b))]
        for l, layer in enumerate(self.layers):
            for i in range(len(layer.b)):
                for j in range(len(layer.W[i])): layer.W[i][j] += self.lr*deltas[l][i]*layer.inp[j]
                layer.b[i] += self.lr*deltas[l][i]
        return sum((x[i]-output[i])**2 for i in range(len(x)))/len(x)

def main():
    random.seed(42)
    # Generate data: 8D with structure
    data = []
    for _ in range(100):
        a, b = random.gauss(0,1), random.gauss(0,1)
        data.append([a, b, a+b, a-b, a*0.5, b*0.5, a+0.1*random.gauss(0,1), b+0.1*random.gauss(0,1)])
    # Normalize to [0,1]
    for j in range(8):
        mn = min(d[j] for d in data); mx = max(d[j] for d in data)
        r = mx - mn or 1
        for d in data: d[j] = (d[j]-mn)/r

    ae = Autoencoder([8, 4, 2, 4, 8], lr=0.5)
    print("=== Autoencoder (8D -> 2D -> 8D) ===")
    for epoch in range(200):
        loss = sum(ae.train_step(d) for d in data)/len(data)
        if epoch % 50 == 0: print(f"Epoch {epoch}: MSE={loss:.6f}")

    print("\nEncoded samples (2D):")
    for i in range(5):
        enc = ae.encode(data[i])
        rec = ae.forward(data[i])
        err = sum((data[i][j]-rec[j])**2 for j in range(8))/8
        print(f"  [{enc[0]:.3f}, {enc[1]:.3f}] (recon error={err:.4f})")

if __name__ == "__main__": main()
