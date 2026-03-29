#!/usr/bin/env python3
"""Simple autoencoder for dimensionality reduction."""
import sys, math, random

def sigmoid(x): return 1/(1+math.exp(-max(-500,min(500,x))))

class Autoencoder:
    def __init__(self, input_dim, hidden_dim):
        self.id, self.hd = input_dim, hidden_dim
        s = math.sqrt(2.0/input_dim)
        self.W1 = [[random.gauss(0,s) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0]*hidden_dim
        self.W2 = [[random.gauss(0,s) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b2 = [0]*input_dim
    def encode(self, x):
        return [sigmoid(sum(self.W1[j][k]*x[k] for k in range(self.id))+self.b1[j]) for j in range(self.hd)]
    def decode(self, h):
        return [sigmoid(sum(self.W2[j][k]*h[k] for k in range(self.hd))+self.b2[j]) for j in range(self.id)]
    def forward(self, x):
        self.h = self.encode(x); self.out = self.decode(self.h); return self.out
    def train(self, X, epochs=100, lr=0.5):
        for _ in range(epochs):
            total_loss = 0
            for x in X:
                out = self.forward(x)
                err = [out[j]-x[j] for j in range(self.id)]
                total_loss += sum(e**2 for e in err)
                d_out = [err[j]*out[j]*(1-out[j]) for j in range(self.id)]
                d_h = [sum(d_out[j]*self.W2[j][k] for j in range(self.id))*self.h[k]*(1-self.h[k]) for k in range(self.hd)]
                for j in range(self.id):
                    for k in range(self.hd): self.W2[j][k] -= lr*d_out[j]*self.h[k]
                    self.b2[j] -= lr*d_out[j]
                for j in range(self.hd):
                    for k in range(self.id): self.W1[j][k] -= lr*d_h[j]*x[k]
                    self.b1[j] -= lr*d_h[j]
            if _ % 100 == 0: print(f"Epoch {_}: loss={total_loss/len(X):.4f}")

def main():
    random.seed(42)
    X = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    ae = Autoencoder(4, 2); ae.train(X, epochs=500, lr=1.0)
    for x in X:
        h = ae.encode(x); out = ae.forward(x)
        print(f"  {x} -> encoded: [{h[0]:.2f},{h[1]:.2f}] -> [{','.join(f'{o:.2f}' for o in out)}]")

if __name__ == "__main__": main()
