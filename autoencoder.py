#!/usr/bin/env python3
"""autoencoder - Simple autoencoder neural network."""
import sys, random, math
def sigmoid(x): return 1/(1+math.exp(-max(-500,min(500,x))))
class Autoencoder:
    def __init__(self, input_dim, hidden_dim):
        self.w_enc = [[random.gauss(0,0.5) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b_enc = [0.0]*hidden_dim
        self.w_dec = [[random.gauss(0,0.5) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b_dec = [0.0]*input_dim
    def encode(self, x):
        return [sigmoid(sum(self.w_enc[j][i]*x[i] for i in range(len(x)))+self.b_enc[j]) for j in range(len(self.w_enc))]
    def decode(self, h):
        return [sigmoid(sum(self.w_dec[j][i]*h[i] for i in range(len(h)))+self.b_dec[j]) for j in range(len(self.w_dec))]
    def train(self, X, lr=0.5, epochs=500):
        for epoch in range(epochs):
            total_loss = 0
            for x in X:
                h = self.encode(x); out = self.decode(h)
                loss = sum((x[i]-out[i])**2 for i in range(len(x))); total_loss += loss
                d_out = [2*(out[i]-x[i])*out[i]*(1-out[i]) for i in range(len(x))]
                d_h = [sum(d_out[j]*self.w_dec[j][i] for j in range(len(x)))*h[i]*(1-h[i]) for i in range(len(h))]
                for j in range(len(x)):
                    for i in range(len(h)): self.w_dec[j][i] -= lr*d_out[j]*h[i]
                    self.b_dec[j] -= lr*d_out[j]
                for j in range(len(h)):
                    for i in range(len(x)): self.w_enc[j][i] -= lr*d_h[j]*x[i]
                    self.b_enc[j] -= lr*d_h[j]
            if epoch%100==0: print(f"Epoch {epoch}: loss={total_loss/len(X):.6f}")
if __name__=="__main__":
    random.seed(42)
    X = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    ae = Autoencoder(4, 2); ae.train(X, epochs=1000)
    print("Reconstructions:")
    for x in X:
        h = ae.encode(x); out = ae.decode(h)
        print(f"  {x} -> [{', '.join(f'{v:.2f}' for v in out)}]  (code: [{', '.join(f'{v:.2f}' for v in h)}])")
