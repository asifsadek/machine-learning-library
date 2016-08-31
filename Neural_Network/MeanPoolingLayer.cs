using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning.Neural_Network {

    public class MeanPoolingLayer : PoolingLayer {

        public MeanPoolingLayer (int kernelWidth, int kernelHeight) : base(kernelWidth, kernelHeight) { }

        public MeanPoolingLayer (StreamReader reader, Layer prev) {
            string[] data = reader.ReadLine().Split();
            this.kernelWidth = int.Parse(data[0]);
            this.kernelHeight = int.Parse(data[1]);

            BindTo(ref prev);
        }

        public override void BindTo (ref Layer layer) {
            prevLayer = layer;

            Layer2D prev = (Layer2D)prevLayer;

            if (prev.width % this.kernelWidth != 0 || prev.height % this.kernelHeight != 0)
                throw new ArgumentException();

            this.depth = prev.depth;
            this.width = prev.width / this.kernelWidth;
            this.height = prev.height / this.kernelHeight;
            this.size = depth * width * height;

            this.neurons = new Neuron[size];

            for (int i = 0; i < size; i++)
                neurons[i] = new Neuron(this.kernelWidth * this.kernelHeight, new WeightSet(0));

            // depth = prev.depth
            for (int d = 0; d < depth; d++)
                for (int i = 0; i < width; i++)
                    for (int j = 0; j < height; j++)
                        for (int m = 0; m < kernelWidth; m++)
                            for (int n = 0; n < kernelHeight; n++) {
                                int prevWidth = i * kernelWidth + m;
                                int prevHeight = j * kernelHeight + n;
                                int prevIndex = d * prev.width * prev.height + prevWidth * prev.height + prevHeight;
                                neurons[d * width * height + i * height + j].prev[m * kernelHeight + n] = prev.neurons[prevIndex];
                            }
        }

        // depth = p.depth
        public override void forwardPropagate () {
            for (int i = 0; i < size; i++) {
                double sum = 0;
                for (int j = 0; j < neurons[i].size; j++)
                    sum += neurons[i].prev[j].activated;
                neurons[i].val = neurons[i].activated = sum / kernelWidth / kernelHeight;
            }
        }

        public override void backPropagate (double learningRate) {
            for (int i = 0; i < size; i++) {
                Debug.Assert(neurons[i].size == kernelHeight * kernelWidth);
                for (int j = 0; j < neurons[i].size; j++)
                    neurons[i].prev[j].error = neurons[i].error / kernelWidth / kernelWidth;
                neurons[i].error = 0;
            }
        }

        public override String ToString () {
            StringBuilder sb = new StringBuilder();
            sb.Append(String.Format("{0}\n{1} {2}", this.GetType().FullName, kernelHeight, kernelWidth));
            return sb.ToString();
        }
    }
}
