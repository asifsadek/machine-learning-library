using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning.Neural_Network {

    public class InputLayer2D : Layer2D {

        public InputLayer2D (int depth, int width, int height) {
            this.depth = depth;
            this.width = width;
            this.height = height;
            this.size = depth * width * height;
            this.neurons = new Neuron[size];

            for (int i = 0; i < size; i++)
                neurons[i] = new Neuron(0);
        }

        public InputLayer2D (StreamReader reader, Layer prev) {
            string[] data = reader.ReadLine().Split();
            this.depth = int.Parse(data[0]);
            this.width = int.Parse(data[1]);
            this.height = int.Parse(data[2]);
            this.size = depth * width * height;

            this.neurons = new Neuron[size];

            for (int i = 0; i < size; i++)
                neurons[i] = new Neuron(0);

            BindTo(ref prev);
        }

        // input layers will never have a previous layer
        public override void BindTo (ref Layer layer) {
            prevLayer = null;
        }

        public void forwardPropagate (double[,,] input) {
            Debug.Assert(input.GetLength(0) * input.GetLength(1) * input.GetLength(2) == size);
            for (int i = 0; i < depth; i++)
                for (int j = 0; j < width; j++)
                    for (int k = 0; k < height; k++)
                        neurons[i * width * height + j * height + k].val = input[i, j, k];
        }

        public override void forwardPropagate () {
            for (int i = 0; i < size; i++)
                neurons[i].activated = neurons[i].val;
        }

        public override void backPropagate (double learningRate) { }

        public override String ToString () {
            StringBuilder sb = new StringBuilder();
            sb.Append(String.Format("{0}\n{1} {2} {3}", "Machine_Learning.InputLayer2D", depth, width, height));
            return sb.ToString();
        }
    }
}
