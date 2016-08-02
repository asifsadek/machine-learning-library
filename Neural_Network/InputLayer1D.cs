using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning.Neural_Network {

    public class InputLayer1D : Layer1D {

        public InputLayer1D (int size) {
            this.size = size;
            this.neurons = new Neuron[size];

            for (int i = 0; i < size; i++)
                neurons[i] = new Neuron(0);
        }

        public InputLayer1D (StreamReader reader, Layer prev) {
            string[] data = reader.ReadLine().Split();
            this.size = int.Parse(data[0]);

            this.neurons = new Neuron[size];

            for (int i = 0; i < size; i++)
                neurons[i] = new Neuron(0);

            BindTo(ref prev);
        }

        // input layers will never have a previous layer
        public override void BindTo (ref Layer layer) {
            prevLayer = null;
        }

        public void forwardPropagate (double[] val) {
            Debug.Assert(size == val.GetLength(0));
            for (int i = 0; i < size; i++) {
                neurons[i].val = val[i];
            }
        }

        // input layer has nothing to forward propagate
        public override void forwardPropagate () {
            for (int i = 0; i < size; i++)
                neurons[i].activated = neurons[i].val;
        }

        // back propagation ends at the input layer
        public override void backPropagate (double learningRate) { }

        public override String ToString () {
            StringBuilder sb = new StringBuilder();
            sb.Append(String.Format("{0}\n{1}", this.GetType().FullName, size));
            return sb.ToString();
        }
    }
}
